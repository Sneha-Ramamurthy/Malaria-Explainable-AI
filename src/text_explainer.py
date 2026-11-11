"""STEP 9 - Text explainable-AI layer.

Turns the numbers produced by Steps 3-8 into written explanations, in four styles:

    brief      one line
    plain      short, jargon-free, for a non-technical audience
    standard   sectioned: Prediction / Input / Why / Where / How much to trust it / Caveat
    technical  the same, with every quantity (logits, IoU vs chance, deletion test, residual)

Design rules (so the text can be trusted):
  * rule-based, NOT a language model: every sentence is a template filled with a number from the saved outputs,
    so it cannot invent facts;
  * direction words (raised / lowered) always follow the sign of the number;
  * weak evidence is worded as weak (borderline margin, explanation methods disagree, deletion no better than random);
  * the proxy-label caveat is always present: the model predicts an ENVIRONMENTAL proxy, not malaria cases.

Needs no TensorFlow / model.  Inputs (outputs/): step8_explanations_test.csv, step8_report.json
(optional extras that add detail when present: step2_patches.npz, step5_shap_patches.npz).

Command line (project root):
    python -m src.text_explainer                         # write outputs/text_explanations.md + .json for every test patch
    python -m src.text_explainer --list                  # which patches can be explained
    python -m src.text_explainer --patch 404             # standard explanation of patch 404
    python -m src.text_explainer --patch 404 --style plain|brief|technical|all
    python -m src.text_explainer --scene 2025-04 --style plain
    python -m src.text_explainer --summary               # one-paragraph overview of the whole system
    python -m src.text_explainer --patch 404 --json      # machine-readable
"""
import argparse
import csv
import json
import math
from pathlib import Path

STYLES = ("brief", "plain", "standard", "technical")
CHANCE_IOU = 0.2 / (2 - 0.2)            # expected top-20% IoU of two independent random maps (~0.111)
PROXY_CAVEAT = ("The 'risk' is an environmental proxy built from distance to water, moisture and vegetation; "
                "it is not a count or forecast of malaria cases.")

_FLOAT_COLS = ("prob", "margin", "ndvi_phi", "ndwi_phi", "enrich_high_risk", "enrich_vegetated", "enrich_moist",
               "enrich_near_water", "iou_gradcam_shap", "iou_gradcam_cbam", "iou_shap_cbam",
               "logit_drop_delete_top20", "logit_drop_delete_random20")
_INT_COLS = ("idx", "row", "col", "y_proxy", "n_pairs_agreeing")


# ============================================================================================ small helpers
def _ok(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _f(x, nd=2, signed=False, na="n/a"):
    if not _ok(x):
        return na
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def _ordinal_pct(p):
    """'76%', but never a misleading '100%' / '0%' for a percentile."""
    if p >= 99.5:
        return "more than 99%"
    if p < 0.5:
        return "fewer than 1%"
    return f"{p:.0f}%"


def _driver_names(r):
    """(dominant, other) as ('NDVI'|'NDWI', ...) by absolute SHAP contribution."""
    return ("NDVI", "NDWI") if abs(r["ndvi_phi"]) >= abs(r["ndwi_phi"]) else ("NDWI", "NDVI")


_PLAIN_NAME = {"NDVI": "vegetation", "NDWI": "moisture/water"}


def outcome_tag(r):
    """TP / FP / FN / TN of the model's HIGH call against the proxy label."""
    pred = r["level"] == "HIGH"
    return ("TP" if r["y_proxy"] else "FP") if pred else ("FN" if r["y_proxy"] else "TN")


def _level_phrase(level):
    return {"HIGH": "high", "ELEVATED": "elevated (below the high-risk cut-off but not low)", "LOW": "low"}[level]


def _band_phrase(band):
    return {"strong": "a clear-cut call",
            "moderate": "a moderately confident call",
            "borderline": "a borderline call, close to the cut-off, so treat it with caution"}[band]


def _support_phrase(r, plain=False):
    s, n = r["support"], r["n_pairs_agreeing"]
    if plain:
        return {"strong": "The three ways of looking inside the model (Grad-CAM, SHAP, CBAM) largely agree on which part of the area matters.",
                "partial": "The three ways of looking inside the model only partly agree on which part of the area matters.",
                "weak": "The three ways of looking inside the model (Grad-CAM, SHAP, CBAM) do not agree on which part of the area matters, so treat the highlighted spots with caution."}[s]
    return {"strong": f"strong support ({n} of 3 method pairs agree well above chance)",
            "partial": f"partial support ({n} of 3 method pairs agree well above chance)",
            "weak": f"weak support (no method pair agrees well above chance)"}[s]


def _fidelity(r):
    """(verdict, text) from the deletion test: does removing the highlighted region matter more than random?"""
    top, rnd = r["logit_drop_delete_top20"], r["logit_drop_delete_random20"]
    if not (_ok(top) and _ok(rnd)):
        return None, None
    if top > max(1.5 * rnd, rnd + 0.25) and top > 0:
        return "matters", (top, rnd)
    return "not_better", (top, rnd)


def _flips(r):
    """True if deleting the highlighted region would push a HIGH prediction below the threshold."""
    top = r["logit_drop_delete_top20"]
    return r["level"] == "HIGH" and _ok(top) and _ok(r["margin"]) and top >= r["margin"]


def _enrich_items(r, thr_hi=1.3, thr_lo=0.77):
    """[(label, value, 'concentrated'|'avoided')] for environmental classes the explanation clearly over/under-uses."""
    names = (("enrich_vegetated", "vegetated pixels"), ("enrich_moist", "moist (high-NDWI) pixels"),
             ("enrich_near_water", "pixels near water"), ("enrich_high_risk", "the proxy high-risk zone"))
    out = []
    for key, label in names:
        v = r.get(key)
        if _ok(v):
            if v >= thr_hi:
                out.append((label, v, "concentrated"))
            elif v <= thr_lo:
                out.append((label, v, "avoided"))
    return out


# ============================================================================================ data loading
def _num(v, as_int=False):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return int(x) if as_int else x


def load_patch_table(out_dir):
    """step8_explanations_test.csv -> {patch_idx: record dict}."""
    path = Path(out_dir) / "step8_explanations_test.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run Step 8 first (python run_all.py --steps 8).")
    recs = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            r = dict(row)
            for c in _FLOAT_COLS:
                r[c] = _num(row.get(c))
            for c in _INT_COLS:
                r[c] = _num(row.get(c), as_int=True)
            recs[r["idx"]] = r
    return recs


def load_report(out_dir):
    path = Path(out_dir) / "step8_report.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run Step 8 first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_extras(out_dir, recs):
    """Optional extra detail per patch from the .npz files; returns {} if they are not available."""
    out_dir = Path(out_dir)
    try:
        import numpy as np
        p2, p5 = out_dir / "step2_patches.npz", out_dir / "step5_shap_patches.npz"
        extras = {}
        if p2.exists():
            d = np.load(p2, allow_pickle=False)
            X, valid, split = d["X"].astype("float32"), d["valid"].astype(bool), d["split"]
            cnt = np.maximum(valid.sum((1, 2)), 1)
            mean_ndvi = (X[..., 0] * valid).sum((1, 2)) / cnt
            mean_ndwi = (X[..., 1] * valid).sum((1, 2)) / cnt
            tr = split == 0
            for i in recs:
                if i < len(X):
                    extras[i] = dict(mean_ndvi=float(mean_ndvi[i]), mean_ndwi=float(mean_ndwi[i]),
                                     ndvi_pct=float((mean_ndvi[tr] <= mean_ndvi[i]).mean() * 100),
                                     ndwi_pct=float((mean_ndwi[tr] <= mean_ndwi[i]).mean() * 100))
        if p5.exists():
            d = np.load(p5, allow_pickle=False)
            base, logit, fsum = float(d["base_value"]), d["logit"].astype("float64"), d["feature_sum"].astype("float64")
            for i in recs:
                if i < len(logit):
                    shift = logit[i] - base
                    resid = shift - fsum[i].sum()
                    extras.setdefault(i, {}).update(base=base, logit=float(logit[i]), resid=float(resid),
                                                    resid_share=float(abs(resid) / max(abs(shift), 1e-9)))
        return extras
    except Exception:                                      # noqa: BLE001 - extras are optional
        return {}


# ============================================================================================ patch explanations
def _input_sentence(ex, plain=False):
    if not ex or "mean_ndvi" not in ex:
        return None
    veg = ex["mean_ndvi"]
    veg_word = "sparse" if veg < 0.2 else ("moderate" if veg < 0.4 else "dense")
    wet = ex["ndwi_pct"]
    if plain:
        return (f"The area has {veg_word} vegetation (greener than {_ordinal_pct(ex['ndvi_pct'])} of the training areas) and is "
                f"{'wetter' if wet >= 50 else 'drier'} than {_ordinal_pct(wet if wet >= 50 else 100 - wet)} of them.")
    return (f"Mean NDVI {_f(ex['mean_ndvi'])} ({veg_word} vegetation; higher than {_ordinal_pct(ex['ndvi_pct'])} of training patches), "
            f"mean NDWI {_f(ex['mean_ndwi'])} (higher than {_ordinal_pct(wet)} of training patches). "
            "Values are decoded from colour renders, so they are an ordered proxy, not calibrated indices.")


def _why_sentence(r, plain=False):
    dom, oth = _driver_names(r)
    dv, ov = (r["ndvi_phi"], r["ndwi_phi"]) if dom == "NDVI" else (r["ndwi_phi"], r["ndvi_phi"])
    if not (_ok(dv) and _ok(ov)):
        return None
    up = lambda v: "raised" if v > 0 else "lowered"
    if plain:
        mv = lambda v: "up" if v > 0 else "down"
        s = f"Compared with a typical training area, the biggest push came from {_PLAIN_NAME[dom]}, which moved the score {mv(dv)}"
        s += f"; {_PLAIN_NAME[oth]} moved it {mv(ov)} by less." if abs(ov) < abs(dv) else "."
        return s
    return (f"{dom} {up(dv)} the risk score by {_f(dv, signed=True)} logit and {oth} {up(ov)} it by {_f(ov, signed=True)} "
            f"(SHAP, relative to the average training patch); {dom} is the dominant driver.")


def _where_sentence(r, plain=False):
    items = _enrich_items(r)
    if not items:
        return ("The highlighted region is not clearly concentrated on any particular land type." if plain else
                "The top-20% explanation region is close to the patch average for vegetation, moisture, water proximity and the proxy high-risk zone (all within 0.77-1.3x).")
    hi = [(l, v) for l, v, k in items if k == "concentrated"]
    lo = [(l, v) for l, v, k in items if k == "avoided"]
    if plain:
        parts = []
        if hi:
            parts.append("concentrates on " + _join([l for l, _ in hi]))
        if lo:
            parts.append("largely ignores " + _join([l for l, _ in lo]))
        return "The model " + " and ".join(parts) + "."
    parts = []
    if hi:
        parts.append("over-represents " + _join([f"{l} ({_f(v, 1)}x)" for l, v in hi]))
    if lo:
        parts.append("under-represents " + _join([f"{l} ({_f(v, 1)}x)" for l, v in lo]))
    return "The top-20% of the fused explanation " + " and ".join(parts) + " relative to the patch average (1.0 = chance)."


def _join(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _trust_sentences(r, ex, plain=False):
    out = [_support_phrase(r, plain=True)] if plain else []
    if not plain:
        out.append(f"Method agreement: Grad-CAM~SHAP IoU {_f(r['iou_gradcam_shap'])}, Grad-CAM~CBAM {_f(r['iou_gradcam_cbam'])}, "
                   f"SHAP~CBAM {_f(r['iou_shap_cbam'])} (chance {CHANCE_IOU:.2f}) -> {_support_phrase(r)}.")
    verdict, vals = _fidelity(r)
    if verdict == "matters":
        top, rnd = vals
        if plain:
            out.append("Covering up the highlighted area changes the score much more than covering up random spots, "
                       "so the highlighted area genuinely matters to the model.")
        else:
            out.append(f"Deletion test: removing the highlighted 20% lowers the logit by {_f(top, signed=True)} versus "
                       f"{_f(rnd, signed=True)} for the same number of random pixels, so the highlighted region is causally "
                       "important to the prediction.")
        if _flips(r):
            out.append("Removing it would be enough to bring this prediction below the high-risk cut-off."
                       if plain else f"The drop exceeds the decision margin ({_f(r['margin'], signed=True)}), so removing it would flip the call.")
    elif verdict == "not_better":
        top, rnd = vals
        out.append("Covering up the highlighted area changes the score no more than covering up random spots, so do not rely on the highlighted location."
                   if plain else f"Deletion test: removing the highlighted 20% lowers the logit by {_f(top, signed=True)}, not clearly more than "
                                 f"{_f(rnd, signed=True)} for random pixels, so the highlighted location is weakly supported.")
    if ex and "resid_share" in ex and ex["resid_share"] >= 1.0:
        if plain:
            out.append("The vegetation/moisture breakdown does not add up to this prediction (it points the other way overall), "
                       "so do not rely on that breakdown.")
        else:
            out.append(f"The NDVI/NDWI attributions do not account for this prediction: the unattributed residual ({ex['resid_share']:.0%} of the shift "
                       "from the baseline) is as large as the shift itself, so do not rely on the split between NDVI and NDWI.")
    elif ex and "resid_share" in ex and ex["resid_share"] > 0.25:
        if plain:
            out.append(f"About {ex['resid_share']:.0%} of the score change is not accounted for by the vegetation/moisture breakdown, "
                       "so treat that breakdown as approximate.")
        else:
            out.append(f"About {ex['resid_share']:.0%} of the shift from the baseline is not attributed to NDVI/NDWI "
                       "(estimation error of the SHAP approximation), so the split between NDVI and NDWI is approximate.")
    return out


def _label_sentence(r, plain=False):
    tag = outcome_tag(r)
    if plain:
        return {"TP": "This agrees with the environmental risk map used as the reference.",
                "TN": "This agrees with the environmental risk map used as the reference.",
                "FP": "However, the reference environmental map does not mark this area as high-risk, so this may be a false alarm.",
                "FN": "However, the reference environmental map marks this area as high-risk, so the model may have missed it."}[tag]
    return {"TP": "Agreement with the proxy label: the proxy score also marks this patch high-risk (true positive).",
            "TN": "Agreement with the proxy label: the proxy score also marks this patch as not high-risk (true negative).",
            "FP": "Disagreement with the proxy label: the proxy score does NOT mark this patch high-risk (possible false alarm).",
            "FN": "Disagreement with the proxy label: the proxy score marks this patch high-risk but the model did not (possible miss)."}[tag]


def explain_patch(r, style="standard", extras=None):
    """Return {'headline', 'text', 'style', 'idx', 'facts'} for one patch record."""
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    ex = extras or {}
    dom, _ = _driver_names(r)
    where = f"Patch #{r['idx']} ({r['month']}, row {r['row']}, col {r['col']})"
    headline = (f"{where}: {r['level']} risk (score {_f(r['prob'])}), driven mainly by "
                f"{_PLAIN_NAME[dom]} ({dom} {_f(r['ndvi_phi'] if dom == 'NDVI' else r['ndwi_phi'], signed=True)} logit); "
                f"explanation agreement: {r['support']}.")
    facts = dict(idx=r["idx"], month=r["month"], level=r["level"], prob=r["prob"], margin=r["margin"], band=r["band"],
                 dominant_feature=dom, support=r["support"], outcome=outcome_tag(r))

    if style == "brief":
        text = headline
    elif style == "plain":
        s = [f"{where}: the model rates this area's environmental malaria risk as {_level_phrase(r['level'])} "
             f"(score {_f(r['prob'])} out of 1, {_band_phrase(r['band'])})."]
        i = _input_sentence(ex, plain=True)
        if i:
            s.append(i)
        for part in (_why_sentence(r, True), _where_sentence(r, True)):
            if part:
                s.append(part)
        s += _trust_sentences(r, ex, plain=True)
        s += [_label_sentence(r, True), PROXY_CAVEAT]
        text = " ".join(s)
    else:
        tech = style == "technical"
        thr = None
        if _ok(r["prob"]) and _ok(r["margin"]):                       # recover threshold: margin = logit(p) - logit(thr)
            lp = math.log(min(max(r["prob"], 1e-7), 1 - 1e-7) / (1 - min(max(r["prob"], 1e-7), 1 - 1e-7)))
            thr = 1 / (1 + math.exp(-(lp - r["margin"])))
        pred = (f"Predicted environmental malaria-risk level: {r['level']}. P(high-risk) = {_f(r['prob'], 3 if tech else 2)}"
                + (f" against a decision threshold of {thr:.3f}" if thr is not None else "")
                + f" (margin {_f(r['margin'], signed=True)} logit: {_band_phrase(r['band'])}).")
        secs = [("Prediction", [pred])]
        inp = _input_sentence(ex)
        if inp:
            secs.append(("Input", [inp]))
        why = [_why_sentence(r)]
        if tech and ex and "base" in ex:
            why.append(f"Additivity: baseline {_f(ex['base'], signed=True)} + NDVI {_f(r['ndvi_phi'], signed=True)} + NDWI "
                       f"{_f(r['ndwi_phi'], signed=True)} + residual {_f(ex['resid'], signed=True)} = {_f(ex['logit'], signed=True)} (logit).")
        secs.append(("Why", [w for w in why if w]))
        secs.append(("Where the model looks", [_where_sentence(r)]))
        secs.append(("How much to trust it", _trust_sentences(r, ex) + [_label_sentence(r)]))
        secs.append(("Caveat", [PROXY_CAVEAT]))
        text = "\n".join(f"{h}: " + " ".join(x for x in body if x) for h, body in secs)
    return dict(idx=r["idx"], style=style, headline=headline, text=text, facts=facts)


# ============================================================================================ scene + overall text
def explain_scene(report, month, style="standard"):
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    sc = report["scene"].get(month)
    if sc is None:
        raise KeyError(f"month {month!r} not in report; available: {sorted(report['scene'])}")
    share = sc.get("predicted_high_share")
    n_px = sc.get("test_px")
    if sc.get("dominant_feature_pred_high") is None:
        return dict(month=month, style=style, text=f"{month}: no unseen land pixels are predicted high-risk. {PROXY_CAVEAT}")
    dom = sc["dominant_feature_pred_high"]
    nd_hi, nw_hi = sc["mean_ndvi_contribution_pred_high"], sc["mean_ndwi_contribution_pred_high"]
    nd_lo, nw_lo = sc["mean_ndvi_contribution_pred_low"], sc["mean_ndwi_contribution_pred_low"]
    prec, rec = sc["fused_top20_precision_for_pred_high"], sc["fused_top20_recall_of_pred_high"]
    cons = sc["consensus_2of3_share_of_test_land"]
    veg_hi, veg_lo = sc["mean_ndvi_pred_high"], sc["mean_ndvi_pred_low"]
    wet_hi, wet_lo = sc["mean_ndwi_pred_high"], sc["mean_ndwi_pred_low"]
    greener = veg_hi > veg_lo
    wetter = wet_hi > wet_lo
    if style == "brief":
        text = f"{month}: {share:.1%} of unseen land flagged high-risk, driven mainly by {_PLAIN_NAME[dom]}."
    elif style == "plain":
        text = (f"{month}: about {share:.0%} of the land the model had not seen during training is flagged as high environmental risk. "
                f"The main reason is {_PLAIN_NAME[dom]}. The flagged areas are {'greener' if greener else 'less green'} and "
                f"{'wetter' if wetter else 'drier'} than the rest. About {prec:.0%} of the map pixels that the explanation methods highlight lie "
                f"inside the flagged areas, and about {cons:.0%} of the land is supported by at least two of the three methods. "
                f"{PROXY_CAVEAT}")
    else:
        text = (f"{month}: {share:.1%} of the {n_px:,} unseen land pixels are predicted HIGH. Inside that region SHAP attributes, per 64x64 window, "
                f"a mean {_f(nd_hi, signed=True)} (NDVI) and {_f(nw_hi, signed=True)} (NDWI) logit units (outside it {_f(nd_lo, signed=True)} / "
                f"{_f(nw_lo, signed=True)}), so {dom} is the dominant driver. Predicted-high pixels have mean NDVI {_f(veg_hi)} vs {_f(veg_lo)} "
                f"and mean NDWI {_f(wet_hi)} vs {_f(wet_lo)} elsewhere (decoded proxy values). The fused top-20% explanation lies inside the predicted "
                f"high-risk region {prec:.1%} of the time"
                + (f" ({prec / share:.1f}x the {share:.1%} base rate)" if share and share > 0 else "")
                + f" and covers {rec:.1%} of the predicted high-risk pixels; {cons:.1%} of the test land is supported by at least two of the three "
                f"methods. {PROXY_CAVEAT}")
    return dict(month=month, style=style, text=text)


def explain_overall(report, table):
    """One paragraph about the whole system: performance, what drives predictions, how consistent the explanations are."""
    ps, perf = report["patch_summary"], report["prediction_performance"]["patch_test"]
    dom = ps["dominant_feature_among_predicted_high"]
    main = max(dom, key=dom.get)
    n_hi = ps["n_predicted_high"]
    sup = ps["support_pred_high"]
    tags = {t: sum(1 for r in table.values() if outcome_tag(r) == t) for t in ("TP", "FP", "FN", "TN")}
    return (f"On {ps['n_test']} unseen test patches the model flags {n_hi} as high environmental risk "
            f"(precision {perf['precision']:.2f}, recall {perf['recall']:.2f}, AUC {perf['auc']:.2f} against the proxy label; "
            f"{tags['TP']} true positives, {tags['FP']} false alarms, {tags['FN']} misses). For the flagged patches "
            f"{_PLAIN_NAME[main]} ({main}) is the dominant driver in {dom[main]} of {n_hi} cases. The explanation methods agree strongly "
            f"for {sup.get('strong', 0)} flagged patches, partly for {sup.get('partial', 0)} and weakly for {sup.get('weak', 0)}. "
            f"Deleting the highlighted region lowers the logit more than deleting random pixels in "
            f"{ps['fraction_pred_high_where_deleting_top20_beats_random']:.0%} of flagged patches. {PROXY_CAVEAT}")


# ============================================================================================ export + CLI
def export_all(out_dir, report, table, extras):
    out_dir = Path(out_dir)
    items = []
    md = ["# Text explanations (Step 9)\n", "## Overview\n", explain_overall(report, table) + "\n", "## Scenes\n"]
    for m in sorted(report["scene"]):
        md += [f"### {m}\n", explain_scene(report, m, "plain")["text"] + "\n", "Technical: " + explain_scene(report, m, "technical")["text"] + "\n"]
    md.append("## Patches (unseen test blocks)\n")
    for i in sorted(table):
        e = {s: explain_patch(table[i], s, extras.get(i)) for s in ("plain", "standard", "technical")}
        md += [f"### Patch #{i} [{outcome_tag(table[i])}] - {table[i]['month']}\n", "**" + e["standard"]["headline"] + "**\n",
               "Plain: " + e["plain"]["text"] + "\n", "```\n" + e["standard"]["text"] + "\n```\n"]
        items.append(dict(idx=i, headline=e["standard"]["headline"], facts=e["standard"]["facts"],
                          plain=e["plain"]["text"], standard=e["standard"]["text"], technical=e["technical"]["text"]))
    (out_dir / "text_explanations.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    scenes = {m: {s: explain_scene(report, m, s)["text"] for s in STYLES} for m in report["scene"]}
    (out_dir / "text_explanations.json").write_text(json.dumps(
        dict(overview=explain_overall(report, table), scenes=scenes, patches=items), indent=1), encoding="utf-8")
    return len(items)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Text explainable-AI layer", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--patch", type=int, help="patch index to explain (see --list)")
    ap.add_argument("--scene", help="month to explain, e.g. 2025-04")
    ap.add_argument("--style", default=None, help="brief | plain | standard | technical | all (default: standard)")
    ap.add_argument("--list", action="store_true", help="list explainable (test) patches")
    ap.add_argument("--n", type=int, default=15, help="rows for --list")
    ap.add_argument("--summary", action="store_true", help="overview paragraph of the whole system")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--out-dir", help="outputs folder (default: project outputs/)")
    a = ap.parse_args(argv)

    if a.out_dir:
        out_dir = Path(a.out_dir)
    else:
        from .config import OUT_DIR
        out_dir = OUT_DIR
    table, report = load_patch_table(out_dir), load_report(out_dir)
    extras = load_extras(out_dir, table)
    styles = STYLES if a.style == "all" else (a.style or "standard",)

    if a.list:
        rows = sorted(table.values(), key=lambda r: -(r["prob"] or 0))[:a.n]
        print(f"{'patch':>6} {'month':8} {'level':9} {'score':>6} {'vs proxy':>8} {'support':>8}")
        for r in rows:
            print(f"{r['idx']:>6} {r['month']:8} {r['level']:9} {_f(r['prob']):>6} {outcome_tag(r):>8} {r['support']:>8}")
        print(f"... {len(table)} explainable patches in total (showing the {len(rows)} highest-scoring).")
        return 0
    if a.summary:
        print(explain_overall(report, table))
        return 0
    if a.patch is not None:
        if a.patch not in table:
            print(f"Patch {a.patch} is not an explainable (test) patch. Use --list. Example ids: {sorted(table)[:8]}")
            return 2
        res = [explain_patch(table[a.patch], s, extras.get(a.patch)) for s in styles]
        if a.json:
            print(json.dumps(res, indent=1))
        else:
            for x in res:
                print(f"\n=== {x['style'].upper()} ===\n{x['text']}")
        return 0
    if a.scene:
        res = [explain_scene(report, a.scene, s) for s in styles]
        print(json.dumps(res, indent=1) if a.json else "\n\n".join(f"=== {x['style'].upper()} ===\n{x['text']}" for x in res))
        return 0
    n = export_all(out_dir, report, table, extras)
    print(f"Wrote {out_dir / 'text_explanations.md'} and text_explanations.json ({n} patches).\n")
    print(explain_overall(report, table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
