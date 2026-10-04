"""STEP 8 (document section 6) - Final integrated explainable output.

Instead of "Malaria risk: High" the system now returns, for every patch and every scene:
    predicted malaria-risk level + P(high-risk) + decision margin          (Step 3)
    risk heatmap and predicted high-risk region overlay                    (Step 3)
    Grad-CAM explanation                                                   (Step 4)
    SHAP feature contributions (NDVI / NDWI, signed, in logit units)       (Step 5)
    CBAM attention map                                                     (Step 6)
    fused explanation + "how many methods agree" + support level + fidelity (Step 7)
    a plain-language explanation generated from the numbers above

Needs no model / TensorFlow: it only combines the saved outputs of Steps 2-7.
Run:  python -m src.step8_final
Env:  FUSE_VARIANT=all3|gc_shap   which Step-7 fusion is shown (default all3 = Grad-CAM + SHAP + CBAM, as in the document)
      N_EXAMPLES=6                number of per-patch explanation panels

Outputs (outputs/):
  step8_patch_<id>_<TP|FN|TN|FP>.png     one integrated explanation panel per example patch
  step8_final_scene_<month>.png          integrated scene explanation (risk map, 3 explanations, fusion, feature contributions)
  step8_final_<month>.tif                GeoTIFF (9 bands) for QGIS: P(high), predicted high-risk, Grad-CAM, SHAP, CBAM, fused,
                                         #methods agreeing, NDVI contribution, NDWI contribution
  step8_explanations_test.csv            one row per unseen (test) patch with every number above
  step8_report.json / step8_report.md    machine- and human-readable report (texts, metrics, caveats)
"""
import os, json, csv, textwrap
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import OUT_DIR
from . import xai_utils as U
from . import validation_core as V

VARIANT = os.environ.get("FUSE_VARIANT", "all3")
N_EX = int(os.environ.get("N_EXAMPLES", 6))
Q = 0.2

ctx = U.load_context()
PIX = ctx.PATCH * ctx.PATCH
X, y, split, valid, hrm, meta = ctx.X, ctx.y, ctx.split, ctx.valid, ctx.hrm, ctx.meta
THR, MONTHS, PATCH = ctx.THR, ctx.MONTHS, ctx.PATCH
LOGIT_THR = V.logit_of(THR)
M3 = ctx.M3
M7 = json.load(open(OUT_DIR / "step7_metrics.json"))
masks = U.env_masks(ctx)
te = split == 2; idx_te = np.where(te)[0]

cam = np.load(OUT_DIR / "step4_gradcam_patches.npz")["cam"].astype("float32")
d5 = np.load(OUT_DIR / "step5_shap_patches.npz")
shap_ev, logit, base, fsum = d5["evidence"].astype("float32"), d5["logit"].astype("float32"), float(d5["base_value"]), d5["feature_sum"]
prob = d5["prob"].astype("float32")
cbam = np.load(OUT_DIR / "step6_cbam_patches.npz")[("spatial_map" if M7["settings"]["cbam_map"] == "spatial" else "energy_map")].astype("float32")
d7 = np.load(OUT_DIR / "step7_validation_patches.npz")
fused = d7["fused_all3" if VARIANT == "all3" else "fused_gc_shap"].astype("float32")
del_top = d7["del_top20_all3" if VARIANT == "all3" else "del_top20_gc_shap"]
count = d7["count"]
VNAME = "Grad-CAM + SHAP + CBAM" if VARIANT == "all3" else "Grad-CAM + SHAP"
print(f"Step 8 | fusion shown: {VNAME} | test patches {len(idx_te)} | threshold {THR:.4f}")


# ------------------------------------------------------------------ per-patch integrated records
def enrich_for(i):
    st = U.focus_stats(fused, ctx, masks, [i])
    return {k: (float(v[0]) if len(v) else float("nan")) for k, v in st.items()}


def record(i):
    p = float(prob[i]); mg = V.decision_margin(p, THR); iou = d7["iou"][:, i]
    r = dict(idx=int(i), month=MONTHS[meta[i, 0]], row=int(meta[i, 1]), col=int(meta[i, 2]), split=("train", "val", "test")[split[i]],
             y_proxy=int(y[i]), prob=p, thr=THR, margin=mg, level=V.risk_level(p, THR), band=V.confidence_band(mg),
             base=base, ndvi_phi=float(fsum[i, 0]), ndwi_phi=float(fsum[i, 1]), resid=float(logit[i] - base - fsum[i].sum()),
             enrich=enrich_for(i), iou_gs=float(iou[0]), iou_gc=float(iou[1]), iou_sc=float(iou[2]),
             support=str(d7["support"][i]), n_pairs=int(d7["n_pairs"][i]), del_top=float(del_top[i]), del_rand=float(d7["del_rand20"][i]))
    r["dominant_feature"] = "NDVI" if abs(r["ndvi_phi"]) >= abs(r["ndwi_phi"]) else "NDWI"
    r["text"] = V.explain_text(r)
    return r


recs = {int(i): record(i) for i in idx_te}
cols = ["idx", "month", "row", "col", "y_proxy", "prob", "level", "margin", "band", "ndvi_phi", "ndwi_phi", "dominant_feature",
        "enrich_high_risk", "enrich_vegetated", "enrich_moist", "enrich_near_water", "iou_gradcam_shap", "iou_gradcam_cbam",
        "iou_shap_cbam", "support", "n_pairs_agreeing", "logit_drop_delete_top20", "logit_drop_delete_random20"]
with open(OUT_DIR / "step8_explanations_test.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(cols)
    for r in recs.values():
        e = r["enrich"]
        w.writerow([r["idx"], r["month"], r["row"], r["col"], r["y_proxy"], f"{r['prob']:.4f}", r["level"], f"{r['margin']:.3f}", r["band"],
                    f"{r['ndvi_phi']:.3f}", f"{r['ndwi_phi']:.3f}", r["dominant_feature"], f"{e['high_risk']:.3f}", f"{e['vegetated']:.3f}",
                    f"{e['moist']:.3f}", f"{e['near_water']:.3f}", f"{r['iou_gs']:.3f}", f"{r['iou_gc']:.3f}", f"{r['iou_sc']:.3f}",
                    r["support"], r["n_pairs"], f"{r['del_top']:.3f}", f"{r['del_rand']:.3f}"])

hi = [r for r in recs.values() if r["level"] == "HIGH"]
PATCH_SUM = dict(n_test=len(recs), n_predicted_high=len(hi),
                 dominant_feature_among_predicted_high={f: sum(1 for r in hi if r["dominant_feature"] == f) for f in ("NDVI", "NDWI")},
                 mean_ndvi_contribution_pred_high=float(np.mean([r["ndvi_phi"] for r in hi])) if hi else None,
                 mean_ndwi_contribution_pred_high=float(np.mean([r["ndwi_phi"] for r in hi])) if hi else None,
                 support_pred_high={s: sum(1 for r in hi if r["support"] == s) for s in ("strong", "partial", "weak")},
                 fraction_pred_high_where_deleting_top20_beats_random=float(np.mean([r["del_top"] > r["del_rand"] for r in hi])) if hi else None)
print("patch summary:", PATCH_SUM)

# ------------------------------------------------------------------ figures: one integrated panel per example patch
def overlay(ax, base_img, c, cmap, vmin, vmax, title):
    ax.imshow(base_img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.imshow(np.ma.masked_less(c, 0.15), cmap="jet", alpha=0.55, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9); ax.axis("off")


def tag_of(i):
    pred = recs[i]["level"] == "HIGH"
    return ("TP" if y[i] else "FP") if pred else ("FN" if y[i] else "TN")


def patch_panel(i, path):
    r = recs[i]; v = valid[i]; nd = np.ma.masked_where(~v, X[i][..., 0]); nw = np.ma.masked_where(~v, X[i][..., 1])
    fig = plt.figure(figsize=(17, 8.6)); gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.12)
    a = [fig.add_subplot(gs[0, k]) for k in range(5)] + [fig.add_subplot(gs[1, k]) for k in range(3)]
    a[0].imshow(nd, cmap="YlGn", vmin=0, vmax=.7); a[0].set_title("NDVI (vegetation)", fontsize=9); a[0].axis("off")
    a[1].imshow(nw, cmap="BrBG", vmin=-1, vmax=1); a[1].set_title("NDWI (water / moisture)", fontsize=9); a[1].axis("off")
    overlay(a[2], nd, cam[i], "YlGn", 0, .7, "Grad-CAM: where the model looks")
    overlay(a[3], nd, shap_ev[i], "YlGn", 0, .7, "SHAP evidence for high risk")
    overlay(a[4], nd, cbam[i], "YlGn", 0, .7, "CBAM spatial attention")
    overlay(a[5], nd, fused[i], "YlGn", 0, .7, f"Fused explanation ({VNAME})")
    if hrm[i].any(): a[5].contour(hrm[i], levels=[0.5], colors="white", linewidths=1.2)
    a[5].contour(np.where(v, count[i], 0) >= 2, levels=[0.5], colors="cyan", linewidths=1.2)
    a[5].text(0.01, -0.07, "white = proxy high-risk zone | cyan = >=2 methods agree", transform=a[5].transAxes, fontsize=7)
    im = a[6].imshow(np.ma.masked_where(~v, count[i]), cmap="viridis", vmin=0, vmax=3); a[6].set_title("# methods agreeing (top-20%)", fontsize=9); a[6].axis("off")
    # SHAP waterfall (Input -> Feature contribution -> Risk prediction)
    ax = a[7]; cur = base; ax.bar(0, base, color="#888"); levels = [0.0, base, float(logit[i]), LOGIT_THR]
    for k, val in enumerate((r["ndvi_phi"], r["ndwi_phi"], r["resid"]), 1):
        ax.bar(k, val, bottom=cur, color="#c0392b" if val >= 0 else "#2980b9"); cur += val; levels.append(cur)
    ax.bar(4, logit[i], color="k", alpha=.8); ax.axhline(LOGIT_THR, color="g", ls="--", lw=1)
    pad = 0.08 * (max(levels) - min(levels) + 1e-6); ax.set_ylim(min(levels) - pad, max(levels) + pad)
    ax.set_xticks(range(5), ["base", "NDVI", "NDWI", "resid.", "f(x)"], fontsize=8); ax.set_ylabel("logit (log-odds)", fontsize=8, labelpad=2)
    ax.set_title("SHAP: input -> contribution -> prediction\n(green dashed = decision threshold)", fontsize=8)
    tx = fig.add_axes([0.62, 0.06, 0.37, 0.4]); tx.axis("off")
    tx.text(0, 1, "\n\n".join(textwrap.fill(l, 64) for l in r["text"].split("\n")), va="top", fontsize=8.4, family="monospace")
    fig.suptitle(f"Patch #{i}  ({r['month']}, row {r['row']}, col {r['col']})  |  predicted {r['level']} (P={r['prob']:.2f})  |  "
                 f"proxy label = {'HIGH' if y[i] else 'low'}  [{tag_of(i)}]", fontsize=12, y=0.97)
    fig.subplots_adjust(top=0.9, bottom=0.05, left=0.02, right=0.99)
    plt.savefig(path, dpi=75); plt.close()


sel = [int(i) for i in U.pick_examples(ctx, prob)]
sel = ([sel[k] for k in (0, 1, 2, 5, 6, 7) if k < len(sel)] if N_EX <= 6 else sel)[:N_EX]
EX = []
for i in sel:
    fn_ = f"step8_patch_{i}_{tag_of(i)}.png"; patch_panel(i, OUT_DIR / fn_)
    EX.append(dict(idx=i, tag=tag_of(i), figure=fn_, **{k: recs[i][k] for k in ("month", "level", "prob", "margin", "band", "support", "text")}))
print("example panels:", [e["figure"] for e in EX])

# ------------------------------------------------------------------ scene-level integrated output
region = U.region_map(ctx)
SCENE_REP = {}
for m in MONTHS:
    s = ctx.scene[m]; sv = s["valid"]
    prb = np.load(OUT_DIR / f"step3_scene_prob_{m}.npz")["prob"].astype("float32")
    gcs = np.load(OUT_DIR / f"step4_scene_cam_{m}.npz")["cam"].astype("float32")
    L5 = np.load(OUT_DIR / f"step5_scene_shap_{m}.npz"); shs, pn, pw = (L5[k].astype("float32") for k in ("evidence", "ndvi_phi", "ndwi_phi"))
    cbs = np.load(OUT_DIR / f"step6_scene_cbam_{m}.npz")["spatial" if M7["settings"]["cbam_map"] == "spatial" else "energy"].astype("float32")
    L7 = np.load(OUT_DIR / f"step7_scene_fusion_{m}.npz"); cover = L7["cover"]; cnt = L7["count"].astype(np.float32)
    fs = L7["fused"].astype("float32")
    if VARIANT != "all3":
        fs = np.where(cover, V.fuse([gcs, shs], cover), np.nan).astype(np.float32)
    cnt = np.where(cover, cnt, np.nan)
    pred = np.nan_to_num(prb, nan=0) >= THR
    ok = cover & (region[m] == 3)                                         # unseen test blocks
    hi_, lo_ = ok & pred, ok & ~pred
    top = V.top_scene(fs, ok, Q)
    nd = s["ndvi"].astype("float32"); nw = s["ndwi"].astype("float32")
    rep = dict(month=m, test_px=int(ok.sum()), predicted_high_share=float(pred[ok].mean()),
               # SHAP is per pixel; x PIX = the contribution a 64x64 window would receive (same units as the patch-level SHAP)
               mean_ndvi_contribution_pred_high=float(np.nanmean(pn[hi_]) * PIX) if hi_.any() else None,
               mean_ndwi_contribution_pred_high=float(np.nanmean(pw[hi_]) * PIX) if hi_.any() else None,
               mean_ndvi_contribution_pred_low=float(np.nanmean(pn[lo_]) * PIX) if lo_.any() else None,
               mean_ndwi_contribution_pred_low=float(np.nanmean(pw[lo_]) * PIX) if lo_.any() else None,
               mean_ndvi_pred_high=float(nd[hi_].mean()) if hi_.any() else None, mean_ndvi_pred_low=float(nd[lo_].mean()) if lo_.any() else None,
               mean_ndwi_pred_high=float(nw[hi_].mean()) if hi_.any() else None, mean_ndwi_pred_low=float(nw[lo_].mean()) if lo_.any() else None,
               fused_top20_precision_for_pred_high=float(pred[top].mean()) if top.any() else None,
               fused_top20_recall_of_pred_high=float(top[hi_].mean()) if hi_.any() else None,
               consensus_2of3_share_of_test_land=float((ok & (cnt >= 2))[ok].mean()) if ok.any() else None,
               consensus_2of3_precision_for_pred_high=float(pred[ok & (cnt >= 2)].mean()) if (ok & (cnt >= 2)).any() else None)
    if hi_.any():
        dv = "NDVI" if abs(rep["mean_ndvi_contribution_pred_high"]) >= abs(rep["mean_ndwi_contribution_pred_high"]) else "NDWI"
        rep["dominant_feature_pred_high"] = dv
        base_rate = rep["predicted_high_share"]
        enr = rep["fused_top20_precision_for_pred_high"] / base_rate if base_rate > 0 and rep["fused_top20_precision_for_pred_high"] is not None else None
        rep["text"] = (
            f"{m}: {rep['predicted_high_share']:.1%} of the unseen land pixels are predicted HIGH malaria risk (P >= {THR:.3f}). "
            f"Inside that region SHAP attributes, per 64x64 window, a mean {rep['mean_ndvi_contribution_pred_high']:+.2f} (NDVI) and "
            f"{rep['mean_ndwi_contribution_pred_high']:+.2f} (NDWI) logit units (outside it: "
            f"{rep['mean_ndvi_contribution_pred_low']:+.2f} / {rep['mean_ndwi_contribution_pred_low']:+.2f}), so {dv} is the dominant driver. "
            f"Predicted-high pixels have mean NDVI {rep['mean_ndvi_pred_high']:.2f} vs {rep['mean_ndvi_pred_low']:.2f} and mean NDWI "
            f"{rep['mean_ndwi_pred_high']:.2f} vs {rep['mean_ndwi_pred_low']:.2f} elsewhere (decoded proxy values). "
            f"The fused top-20% explanation region lies inside the predicted high-risk region {rep['fused_top20_precision_for_pred_high']:.1%} of the time"
            + (f" ({enr:.1f}x the {base_rate:.1%} base rate)" if enr else "") +
            f" and covers {rep['fused_top20_recall_of_pred_high']:.1%} of the predicted high-risk pixels; "
            f"{rep['consensus_2of3_share_of_test_land']:.1%} of the test land is supported by at least two of the three methods.")
    else:
        rep["text"] = f"{m}: no unseen land pixels are predicted high-risk."
    SCENE_REP[m] = rep
    print(rep["text"])

    U.save_geotiff(OUT_DIR / f"step8_final_{m}.tif", [prb, pred.astype(np.float32), gcs, shs, cbs, fs, cnt, pn, pw],
                   ["P(high-risk)", "predicted high-risk (P>=threshold)", "Grad-CAM (0-1)", "SHAP evidence (0-1)", "CBAM spatial attention (0-1)",
                    f"Fused explanation: {VNAME} (0-1)", "# methods agreeing (0-3)", "SHAP NDVI contribution (logit)", "SHAP NDWI contribution (logit)"], ctx)

    # ---- integrated scene figure
    def ov(ax, c, title):
        ax.imshow(np.ma.masked_where(~sv, nd), cmap="YlGn", vmin=0, vmax=.7)
        ax.imshow(np.ma.masked_invalid(c), cmap="jet", alpha=0.55, vmin=0, vmax=1)
        ax.contour(pred & cover, levels=[0.5], colors="white", linewidths=.5); ax.set_title(title, fontsize=10); ax.axis("off")
    lim = float(np.nanpercentile(np.abs(np.concatenate([pn[cover & np.isfinite(pn)], pw[cover & np.isfinite(pw)]])), 99))
    fig, ax = plt.subplots(2, 4, figsize=(27, 13.5))
    vmax = float(np.nanpercentile(prb[cover], 99)) if cover.any() else 1.0
    im = ax[0, 0].imshow(np.ma.masked_where(~cover, np.nan_to_num(prb)), cmap="YlOrRd", vmin=0, vmax=max(vmax, 1e-3))
    ax[0, 0].contour(pred & cover, levels=[0.5], colors="cyan", linewidths=.6)
    ax[0, 0].set_title(f"Predicted malaria-risk map P(high)  (cyan = predicted high-risk, P >= {THR:.3f})", fontsize=10); ax[0, 0].axis("off")
    plt.colorbar(im, ax=ax[0, 0], fraction=.03)
    ov(ax[0, 1], gcs, "Grad-CAM: where the model looks"); ov(ax[0, 2], shs, "SHAP evidence for high risk"); ov(ax[0, 3], cbs, "CBAM spatial attention")
    ov(ax[1, 0], fs, f"Fused explanation ({VNAME})  | white = predicted high-risk")
    im = ax[1, 1].imshow(np.ma.masked_invalid(cnt), cmap="viridis", vmin=0, vmax=3); plt.colorbar(im, ax=ax[1, 1], fraction=.03)
    ax[1, 1].set_title("# methods whose top-20% contains the pixel (0-3)", fontsize=10); ax[1, 1].axis("off")
    for k, (arr, nm) in enumerate(((pn, "NDVI"), (pw, "NDWI"))):
        im = ax[1, 2 + k].imshow(np.ma.masked_where(~cover | ~np.isfinite(arr), arr), cmap="bwr", vmin=-lim, vmax=lim)
        ax[1, 2 + k].set_title(f"SHAP {nm} contribution (red = pushes risk up, blue = down)", fontsize=10); ax[1, 2 + k].axis("off")
        plt.colorbar(im, ax=ax[1, 2 + k], fraction=.03)
    fig.suptitle("\n".join(textwrap.wrap(rep["text"], 230)), fontsize=11, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig(OUT_DIR / f"step8_final_scene_{m}.png", dpi=50); plt.close()

# ------------------------------------------------------------------ report
cav = ["Risk labels are PROXY labels built in Step 2 (distance-to-water, moisture and vegetation), not observed malaria cases: "
       "the explanations show what the model uses to reproduce that proxy.",
       "NDVI / NDWI are decoded from colour-rendered GeoTIFFs (decode.py): values are an ordered proxy, not calibrated indices.",
       "Only two months (2025-04, 2025-05) and one region are available; test blocks are few, so per-method differences should be read as indicative.",
       "The SHAP evidence map is computed from input gradients of the same logit that the deletion test perturbs, so it is favoured by "
       "that test; Grad-CAM and CBAM operate on a coarse 16x16 feature grid. CBAM attention is a learned gate, not an attribution of the output, "
       "and agrees only weakly with Grad-CAM, so it adds the least to the fusion."]
rep_json = dict(fusion_variant=VNAME, threshold=THR, prediction_performance=dict(patch_test=M3["patch_test"], pixel_test=M3["pixel_test"]),
                patch_summary=PATCH_SUM, scene=SCENE_REP, examples=EX, caveats=cav,
                step7_summary=M7["summary"])
json.dump(rep_json, open(OUT_DIR / "step8_report.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

pt, px = M3["patch_test"], M3["pixel_test"]
dg = M7["fidelity_deletion"]; ag = M7["spatial_agreement"]["predicted_high"]; st = M7["stability"]
md = [f"# Final explainable output (Step 8)\n",
      f"Fusion shown: **{VNAME}**. Decision threshold P(high-risk) >= {THR:.3f}.\n",
      "## 1. What the system returns\n",
      "| Component | Source |\n|---|---|",
      "| Predicted malaria-risk level, P(high-risk), decision margin | Step 3 model |",
      "| Risk heatmap + predicted high-risk region overlay | Step 3 scene probabilities |",
      "| Grad-CAM explanation (where the model looks) | Step 4 |",
      "| SHAP feature contributions (NDVI, NDWI; signed, logit units) | Step 5 |",
      "| CBAM spatial attention map | Step 6 |",
      "| Fused explanation, #methods agreeing, support level, deletion fidelity | Step 7 |",
      "| Plain-language explanation | generated from the numbers above |\n",
      "## 2. Prediction performance vs explanation quality (kept separate)\n",
      f"- Prediction (unseen test blocks): patch precision {pt['precision']:.2f}, recall {pt['recall']:.2f}, F1 {pt['f1']:.2f}, AUC {pt['auc']:.2f}; "
      f"pixel IoU {px['iou']:.2f} (precision {px['precision']:.2f}, recall {px['recall']:.2f}). The XAI layer does not change these.",
      f"- Spatial agreement (predicted-high test patches, mean top-20% IoU; chance {M7['spatial_agreement']['chance_top20_iou']:.2f}): "
      + ", ".join(f"{k} {v['top20_iou']['mean']:.2f}" for k, v in ag.items() if isinstance(v, dict) and 'top20_iou' in v) + ".",
      f"- Stability under {st['noise_level_x_feature_sd']}x-sd input noise (top-20% IoU, clean vs noisy): Grad-CAM {st['gradcam']['top20_iou']['mean']:.2f}, "
      f"SHAP {st['shap']['top20_iou']['mean']:.2f} (its own Monte-Carlo noise floor {st['shap_noise_floor_same_input_new_seed']['top20_iou']['mean']:.2f}), "
      f"CBAM {st['cbam']['top20_iou']['mean']:.2f}, fused {st['fused_all3']['top20_iou']['mean']:.2f}.",
      "- Fidelity (deleting each method's top pixels, predicted-high test patches, mean logit drop over 5-50% deleted; random-pixel baseline "
      f"{dg['gradcam']['predicted_high']['aopc_random']:.2f}): "
      + ", ".join(f"{k} {dg[k]['predicted_high']['aopc_top']:.2f}" for k in ("gradcam", "shap", "cbam", "fused_gc_shap", "fused_all3")) + ".\n",
      "## 3. Scene-level explanations\n"]
for m in MONTHS:
    md += [f"### {m}\n", SCENE_REP[m]["text"] + "\n", f"![scene {m}](step8_final_scene_{m}.png)\n", f"GeoTIFF for QGIS: `step8_final_{m}.tif`\n"]
md += ["## 4. Example patch explanations (unseen test blocks)\n"]
for e in EX:
    md += [f"### Patch #{e['idx']} [{e['tag']}] - {e['month']}\n", "```\n" + e["text"] + "\n```\n", f"![patch {e['idx']}]({e['figure']})\n"]
md += ["## 5. Caveats\n"] + [f"- {c}" for c in cav]
open(OUT_DIR / "step8_report.md", "w").write("\n".join(md) + "\n")
print("done")