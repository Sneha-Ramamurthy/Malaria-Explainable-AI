"""Tests of the text explainable-AI layer (no TensorFlow, no model, no output files needed).
Run: python -m tests.test_text_explainer   (or pytest)"""
import json
import re
import tempfile
from pathlib import Path

from src import text_explainer as T


def rec(**kw):
    r = dict(idx=7, month="2025-04", row=64, col=32, y_proxy=1, prob=0.80, level="HIGH", margin=2.0, band="strong",
             ndvi_phi=3.0, ndwi_phi=1.0, dominant_feature="NDVI", enrich_high_risk=1.8, enrich_vegetated=1.6,
             enrich_moist=1.0, enrich_near_water=0.6, iou_gradcam_shap=0.30, iou_gradcam_cbam=0.25, iou_shap_cbam=0.20,
             support="strong", n_pairs_agreeing=3, logit_drop_delete_top20=4.0, logit_drop_delete_random20=0.5)
    r.update(kw); return r


def test_every_style_is_complete_and_never_leaks_python_junk():
    for style in T.STYLES:
        out = T.explain_patch(rec(), style)["text"]
        assert out and "None" not in out and not re.search(r"\bnan\b", out.lower()) and "{" not in out, (style, out)
        if style != "brief":
            assert "not a count or forecast of malaria cases" in out, f"{style} must carry the proxy caveat"


def test_direction_words_follow_the_sign_of_the_numbers():
    down = T.explain_patch(rec(ndvi_phi=-2.0, ndwi_phi=-0.5), "technical")["text"]
    assert "NDVI lowered the risk score by -2.00" in down and "NDWI lowered it by -0.50" in down and "raised the risk score" not in down
    up = T.explain_patch(rec(), "technical")["text"]
    assert "NDVI raised the risk score by +3.00" in up
    plain_down = T.explain_patch(rec(ndvi_phi=-2.0, ndwi_phi=0.5), "plain")["text"]
    assert "vegetation, which moved the score down" in plain_down and "moisture/water moved it up by less" in plain_down
    assert T.explain_patch(rec(ndvi_phi=0.4, ndwi_phi=-2.5), "brief")["text"].count("moisture/water (NDWI -2.50") == 1   # dominant = larger |phi|


def test_numbers_in_the_text_come_from_the_record():
    r = rec(prob=0.9371, margin=4.46, ndvi_phi=3.29, ndwi_phi=0.97, logit_drop_delete_top20=5.72, logit_drop_delete_random20=1.43)
    txt = T.explain_patch(r, "technical")["text"]
    for needle in ("0.937", "+4.46", "+3.29", "+0.97", "+5.72", "+1.43", "0.30", "0.25", "0.20", "0.11"):
        assert needle in txt, needle
    thr = re.search(r"threshold of ([0-9.]+)", txt)
    assert thr and 0 < float(thr.group(1)) < 1                     # recovered from prob and margin, not invented
    import math
    lp = math.log(0.9371 / (1 - 0.9371)); assert abs(float(thr.group(1)) - 1 / (1 + math.exp(-(lp - 4.46)))) < 1e-3


def test_outcome_tags_and_label_wording():
    assert T.outcome_tag(rec()) == "TP"
    assert T.outcome_tag(rec(y_proxy=0)) == "FP"
    assert T.outcome_tag(rec(level="LOW", y_proxy=1)) == "FN"
    assert T.outcome_tag(rec(level="ELEVATED", y_proxy=0)) == "TN"
    assert "false alarm" in T.explain_patch(rec(y_proxy=0), "plain")["text"]
    assert "may have missed it" in T.explain_patch(rec(level="LOW", y_proxy=1, prob=0.05), "plain")["text"]
    assert "(true positive)" in T.explain_patch(rec(), "standard")["text"]


def test_weak_evidence_is_worded_as_weak():
    t = T.explain_patch(rec(band="borderline", margin=0.1, support="weak", n_pairs_agreeing=0,
                            logit_drop_delete_top20=0.2, logit_drop_delete_random20=0.3), "plain")["text"]
    assert "borderline call" in t and "do not agree" in t and "no more than covering up random spots" in t
    assert "genuinely matters" not in t
    tech = T.explain_patch(rec(support="weak", n_pairs_agreeing=0, logit_drop_delete_top20=0.2, logit_drop_delete_random20=0.3), "technical")["text"]
    assert "weak support" in tech and "weakly supported" in tech


def test_flip_statement_only_when_a_high_call_would_really_flip():
    assert "flip the call" in T.explain_patch(rec(margin=2.0, logit_drop_delete_top20=4.0), "standard")["text"]
    assert "flip the call" not in T.explain_patch(rec(margin=5.0, logit_drop_delete_top20=4.0), "standard")["text"]
    assert "flip the call" not in T.explain_patch(rec(level="LOW", margin=-2.0, prob=0.02, y_proxy=0), "standard")["text"]


def test_missing_values_do_not_crash_or_print_none():
    r = rec(enrich_moist=None, enrich_vegetated=None, enrich_near_water=None, enrich_high_risk=None,
            logit_drop_delete_top20=None, logit_drop_delete_random20=None, iou_shap_cbam=None)
    for s in T.STYLES:
        out = T.explain_patch(r, s)["text"]
        assert "None" not in out and out
    assert "Deletion test" not in T.explain_patch(r, "technical")["text"]


def test_shap_residual_warnings_scale_with_severity():
    ok = T.explain_patch(rec(), "standard", dict(resid_share=0.10, base=-3.0, resid=0.3, logit=1.0))["text"]
    mid = T.explain_patch(rec(), "standard", dict(resid_share=0.40, base=-3.0, resid=1.0, logit=1.0))["text"]
    bad = T.explain_patch(rec(), "plain", dict(resid_share=1.22, base=-3.0, resid=1.0, logit=1.0))["text"]
    assert "not attributed" not in ok and "40%" in mid and "approximate" in mid
    assert "do not rely on that breakdown" in bad and "122%" not in bad
    add = T.explain_patch(rec(), "technical", dict(resid_share=0.1, base=-3.1, resid=1.55, logit=2.7))["text"]
    assert "baseline -3.10" in add and "residual +1.55" in add and "= +2.70" in add


def test_input_description_uses_percentiles_without_100_percent():
    ex = dict(mean_ndvi=0.45, mean_ndwi=-0.2, ndvi_pct=100.0, ndwi_pct=0.0)
    t = T.explain_patch(rec(), "plain", ex)["text"]
    assert "dense vegetation" in t and "more than 99%" in t and "100%" not in t and "drier than more than 99%" in t
    assert T._ordinal_pct(0.1) == "fewer than 1%" and T._ordinal_pct(76.4) == "76%"


def test_invalid_style_is_rejected():
    try:
        T.explain_patch(rec(), "shouty"); assert False
    except ValueError:
        pass


REPORT = dict(
    scene={"2025-04": dict(month="2025-04", test_px=100000, predicted_high_share=0.33, mean_ndvi_contribution_pred_high=2.0,
                           mean_ndwi_contribution_pred_high=0.5, mean_ndvi_contribution_pred_low=-1.0, mean_ndwi_contribution_pred_low=0.1,
                           mean_ndvi_pred_high=0.4, mean_ndvi_pred_low=0.2, mean_ndwi_pred_high=-0.3, mean_ndwi_pred_low=-0.4,
                           fused_top20_precision_for_pred_high=0.5, fused_top20_recall_of_pred_high=0.3,
                           consensus_2of3_share_of_test_land=0.1, consensus_2of3_precision_for_pred_high=0.7,
                           dominant_feature_pred_high="NDVI", text="")},
    patch_summary=dict(n_test=180, n_predicted_high=33, dominant_feature_among_predicted_high={"NDVI": 25, "NDWI": 8},
                       support_pred_high={"strong": 8, "partial": 23, "weak": 2}, fraction_pred_high_where_deleting_top20_beats_random=1.0),
    prediction_performance=dict(patch_test=dict(precision=0.6, recall=0.57, auc=0.84)))


def test_scene_and_overview_text():
    for s in T.STYLES:
        t = T.explain_scene(REPORT, "2025-04", s)["text"]
        assert "2025-04" in t and "None" not in t
    plain = T.explain_scene(REPORT, "2025-04", "plain")["text"]
    assert "about 33%" in plain and "greener and wetter" in plain and "50%" in plain
    assert "33.0%" in T.explain_scene(REPORT, "2025-04", "technical")["text"]
    try:
        T.explain_scene(REPORT, "1999-01"); assert False
    except KeyError:
        pass
    table = {1: rec(), 2: rec(y_proxy=0), 3: rec(level="LOW", y_proxy=1, prob=0.05), 4: rec(level="LOW", y_proxy=0, prob=0.02)}
    o = T.explain_overall(REPORT, table)
    assert "1 true positives, 1 false alarms, 1 misses" in o and "25 of 33" in o and "AUC 0.84" in o


def test_cli_end_to_end_on_a_temp_outputs_folder():
    import io, contextlib
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cols = ["idx", "month", "row", "col", "y_proxy", "prob", "level", "margin", "band", "ndvi_phi", "ndwi_phi", "dominant_feature",
                "enrich_high_risk", "enrich_vegetated", "enrich_moist", "enrich_near_water", "iou_gradcam_shap", "iou_gradcam_cbam",
                "iou_shap_cbam", "support", "n_pairs_agreeing", "logit_drop_delete_top20", "logit_drop_delete_random20"]
        r = rec(idx=404)
        (d / "step8_explanations_test.csv").write_text(",".join(cols) + "\n" + ",".join(str(r[c]) for c in cols) + "\n", encoding="utf-8")
        (d / "step8_report.json").write_text(json.dumps(REPORT), encoding="utf-8")
        run = lambda *a: (lambda b: (T.main(["--out-dir", str(d), *a]), b.getvalue()))(io.StringIO())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert T.main(["--out-dir", str(d), "--patch", "404", "--style", "plain"]) == 0
            assert T.main(["--out-dir", str(d), "--patch", "999"]) == 2
            assert T.main(["--out-dir", str(d), "--list"]) == 0
            assert T.main(["--out-dir", str(d), "--scene", "2025-04", "--style", "brief"]) == 0
            assert T.main(["--out-dir", str(d), "--summary"]) == 0
            assert T.main(["--out-dir", str(d)]) == 0                       # default = export everything
        out = buf.getvalue()
        assert "Patch #404" in out and "not an explainable" in out
        data = json.loads((d / "text_explanations.json").read_text(encoding="utf-8"))
        assert data["patches"][0]["idx"] == 404 and {"plain", "standard", "technical"} <= set(data["patches"][0])
        assert "### Patch #404 [TP]" in (d / "text_explanations.md").read_text(encoding="utf-8")


def test_works_on_real_step8_outputs_if_present():
    from src.config import OUT_DIR
    if not (Path(OUT_DIR) / "step8_explanations_test.csv").exists():
        return
    table, report = T.load_patch_table(OUT_DIR), T.load_report(OUT_DIR)
    ex = T.load_extras(OUT_DIR, table)
    for i, r in table.items():
        for s in T.STYLES:
            assert T.explain_patch(r, s, ex.get(i))["text"]
    for m in report["scene"]:
        assert T.explain_scene(report, m, "plain")["text"]


if __name__ == "__main__":
    import sys, traceback
    bad = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            try: v(); print("PASS ", k)
            except Exception: bad += 1; print("FAIL ", k); traceback.print_exc()
    sys.exit(1 if bad else 0)
