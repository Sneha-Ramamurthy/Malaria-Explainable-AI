# Malaria Risk Prediction - Explainable AI layer

A MobileNetV2 + CBAM patch classifier on Sentinel-2 NDVI / NDWI, explained with Grad-CAM, SHAP (expected gradients)
and CBAM attention, then validated and fused.

> **Important - what the model predicts.** The "risk" labels are *proxy* labels built in Step 2 from distance to
> water, moisture (NDWI) and vegetation (NDVI). They are **not observed malaria cases**. The explanations show what
> the model uses to reproduce that proxy; they are not epidemiological evidence.

## Setup
```
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # or requirements-lock.txt for the exact tested versions
```
Place the extracted `dataset/` folder (the Sentinel-2 `.tiff` files) inside this directory.

## Run everything
```
python run_all.py                  # all 8 steps, checks inputs/outputs of each step
python run_all.py --steps 3 4 5    # only some steps
```
or step by step (each needs the outputs of the earlier ones):
```
python -m src.step1_data_audit
python -m src.step2_build_dataset
python -m src.step3_train           # ~4 min on CPU; saves models/mobilenetv2_cbam.keras
python -m src.step4_gradcam         # ~6 min on CPU   (CACHE=1 reuses saved heatmaps)
python -m src.step5_shap            # SHAP_SAMPLES (default 300; ~3x slower than the old 100) / SCENE_SAMPLES / CACHE env vars
python -m src.step6_cbam            # CBAM channel + spatial attention vs Grad-CAM
python -m src.step7_validation      # agreement, stability, deletion fidelity, fusion
python -m src.step8_final_output    # integrated explanation + report (no TensorFlow needed)
python -m src.text_explainer        # Step 9: writes outputs/text_explanations.md/.json (no TensorFlow needed)
```
`CACHE=1` is the only way to enable caching (`CACHE=0` means off). **After changing any code or data in Steps 2-3,
re-run Steps 3-8 without `CACHE`**, otherwise later steps explain a stale model.

## Text explanations (Step 9)
Rule-based (no language model): every sentence is a template filled with a number from the saved outputs, so it cannot
invent facts. Wording always follows the sign of the numbers, weak evidence is worded as weak, and the "proxy, not malaria
cases" caveat is always included. Needs only Step 8 outputs.
```
python -m src.text_explainer --list                       # which patches can be explained
python -m src.text_explainer --patch 404                  # standard (sectioned) explanation
python -m src.text_explainer --patch 404 --style plain    # jargon-free; also: brief | technical | all
python -m src.text_explainer --scene 2025-04 --style plain
python -m src.text_explainer --summary                    # one paragraph about the whole system
python -m src.text_explainer --patch 404 --json           # machine-readable
python -m src.text_explainer                              # export every test patch -> outputs/text_explanations.md/.json
```
Per patch it states: the predicted level and score, what the input looks like (percentile vs training patches), what pushed
the score up/down (SHAP), where the model looks, how much to trust it (method agreement, deletion test, whether removing the
highlighted area would flip the call, how much of the score SHAP could not attribute) and whether it agrees with the proxy label.

## Proof that the fixes work (one command)
```
python verify_changes.py                # Level A (code + 31 tests) and whatever B/C data already exists; writes nothing
python verify_changes.py --run-data     # also RUNS Steps 1-2 on dataset/ (needs rasterio), then checks the fixes
python run_all.py                       # full pipeline (needs TensorFlow) ...
python verify_changes.py --level C      # ... then verify the explanation outputs (SHAP, Grad-CAM, stability, fidelity)
python verify_changes.py --json report.json
```
Each check prints PASS / WARN / FAIL / SKIP; the exit code is 1 if anything FAILS. SKIP means a prerequisite is missing
and says which command creates it.

## Tests
```
python -m tests.test_train_utils    # threshold selection + bootstrap CIs
python -m tests.test_project_integrity   # structural checks
python -m tests.test_xai_core      # no TensorFlow / model / pytest needed: known-answer tests of the SHAP,
                                   # fusion, agreement and deletion-test numerics
python -m pytest tests -q          # everything (tests/test_dataset.py needs Step 2 outputs)
python -m src.diag_model           # does THIS machine reproduce the trained model? (needs TensorFlow + Steps 3-5)
```

## Steps
1. Data audit
2. Build inputs: colour->index decoding, sea/inland-water split, PROXY risk labels, 64x64 patches
3. MobileNetV2+CBAM patch classifier + predicted high-risk map
4. Grad-CAM: patch + scene heatmaps, focus analysis, randomised-model check, GeoTIFF export
5. SHAP: per-pixel NDVI/NDWI contributions to the logit, completeness check, focus + Grad-CAM agreement,
   randomised-model check, optional `shap`-library cross-check, scene maps + GeoTIFF
6. CBAM attention: channel + spatial attention, comparison with Grad-CAM, randomised-model check, scene maps
7. Validation & fusion: pairwise agreement, input-noise stability, deletion fidelity, rank fusion
8. Final integrated explanation: per-patch text + figures, scene figures, 9-band GeoTIFF, `step8_report.md/json`
9. **Text explainable-AI layer** (`src/text_explainer.py`): written explanations in four styles (see below)

## Checking that the explanations are trustworthy
After a full run, read these in `outputs/` (they are also summarised in `step8_report.md`):

| Check | File / key | Healthy sign |
|---|---|---|
| SHAP adds up to the prediction | `step5_metrics.json` -> `completeness` | `relative_mean_abs_err` small (< ~0.1), `mean_signed_err` ~ 0, `r2_sum_phi_vs_target` close to 1. Raise `SHAP_SAMPLES` if not |
| SHAP agrees with the reference implementation | `step5_metrics.json` -> `shap_library_crosscheck` | a `pearson` value close to 1 (an `error` entry means it did not run - fix before trusting SHAP) |
| Explanations depend on the learned weights | `step4/5/6_metrics.json` -> randomised-model checks | trained vs random correlation low; trained enrichment > random |
| Explanations are stable | `step7_metrics.json` -> `stability` | clean-vs-noisy IoU close to the SHAP noise floor |
| Explanations are faithful | `step7_metrics.json` -> `fidelity_deletion` | deleting top pixels drops the logit more than random pixels |
