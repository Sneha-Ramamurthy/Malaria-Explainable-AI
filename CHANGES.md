# Changes made to Malaria-Explainable-AI

Only code, configs, tests and docs are included (no dataset / model / outputs). Copy these files over your project.

## A. Errors that stopped the pipeline (fixed)
| File | Problem | Fix |
|---|---|---|
| `src/step2_build_dataset.py` | `np.savez_compressed(..., valid=..., valid=...)` - repeated keyword = **SyntaxError**, Step 2 could not even start | Save `valid` (=observed pixels), `valid_in`, and `valid_risk` (=observed LAND, what the model uses) |
| `src/step1_data_audit.py` | `if _name_ == "_main_":` - `main()` never ran; the step silently did nothing | `__name__ == "__main__"` |
| `src/step3_train.py` | `iou`, `pixel_precision`, `pixel_recall`, `pixel_f1` written to metrics but **never computed** (NameError) | Pixel metrics computed on unseen test-block land pixels |
| `run_all.py` | Steps 7/8 pointed at `src.step7_validation_fusion` / `src.step8_final`, which don't exist | `src.step7_validation`, `src.step8_final_output` |
| `README.md` | Step 8 docstring/README gave wrong module names; test command `python tests/test_xai_core.py` could not work | Corrected |

## B. Silent bugs that gave wrong/inconsistent results (fixed)
| File | Problem | Fix |
|---|---|---|
| `xai_utils.py`, `step6`, `step7`, `step8` | Scene-level explanations used `valid` (includes inland water) while training patches zero all non-land pixels. Verified on your data: **270/270** scene windows differed from the training patch at the same location | New `U.scene_valid()` (land mask) used everywhere; now **0/270** differ |
| `step6_cbam.py` | `CACHE = bool(os.environ.get("CACHE"))` -> `CACHE=0` turned caching **on** | Same parsing as Steps 4/5 |
| `step4_gradcam.py` | `pred_contour.all() is False` is always False for a NumPy bool -> predicted-risk contour never drawn | `not pred_contour.all()` |
| `step5_shap.py` | "Feature dependence" plot had a `break`, drew NDVI only | Both NDVI and NDWI |
| `run_all.py` | Reported SUCCESS from stale outputs of an earlier run even if a step did nothing | Outputs must be newer than the step start (skipped when `CACHE=1`) |
| `step6/7/8` | `json.dump(..., open(...))` / `open().write()` never closed | `with open(...)` |
| `step8_final_output.py` | Crash if no finite SHAP values for colour limits | Guarded |

## C. Explainable-AI correctness
| Item | Problem | Fix / evidence |
|---|---|---|
| SHAP stability test (`step7`) | Noisy run reused the clean run's Monte-Carlo seed (10) while its "noise floor" used a different seed -> stability inflated, `shap_stability_above_noise_floor` true by construction. Toy demo: 0.93 (shared seed) vs 0.73 floor vs 0.74 (fixed) | Independent seed (12) for the noisy run |
| SHAP background (`step7`) | Claimed to match Step 5 but was not sorted -> different order | Identical to Step 5 (sorted, `min(N_BG, n_train)`) |
| SHAP completeness metric (`shap_explain.py`) | `rel_err_of_variance` compares error *variance* to target variance, not error size - it reported "0.02" while mean error was 25% of the effect | Added `relative_mean_abs_err`, `mean_signed_err`, `r2_sum_phi_vs_target` (old key kept) |
| Step 8 caveat | Hard-coded "CBAM agrees only weakly" regardless of results | Computed from Step 7 metrics |
| `explain_text` | Said "Predicted malaria risk" - the model predicts a proxy | "(environmental proxy)" |
| **Tests** | `tests/test_xai_core.py` held Step-2 dataset tests, nothing tested the XAI maths | Dataset tests -> `tests/test_dataset.py`; new **23 known-answer tests** in `test_xai_core.py` (all pass): SHAP recovers exact Shapley values for a linear model, completeness + convergence for a non-linear one, null-feature = 0, rank fusion, chance IoU constant, deletion test rewards a faithful map and punishes a wrong one, scene windows/stitching |

## D. Dependencies / housekeeping
- `requirements-lock.txt`: was UTF-16 + CRLF (breaks many tools) and listed **both** `keras` and `keras-nightly` (same package, conflicting) -> UTF-8, `keras-nightly` removed. It also pins `tensorflow_cpu==2.22.0rc0` (a release candidate) - consider a stable release.
- `requirements.txt`: states Keras 3 requirement (`tensorflow-cpu>=2.16`, `keras>=3.0`), adds `pytest`.
- `pytest.ini` added (`pythonpath = .`); `.gitignore` now also ignores `dataset/`.

## E. You MUST do on your machine (I could not run these: no TensorFlow/rasterio/shap here)
1. **Re-run Steps 2-8 from scratch** (`python run_all.py`, no `CACHE`). The shipped outputs/model are stale: current Step 2 yields 761 patches (403/196/162) vs 768 (430/158/180) in your shipped `step2_patches.npz`, and the shipped `step5_shap_patches.npz` predates the current Step 5 code.
2. Confirm `step5_metrics.json -> shap_library_crosscheck` shows a `pearson` near 1 (it previously failed with "No module named shap", so SHAP was never cross-checked against the library).
3. SHAP completeness error was ~25% of the effect at 100 samples: raise `SHAP_SAMPLES` (e.g. 300-500) until `relative_mean_abs_err` is small; likewise `STAB_SHAP_SAMPLES` (noise-floor IoU was only 0.55).
4. Shipped Step 5 showed NDVI/NDWI shares of 60.5%/39.5% *identical to 3 decimals* for all test, predicted-high and predicted-low patches - check this still holds after retraining; if so, investigate.
5. `python -m src.diag_model` to confirm the saved model reproduces on your install.

## F. Second round: improvements applied (need the re-run in section E to take effect)
| File | Change |
|---|---|
| `step2_build_dataset.py` | Positive-label cutoff (`FRAC_THR`) now computed from **training** patches only (was all patches incl. test). Verified on your data: train positive rate exactly 30%; val/test differ (38.8% / 43.8%) because of spatial heterogeneity |
| `step3_train.py` | Decision threshold = **Youden's J** maximised on validation (was "70th percentile of validation predictions", which forces ~30% predicted positives regardless of the model) |
| `step3_train.py` | Augmentation noise/offset applied **only to observed land pixels**; the validity mask is flipped/rotated with the patch (masked pixels stay exactly 0, as at inference) |
| `step3_train.py` | Scene windows built with the shared `U.scene_windows` (land-masked, edge-aligned). Verified: every old window kept, +43 / +16 windows, bottom/right edge strips now covered |
| `step3_train.py`, `src/train_utils.py` (new) | **95% block-bootstrap confidence intervals** of patch AUC/precision/recall/F1 saved as `patch_test_ci95_block_bootstrap` (resampling spatial blocks, not single patches); shown in `step8_report.md` |
| `step4_gradcam.py` | GeoTIFF is rewritten also when `CACHE=1` (previously missing if the file had been deleted) |
| `models.py` | Corrected misleading feature-map-size comment |
| tests | New `test_train_utils.py` (threshold + bootstrap) and `test_project_integrity.py` (run_all modules exist, all files parse, main guard, no water-inclusive mask, lock file encoding). Total: 23 + 3 + 5 tests, all passing; the TensorFlow-free ones run with `python -m tests.<name>` |

Not changed on purpose: pinned versions in `requirements-lock.txt` (can't verify offline; `tensorflow_cpu==2.22.0rc0` is a release candidate), and Step 4's duplicate `focus_stats` (works; refactor would add risk without a way to run TensorFlow here).

| `step5_shap.py`, `step7_validation.py` | `SHAP_SAMPLES` default 100 -> **300**, `STAB_SHAP_SAMPLES` 50 -> **150** (shipped results: ~25% completeness error and 0.55 noise-floor IoU). Step 5 now prints a WARNING if the relative completeness error is still > 10%. Slower: Step 5 takes roughly 3x longer; override with the env vars |

| `verify_changes.py` (new) | One-command proof-of-concept verifier with 3 levels (code/tests, Step 1-2 data checks, full-pipeline output checks). Tested: on your real data Levels A+B pass (13 checks, incl. 270/270 scene windows identical to training patches); on your old shipped outputs Level C correctly FAILS 4 checks that identify old-code behaviour |

| `src/text_explainer.py` (new, **Step 9**) | **Text explainable-AI layer**: brief / plain / standard / technical explanations per patch and per month, plus a one-paragraph system overview; CLI (`--patch`, `--scene`, `--list`, `--summary`, `--json`, default = export all to `outputs/text_explanations.md/.json`); wired into `run_all.py` (`--steps 9`); 13 tests (`tests/test_text_explainer.py`) + 2 verifier checks. Verified on your shipped outputs: 720 explanations (180 patches x 4 styles). Reproduces Step 8's numbers exactly and additionally reports the SHAP residual: in the shipped run the median patch has 28% of its score change unattributed and 32/180 patches have a residual >= 100% (flagged in the text as 'do not rely on the NDVI/NDWI split') |

## G. Limits that code cannot fix
- The CNN (AUC 0.84) does not beat the 5-feature logistic baseline (AUC 0.84) and the labels are computed from the same NDVI/NDWI inputs, so the explanations show reproduction of a formula, not malaria risk. Real case data would be needed to validate.
- Only two months and one region; the new confidence intervals will show how wide the uncertainty is.
