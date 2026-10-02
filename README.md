# Malaria Risk Prediction - Explainable AI layer.
Place the extracted `dataset/` folder inside this directory, then:
    pip install -r requirements.txt
    python -m src.step1_data_audit
    python -m src.step2_build_dataset
    python -m src.step3_train      # ~4 min on CPU; saves models/mobilenetv2_cbam.keras
    python -m src.step4_gradcam    # ~6 min on CPU (set CACHE=1 to reuse saved heatmaps)
    python -m src.step5_shap       # SHAP (expected gradients) - needs step 3 + 4 outputs; SHAP_SAMPLES / SCENE_SAMPLES / CACHE env vars
    python -m src.step6_cbam       # CBAM channel + spatial attention vs Grad-CAM
    python tests/test_xai_core.py  # backend-free unit tests of the SHAP/agreement numerics
## Steps
1. Data audit  [done]
2. Build inputs: colour->index decoding, sea/inland-water split, PROXY risk labels, 64x64 patches  [done]
3. MobileNetV2+CBAM patch classifier + predicted high-risk map  [done]
4. Grad-CAM: patch + scene heatmaps, focus analysis, randomised-model check, GeoTIFF export  [done]
5. SHAP (document section 3): per-pixel NDVI/NDWI contributions to the logit, Input -> Feature Contribution -> Risk Prediction, completeness check, focus + Grad-CAM agreement, randomised-model check, scene maps + GeoTIFF  [code written]
6. CBAM attention (document section 4): channel + spatial attention extraction, comparison with Grad-CAM, randomised-model check, scene maps + GeoTIFF  [code written]
7. Validation & fusion
8. Final integrated explanation
