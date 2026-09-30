# Malaria Risk Prediction - Explainable AI layer
Place the extracted `dataset/` folder inside this directory, then:
    pip install -r requirements.txt
    python -m src.step1_data_audit
    python -m src.step2_build_dataset
    python -m src.step3_train      # ~4 min on CPU; saves models/mobilenetv2_cbam.keras
## Steps
1. Data audit  [done]
2. Build inputs: colour->index decoding, sea/inland-water split, PROXY risk labels, 64x64 patches  [done]
3. MobileNetV2+CBAM patch classifier + predicted high-risk map  [done]
4. Grad-CAM
5. SHAP
6. CBAM attention
7. Validation & fusion
8. Final integrated explanation
