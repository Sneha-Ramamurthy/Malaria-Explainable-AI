import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(file).resolve().parent
OUT_DIR = ROOT / “outputs”
MODEL_DIR = ROOT / “models”

STEPS = [
{
“module”: “src.step1_data_audit”,
“title”: “Step 1 - Data audit”,
“needs”: [],
“makes”: [
“outputs/step1_data_audit.csv”,
“outputs/step1_monthly_overview.png”,
],
},
{
“module”: “src.step2_build_dataset”,
“title”: “Step 2 - Build dataset”,
“needs”: [],
“makes”: [
“outputs/step2_patches.npz”,
“outputs/step2_meta.json”,
“outputs/step2_scene_2025-04.npz”,
“outputs/step2_scene_2025-05.npz”,
“outputs/step2_decoded_and_risk.png”,
“outputs/step2_patch_examples.png”,
],
},
{
“module”: “src.step3_train”,
“title”: “Step 3 - Train model”,
“needs”: [
“outputs/step2_patches.npz”,
“outputs/step2_meta.json”,
“outputs/step2_scene_2025-04.npz”,
“outputs/step2_scene_2025-05.npz”,
],
“makes”: [
“models/mobilenetv2_cbam.keras”,
“outputs/step3_metrics.json”,
“outputs/step3_scene_prob_2025-04.npz”,
“outputs/step3_scene_prob_2025-05.npz”,
“outputs/step3_training_and_test.png”,
“outputs/step3_riskmap_2025-04.png”,
“outputs/step3_riskmap_2025-05.png”,
],
},
{

“module”: “src.step4_gradcam”,
“title”: “Step 4 - Grad-CAM explanations”,
“needs”: [
“models/mobilenetv2_cbam.keras”,
“outputs/step2_meta.json”,
“outputs/step2_patches.npz”,
“outputs/step2_scene_2025-04.npz”,
“outputs/step2_scene_2025-05.npz”,
“outputs/step3_metrics.json”,
“outputs/step3_scene_prob_2025-04.npz”,
“outputs/step3_scene_prob_2025-05.npz”,
],
“makes”: [
“outputs/step4_metrics.json”,
“outputs/step4_gradcam_patches.npz”,
“outputs/step4_gradcam_patches.png”,
“outputs/step4_focus_analysis.png”,
“outputs/step4_scene_cam_2025-04.npz”,
“outputs/step4_scene_cam_2025-05.npz”,
“outputs/step4_gradcam_2025-04.tif”,
“outputs/step4_gradcam_2025-05.tif”,
“outputs/step4_gradcam_scene_2025-04.png”,
“outputs/step4_gradcam_scene_2025-05.png”,
],
}
{
“module”: “src.step5_shap”,
“title”: “Step 5 - SHAP explanations”,
“needs”: [
“models/mobilenetv2_cbam.keras”,
“outputs/step2_patches.npz”,
“outputs/step2_meta.json”,
“outputs/step2_scene_2025-04.npz”,
“outputs/step2_scene_2025-05.npz”,
“outputs/step3_metrics.json”,
“outputs/step3_scene_prob_2025-04.npz”,
“outputs/step3_scene_prob_2025-05.npz”,
“outputs/step4_gradcam_patches.npz”,
“outputs/step4_scene_cam_2025-04.npz”,
“outputs/step4_scene_cam_2025-05.npz”,
],
“makes”: [
“outputs/step5_shap_patches.npz”,
“outputs/step5_metrics.json”,
“outputs/step5_shap_patches.png”,
“outputs/step5_shap_flow.png”,
“outputs/step5_shap_global.png”,
“outputs/step5_scene_shap_2025-04.npz”,
“outputs/step5_scene_shap_2025-05.npz”,
“outputs/step5_shap_2025-04.tif”,
“outputs/step5_shap_2025-05.tif”,
“outputs/step5_shap_scene_2025-04.png”,
“outputs/step5_shap_scene_2025-05.png”,
],
}
{
“module”: “src.step6_cbam”,
“title”: “Step 6 - CBAM attention analysis”,
“needs”: [
“models/mobilenetv2_cbam.keras”,
“outputs/step2_patches.npz”,
“outputs/step2_meta.json”,
“outputs/step2_scene_2025-04.npz”,
“outputs/step2_scene_2025-05.npz”,
“outputs/step3_metrics.json”,
“outputs/step3_scene_prob_2025-04.npz”,
“outputs/step3_scene_prob_2025-05.npz”,
“outputs/step4_gradcam_patches.npz”,
“outputs/step4_scene_cam_2025-04.npz”,
“outputs/step4_scene_cam_2025-05.npz”,
],
“makes”: [
“outputs/step6_cbam_patches.npz”,
“outputs/step6_metrics.json”,
“outputs/step6_cbam_patches.png”,
“outputs/step6_cbam_channels.png”,
“outputs/step6_cbam_focus.png”,
“outputs/step6_scene_cbam_2025-04.npz”,
“outputs/step6_scene_cbam_2025-05.npz”,
“outputs/step6_cbam_2025-04.tif”,
“outputs/step6_cbam_2025-05.tif”,
“outputs/step6_cbam_scene_2025-04.png”,
“outputs/step6_cbam_scene_2025-05.png”,
],
},
{
“module”: “src.step7_validation_fusion”,
“title”: “Step 7 - Validation and fusion”,
“needs”: [
“models/mobilenetv2_cbam.keras”,
“outputs/step2_patches.npz”,
“outputs/step2_meta.json”,
“outputs/step2_scene_2025-04.npz”,
“outputs/step2_scene_2025-05.npz”,
“outputs/step3_metrics.json”,
“outputs/step3_scene_prob_2025-04.npz”,
“outputs/step3_scene_prob_2025-05.npz”,
“outputs/step4_gradcam_patches.npz”,
“outputs/step4_scene_cam_2025-04.npz”,
“outputs/step4_scene_cam_2025-05.npz”,
“outputs/step5_shap_patches.npz”,
“outputs/step5_scene_shap_2025-04.npz”,
“outputs/step5_scene_shap_2025-05.npz”,
“outputs/step6_cbam_patches.npz”,
“outputs/step6_scene_cbam_2025-04.npz”,
“outputs/step6_scene_cbam_2025-05.npz”,
],
“makes”: [
“outputs/step7_validation_patches.npz”,
“outputs/step7_scene_fusion_2025-04.npz”,
“outputs/step7_scene_fusion_2025-05.npz”,
“outputs/step7_fusion_2025-04.tif”,
“outputs/step7_fusion_2025-05.tif”,
“outputs/step7_agreement.png”,
“outputs/step7_fidelity.png”,
“outputs/step7_stability.png”,
“outputs/step7_fusion_patches.png”,
“outputs/step7_fusion_scene_2025-04.png”,
“outputs/step7_fusion_scene_2025-05.png”,
“outputs/step7_metrics.json”,
],
},
{
“module”: “src.step8_integrated_explanation”,
“title”: “Step 8 - Integrated explanation”,
“needs”: [
“outputs/step5_shap_patches.npz”,
“outputs/step6_cbam_patches.npz”,
“outputs/step7_validation_patches.npz”,
“outputs/step7_metrics.json”,
],
“makes”: [
“outputs/step8_explanation_summary.csv”,
“outputs/step8_integrated_report.png”,
],
},
]

def check_prerequisites(step):
missing = [
item for item in step[“needs”]
if not (ROOT / item).exists()
]

if missing:
    print(f"\nMissing prerequisites for {step['title']}:")
    for item in missing:
        print(f"  - {item}")
    print("Run the earlier steps first.")
    return False
return True

def run_step(step):
print(”\n” + “=” * 60)
print(step[“title”])
print(”=” * 60)

if not check_prerequisites(step):
    return False
command = [
    sys.executable,
    "-m",
    step["module"],
]
try:
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )
except subprocess.CalledProcessError as error:
    print(f"\nFAILED: {step['title']}")
    print(f"Exit code: {error.returncode}")
    return False
missing_outputs = [
    item for item in step["makes"]
    if not (ROOT / item).exists()
]
if missing_outputs:
    print("\nStep finished, but these expected outputs were not found:")
    for item in missing_outputs:
        print(f"  - {item}")
    print("Check the step's source code and output filenames.")
    return False
print(f"\nSUCCESS: {step['title']}")
return True

def main():
parser = argparse.ArgumentParser(
description=“Run the Malaria Explainable AI pipeline.”
)
parser.add_argument(
“–steps”,
nargs=”+”,
type=int,
help=“Optional step numbers to run, for example: –steps 1 2”,
)
args = parser.parse_args()

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
if args.steps:
    if any(number < 1 or number > len(STEPS) for number in args.steps):
        parser.error("Step numbers must be between 1 and 8.")
    selected_numbers = list(dict.fromkeys(args.steps))
else:
    selected_numbers = list(range(1, len(STEPS) + 1))
print("Malaria Risk Prediction with Explainable AI")
print(f"Project folder: {ROOT}")
print(f"Steps selected: {selected_numbers}")
for number in selected_numbers:
    step = STEPS[number - 1]
    if not run_step(step):
        print("\nPipeline stopped because a step failed.")
        sys.exit(1)
print("\nAll selected steps completed successfully.")

if name == “main”:
main()
