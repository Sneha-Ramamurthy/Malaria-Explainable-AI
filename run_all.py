from pathlib import Path
import argparse
import subprocess
import sys


# Project paths
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"


# Pipeline steps, in execution order
STEPS = [
    {
        "number": 1,
        "module": "src.step1_data_audit",
        "title": "Step 1 - Data audit",
        "needs": [],
        "makes": [
            "outputs/step1_data_audit.csv",
            "outputs/step1_monthly_overview.png",
        ],
    },
    {
        "number": 2,
        "module": "src.step2_build_dataset",
        "title": "Step 2 - Build dataset",
        "needs": [],
        "makes": [
            "outputs/step2_patches.npz",
            "outputs/step2_meta.json",
            "outputs/step2_scene_2025-04.npz",
            "outputs/step2_scene_2025-05.npz",
            "outputs/step2_decoded_and_risk.png",
            "outputs/step2_patch_examples.png",
        ],
    },
    {
        "number": 3,
        "module": "src.step3_train",
        "title": "Step 3 - Train model",
        "needs": [
            "outputs/step2_patches.npz",
            "outputs/step2_meta.json",
            "outputs/step2_scene_2025-04.npz",
            "outputs/step2_scene_2025-05.npz",
        ],
        "makes": [
            "models/mobilenetv2_cbam.keras",
            "outputs/step3_metrics.json",
            "outputs/step3_scene_prob_2025-04.npz",
            "outputs/step3_scene_prob_2025-05.npz",
            "outputs/step3_training_and_test.png",
            "outputs/step3_riskmap_2025-04.png",
            "outputs/step3_riskmap_2025-05.png",
        ],
    },
    {
        "number": 4,
        "module": "src.step4_gradcam",
        "title": "Step 4 - Grad-CAM explanations",
        "needs": [
            "models/mobilenetv2_cbam.keras",
            "outputs/step2_meta.json",
            "outputs/step2_patches.npz",
            "outputs/step2_scene_2025-04.npz",
            "outputs/step2_scene_2025-05.npz",
            "outputs/step3_metrics.json",
            "outputs/step3_scene_prob_2025-04.npz",
            "outputs/step3_scene_prob_2025-05.npz",
        ],
        "makes": [
            "outputs/step4_metrics.json",
            "outputs/step4_gradcam_patches.npz",
            "outputs/step4_gradcam_patches.png",
            "outputs/step4_focus_analysis.png",
            "outputs/step4_scene_cam_2025-04.npz",
            "outputs/step4_scene_cam_2025-05.npz",
            "outputs/step4_gradcam_scene_2025-04.png",
            "outputs/step4_gradcam_scene_2025-05.png",
        ],
    },
    {
        "number": 5,
        "module": "src.step5_shap",
        "title": "Step 5 - SHAP explanations",
        "needs": [
            "models/mobilenetv2_cbam.keras",
            "outputs/step2_patches.npz",
            "outputs/step2_meta.json",
            "outputs/step2_scene_2025-04.npz",
            "outputs/step2_scene_2025-05.npz",
            "outputs/step3_metrics.json",
            "outputs/step3_scene_prob_2025-04.npz",
            "outputs/step3_scene_prob_2025-05.npz",
            "outputs/step4_gradcam_patches.npz",
            "outputs/step4_scene_cam_2025-04.npz",
            "outputs/step4_scene_cam_2025-05.npz",
        ],
        "makes": [
            "outputs/step5_shap_patches.npz",
            "outputs/step5_metrics.json",
            "outputs/step5_shap_patches.png",
            "outputs/step5_shap_flow.png",
            "outputs/step5_shap_global.png",
            "outputs/step5_scene_shap_2025-04.npz",
            "outputs/step5_scene_shap_2025-05.npz",
            "outputs/step5_shap_2025-04.tif",
            "outputs/step5_shap_2025-05.tif",
            "outputs/step5_shap_scene_2025-04.png",
            "outputs/step5_shap_scene_2025-05.png",
        ],
    },
    {
        "number": 6,
        "module": "src.step6_cbam",
        "title": "Step 6 - CBAM attention analysis",
        "needs": [
            "models/mobilenetv2_cbam.keras",
            "outputs/step2_patches.npz",
            "outputs/step2_meta.json",
            "outputs/step2_scene_2025-04.npz",
            "outputs/step2_scene_2025-05.npz",
            "outputs/step3_metrics.json",
            "outputs/step3_scene_prob_2025-04.npz",
            "outputs/step3_scene_prob_2025-05.npz",
            "outputs/step4_gradcam_patches.npz",
            "outputs/step4_scene_cam_2025-04.npz",
            "outputs/step4_scene_cam_2025-05.npz",
        ],
        "makes": [
            "outputs/step6_cbam_patches.npz",
            "outputs/step6_metrics.json",
            "outputs/step6_cbam_patches.png",
            "outputs/step6_cbam_channels.png",
            "outputs/step6_cbam_focus.png",
            "outputs/step6_scene_cbam_2025-04.npz",
            "outputs/step6_scene_cbam_2025-05.npz",
            "outputs/step6_cbam_2025-04.tif",
            "outputs/step6_cbam_2025-05.tif",
            "outputs/step6_cbam_scene_2025-04.png",
            "outputs/step6_cbam_scene_2025-05.png",
        ],
    },
    {
        "number": 7,
        "module": "src.step7_validation_fusion",
        "title": "Step 7 - Validation and fusion",
        "needs": [
            "models/mobilenetv2_cbam.keras",
            "outputs/step2_patches.npz",
            "outputs/step2_meta.json",
            "outputs/step2_scene_2025-04.npz",
            "outputs/step2_scene_2025-05.npz",
            "outputs/step3_metrics.json",
            "outputs/step3_scene_prob_2025-04.npz",
            "outputs/step3_scene_prob_2025-05.npz",
            "outputs/step4_gradcam_patches.npz",
            "outputs/step4_scene_cam_2025-04.npz",
            "outputs/step4_scene_cam_2025-05.npz",
            "outputs/step5_shap_patches.npz",
            "outputs/step5_scene_shap_2025-04.npz",
            "outputs/step5_scene_shap_2025-05.npz",
            "outputs/step6_cbam_patches.npz",
            "outputs/step6_scene_cbam_2025-04.npz",
            "outputs/step6_scene_cbam_2025-05.npz",
        ],
        "makes": [
            "outputs/step7_validation_patches.npz",
            "outputs/step7_scene_fusion_2025-04.npz",
            "outputs/step7_scene_fusion_2025-05.npz",
            "outputs/step7_fusion_2025-04.tif",
            "outputs/step7_fusion_2025-05.tif",
            "outputs/step7_agreement.png",
            "outputs/step7_fidelity.png",
            "outputs/step7_stability.png",
            "outputs/step7_fusion_patches.png",
            "outputs/step7_fusion_scene_2025-04.png",
            "outputs/step7_fusion_scene_2025-05.png",
            "outputs/step7_metrics.json",
        ],
    },
    {
        "number": 8,
        "module": "src.step8_final",
        "title": "Step 8 - Integrated explanation and final report",
        "needs": [
            "outputs/step2_patches.npz",
            "outputs/step2_meta.json",
            "outputs/step2_scene_2025-04.npz",
            "outputs/step2_scene_2025-05.npz",
            "outputs/step3_metrics.json",
            "outputs/step3_scene_prob_2025-04.npz",
            "outputs/step3_scene_prob_2025-05.npz",
            "outputs/step4_gradcam_patches.npz",
            "outputs/step4_scene_cam_2025-04.npz",
            "outputs/step4_scene_cam_2025-05.npz",
            "outputs/step5_shap_patches.npz",
            "outputs/step5_scene_shap_2025-04.npz",
            "outputs/step5_scene_shap_2025-05.npz",
            "outputs/step6_cbam_patches.npz",
            "outputs/step6_scene_cbam_2025-04.npz",
            "outputs/step6_scene_cbam_2025-05.npz",
            "outputs/step7_validation_patches.npz",
            "outputs/step7_scene_fusion_2025-04.npz",
            "outputs/step7_scene_fusion_2025-05.npz",
            "outputs/step7_metrics.json",
        ],
        "makes": [
            "outputs/step8_explanations_test.csv",
            "outputs/step8_report.json",
            "outputs/step8_report.md",
            "outputs/step8_final_scene_2025-04.png",
            "outputs/step8_final_scene_2025-05.png",
            "outputs/step8_final_2025-04.tif",
            "outputs/step8_final_2025-05.tif",
        ],
    },
]


def check_files(paths, step_number, file_type):
    missing = [
        str(ROOT / path)
        for path in paths
        if not (ROOT / path).is_file()
    ]

    if missing:
        print(f"\nERROR: Step {step_number} cannot continue.")
        print(f"Missing {file_type} file(s):")

        for path in missing:
            print(f"  - {path}")

        return False

    return True


def run_step(step):
    number = step["number"]
    title = step["title"]

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    if not check_files(step["needs"], number, "required input"):
        return False

    command = [
        sys.executable,
        "-m",
        step["module"],
    ]

    print("Running:", " ".join(command))

    try:
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        print(f"\nERROR: {title} failed.")
        print(f"Exit code: {error.returncode}")
        print("Fix the error above before continuing.")
        return False

    if not check_files(step["makes"], number, "expected output"):
        print(f"\nERROR: {title} did not create all expected outputs.")
        return False

    print(f"\nSUCCESS: {title} completed.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the eight-step Malaria Explainable AI pipeline."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        choices=range(1, 9),
        help="Optional step numbers to run, for example: --steps 1 2",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    selected_steps = STEPS

    if args.steps:
        selected_numbers = set(args.steps)
        selected_steps = [
            step for step in STEPS
            if step["number"] in selected_numbers
        ]

    print("Malaria Risk Prediction with Explainable AI")
    print(f"Project directory: {ROOT}")
    print(f"Steps selected: {[step['number'] for step in selected_steps]}")

    for step in selected_steps:
        if not run_step(step):
            print("\nPipeline stopped because a step failed.")
            sys.exit(1)

    print("\n" + "=" * 65)
    print("ALL SELECTED STEPS COMPLETED SUCCESSFULLY.")
    print(f"Check the generated files in: {OUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
