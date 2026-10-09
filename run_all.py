"""Run the Malaria-Risk XAI pipeline from a single file.

Examples:
    python run_all.py
    python run_all.py --steps 1 2
    python run_all.py --steps 5 6 7 8
    python run_all.py --steps 4 --cache
    python run_all.py --dry-run
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Step definitions:
# (module, title, required inputs, expected outputs)
STEPS = {
    1: (
        "src.step1_data_audit",
        "Data audit",
        [],
        [
            "outputs/step1_data_audit.csv",
            "outputs/step1_monthly_overview.png",
        ],
    ),
    2: (
        "src.step2_build_dataset",
        "Decode data + build patches",
        [],
        [
            "outputs/step2_patches.npz",
            "outputs/step2_meta.json",
            "outputs/step2_decoded_and_risk.png",
            "outputs/step2_patch_examples.png",
        ],
    ),
    3: (
        "src.step3_train",
        "Train MobileNetV2+CBAM model",
        [
            "outputs/step2_patches.npz",
            "outputs/step2_meta.json",
        ],
        [
            "models/mobilenetv2_cbam.keras",
            "outputs/step3_metrics.json",
            "outputs/step3_scene_prob_2025-04.npz",
        ],
    ),
    4: (
        "src.step4_gradcam",
        "Grad-CAM + focus analysis",
        [
            "models/mobilenetv2_cbam.keras",
            "outputs/step3_metrics.json",
            "outputs/step3_scene_prob_2025-04.npz",
        ],
        [
            "outputs/step4_metrics.json",
            "outputs/step4_gradcam_patches.npz",
        ],
    ),
    5: (
        "src.step5_shap",
        "SHAP explanations",
        [
            "models/mobilenetv2_cbam.keras",
            "outputs/step3_scene_prob_2025-04.npz",
            "outputs/step4_gradcam_patches.npz",
        ],
        [
            "outputs/step5_shap_patches.npz",
            "outputs/step5_metrics.json",
        ],
    ),
    6: (
        "src.step6_cbam",
        "CBAM attention analysis",
        [
            "models/mobilenetv2_cbam.keras",
            "outputs/step4_gradcam_patches.npz",
        ],
        [
            "outputs/step6_cbam_patches.npz",
            "outputs/step6_metrics.json",
        ],
    ),
    7: (
        "src.step7_validation",
        "Validation & fusion",
        [
            "outputs/step4_gradcam_patches.npz",
            "outputs/step5_shap_patches.npz",
            "outputs/step6_cbam_patches.npz",
        ],
        [
            "outputs/step7_validation_patches.npz",
            "outputs/step7_metrics.json",
        ],
    ),
    8: (
        "src.step8_final_output",
        "Final integrated output",
        [
            "outputs/step5_shap_patches.npz",
            "outputs/step6_cbam_patches.npz",
            "outputs/step7_validation_patches.npz",
            "outputs/step7_metrics.json",
        ],
        [
            "outputs/step8_explanations_test.csv",
            "outputs/step8_report.json",
        ],
    ),
}


def check(selected):
    """Check dataset availability, step files and prerequisites."""
    problems = []
    dataset_dir = ROOT / "dataset"

    dataset_files = [
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    ] if dataset_dir.exists() else []

    if not dataset_files:
        problems.append(
            "No .tif or .tiff files found under ./dataset. "
            "Check the dataset location."
        )

    # Outputs expected from steps selected earlier in the run.
    produced = set()

    for step in selected:
        module, title, needs, makes = STEPS[step]
        module_path = ROOT / (module.replace(".", "/") + ".py")

        if not module_path.is_file():
            problems.append(
                f"Step {step}: module file not found: "
                f"{module_path.relative_to(ROOT)}"
            )

        for need in needs:
            if not (ROOT / need).is_file() and need not in produced:
                problems.append(
                    f"Step {step} requires '{need}', which is missing. "
                    "Run the step that creates it first."
                )

        produced.update(makes)

    return problems


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        choices=sorted(STEPS),
        default=sorted(STEPS),
        help="Steps to run (default: all)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Set CACHE=1 for Step 4 to reuse saved heatmaps if supported",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check prerequisites without executing steps",
    )

    args = parser.parse_args()
    selected = sorted(set(args.steps))

    print(
        "Pipeline:",
        " -> ".join(f"[{s}] {STEPS[s][1]}" for s in selected),
        flush=True,
    )

    problems = check(selected)

    if problems:
        print("\nPrerequisite checks failed:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)

    if args.dry_run:
        print("\nPrerequisite checks passed.")
        print("Dry run only: no pipeline steps were executed.")
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src")]
    )
    env.pop("CACHE", None)

    if args.cache:
        env["CACHE"] = "1"

    results = []

    for step in selected:
        module, title, _, expected_outputs = STEPS[step]

        print(
            f"\n{'=' * 70}\n"
            f"STEP {step}: {title}\n"
            f"Command: {sys.executable} -m {module}\n"
            f"{'=' * 70}",
            flush=True,
        )

        start = time.time()

        try:
            result = subprocess.run(
                [sys.executable, "-m", module],
                cwd=ROOT,
                env=env,
                check=False,
            )
            return_code = result.returncode

        except OSError as exc:
            print(f"\nCould not start Step {step}: {exc}")
            return_code = 1

        elapsed = time.time() - start
        results.append((step, title, return_code, elapsed))

        if return_code != 0:
            print(f"\nERROR: Step {step} ({title}) failed.")
            print(f"Exit code: {return_code}")
            print("The pipeline will stop to avoid running later steps.")
            break

        print(f"\nStep {step} completed in {elapsed:.1f} seconds.")

        missing_outputs = [
            path for path in expected_outputs
            if not (ROOT / path).is_file()
        ]

        if missing_outputs:
            print("WARNING: Expected output files were not found:")
            for path in missing_outputs:
                print(f"  - {path}")
            print(
                "Check the step's actual output filenames before "
                "relying on downstream prerequisite checks."
            )

    print(f"\n{'=' * 70}\nPIPELINE SUMMARY\n{'=' * 70}")

    for step, title, return_code, elapsed in results:
        status = "OK" if return_code == 0 else "FAILED"
        print(
            f"Step {step}: {title:<32} "
            f"{status:<7} {elapsed:6.1f}s"
        )

    successful = bool(results) and all(
        result[2] == 0 for result in results
    )
    completed_all = len(results) == len(selected)

    if successful and completed_all:
        print("\nAll selected steps completed successfully.")
        sys.exit(0)

    print("\nPipeline did not complete all selected steps.")
    sys.exit(1)


if __name__ == "__main__":
    main()
