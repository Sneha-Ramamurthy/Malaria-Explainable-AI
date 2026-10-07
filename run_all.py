"""Run the Malaria-Risk XAI pipeline from a single file.

    python run_all.py                 # all 8 steps, in order
    python run_all.py --steps 1 2     # only some steps
    python run_all.py --steps 5 6 7 8 # only the XAI steps (needs steps 1-4 outputs already present)
    python run_all.py --steps 4 --cache   # Step 4 re-using saved heatmaps (fast)
    python run_all.py --dry-run       # show what would run, check prerequisites

Each step runs in its own Python process (so TensorFlow/matplotlib state never leaks between
steps) and the pipeline stops at the first failure.
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

#        id: (module,                      title,                          needs (paths),                                            makes (paths))
STEPS = {
    1: ("src.step1_data_audit",   "Data audit",                    [],
        ["outputs/step1_data_audit.csv"]),
    2: ("src.step2_build_dataset", "Decode data + build patches",  [],
        ["outputs/step2_patches.npz", "outputs/step2_meta.json"]),
    3: ("src.step3_train",         "Train MobileNetV2+CBAM model",  ["outputs/step2_patches.npz", "outputs/step2_meta.json"],
        ["models/mobilenetv2_cbam.keras", "outputs/step3_metrics.json"]),
    4: ("src.step4_gradcam",       "Grad-CAM + focus analysis",     ["models/mobilenetv2_cbam.keras", "outputs/step3_metrics.json",
                                                                     "outputs/step3_scene_prob_2025-04.npz"],
        ["outputs/step4_metrics.json"]),
    # Steps 5-8 live in your own src/ folder; their inputs/outputs are not checked here.
    5: ("src.step5_shap",          "SHAP explanations",             [], []),
    6: ("src.step6_cbam",          "CBAM attention analysis",       [], []),
    7: ("src.step7_validation",    "Validation & fusion",           [], []),
    8: ("src.step8_final_output",  "Final integrated output",       [], []),
}
# validation_core.py and xai_utils.py are helper modules imported by the steps - they are not run directly.


def check(selected):
    """Return a list of problems (missing dataset / missing prerequisites)."""
    problems = []
    if not list((ROOT / "dataset").glob("*.tiff")):
        problems.append("No .tiff files in ./dataset  ->  unzip dataset.zip into the project folder.")
    produced = set()
    for s in selected:
        if not (ROOT / (STEPS[s][0].replace(".", "/") + ".py")).exists():
            problems.append(f"Step {s}: file '{STEPS[s][0].replace('.', '/')}.py' not found.")
        for need in STEPS[s][2]:
            if not (ROOT / need).exists() and need not in produced:
                problems.append(f"Step {s} needs '{need}', which doesn't exist yet -> include the earlier step (e.g. --steps ... ).")
        produced.update(STEPS[s][3])
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, nargs="+", choices=sorted(STEPS), default=sorted(STEPS),
                    help="which steps to run (default: all)")
    ap.add_argument("--cache", action="store_true", help="Step 4 only: reuse saved Grad-CAM heatmaps instead of recomputing")
    ap.add_argument("--dry-run", action="store_true", help="only list the steps and check prerequisites")
    a = ap.parse_args()
    selected = sorted(set(a.steps))

    print("Pipeline:", " -> ".join(f"[{s}] {STEPS[s][1]}" for s in selected))
    problems = check(selected)
    if problems:
        print("\nCannot run:"); [print("  -", p) for p in problems]; sys.exit(1)
    if a.dry_run:
        print("Prerequisites OK (dry run, nothing executed)."); return

    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "src")])}
    env.pop("CACHE", None)
    if a.cache: env["CACHE"] = "1"
    results = []
    for s in selected:
        mod, title, _, _ = STEPS[s]
        print(f"\n{'=' * 70}\nSTEP {s}: {title}   (python -m {mod})\n{'=' * 70}", flush=True)
        t0 = time.time()
        rc = subprocess.run([sys.executable, "-m", mod], cwd=ROOT, env=env).returncode
        results.append((s, title, rc, time.time() - t0))
        if rc != 0:
            print(f"\nStep {s} failed (exit code {rc}); stopping."); break

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for s, title, rc, dt in results:
        print(f"  Step {s}  {title:<32} {'OK    ' if rc == 0 else 'FAILED'}  {dt:6.0f}s")
    sys.exit(0 if all(r[2] == 0 for r in results) else 1)


if __name__ == "__main__":
    main()