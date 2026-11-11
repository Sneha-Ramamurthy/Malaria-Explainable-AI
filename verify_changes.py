"""Proof-of-concept verifier: checks from the command line that the fixes made to this project are in place AND work.

    python verify_changes.py                 # level A (+ B/C on whatever already exists); safe, writes nothing
    python verify_changes.py --run-data      # also RUNS steps 1-2 on dataset/ (needs rasterio), then checks the fixes
    python verify_changes.py --level C       # only check the outputs of a full `python run_all.py` run
    python verify_changes.py --json report.json

Levels
  A  code      files parse, run_all references real modules, 40+ tests (XAI maths, text layer) pass
  B  data      Step 2 outputs: land mask consistent, scene windows == training patches, label cutoff from train only,
               no spatial-block leakage between train/val/test
  C  pipeline  outputs of Steps 3-8: new metrics present, SHAP adds up and matches the shap library, explanations
               depend on the trained weights, stability not inflated, deletion test passes, prediction paths agree

Every check prints PASS / WARN / FAIL / SKIP (SKIP = prerequisite missing, with the command that creates it).
Exit code 1 if anything FAILS, otherwise 0.
"""
import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
RESULTS = []                                        # (level, name, status, detail)
ICON = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}


def record(level, name, status, detail=""):
    RESULTS.append((level, name, status, detail))
    print(f"  {ICON[status]} {name}" + (f"  -> {detail}" if detail else ""))


def check(level, name, fn):
    """Run fn() -> (status, detail); an exception is a FAIL, not a crash."""
    try:
        status, detail = fn()
    except Exception as exc:                        # noqa: BLE001
        status, detail = "FAIL", f"{type(exc).__name__}: {str(exc)[:200]}"
    record(level, name, status, detail)


def jload(name):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def need(*names):
    missing = [n for n in names if not (OUT / n).exists()]
    return missing


def skip_if_missing(level, name, files, hint):
    miss = need(*files)
    if miss:
        record(level, name, "SKIP", f"missing {', '.join(miss)} - {hint}")
        return True
    return False


def run_cmd(args, timeout=3600):
    p = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)


# ============================================================================================ level A
def level_a(skip_tests):
    print("\nLEVEL A - code")

    def parse_all():
        files = list(ROOT.glob("src/*.py")) + list(ROOT.glob("tests/*.py")) + [ROOT / "run_all.py"]
        for f in files:
            ast.parse(f.read_text(encoding="utf-8"), str(f))        # SyntaxError e.g. repeated keyword argument
        return "PASS", f"{len(files)} files parse (the original Step 2 did not)"
    check("A", "all Python files parse", parse_all)

    def modules():
        import re
        mods = re.findall(r'"module": "(src\.[A-Za-z0-9_]+)"', (ROOT / "run_all.py").read_text())
        bad = [m for m in mods if not (ROOT / (m.replace(".", "/") + ".py")).exists()]
        return ("FAIL", f"run_all.py points to missing {bad}") if bad else ("PASS", f"{len(mods)} step modules exist")
    check("A", "run_all.py references real modules", modules)

    def guard():
        t = (ROOT / "src" / "step1_data_audit.py").read_text()
        return ("PASS", "Step 1 has a correct __main__ guard") if 'if __name__ == "__main__":' in t else ("FAIL", "bad guard")
    check("A", "Step 1 main guard", guard)

    if skip_tests:
        record("A", "unit tests", "SKIP", "--skip-tests given")
        return
    for mod, label in (("tests.test_xai_core", "XAI maths (SHAP/expected gradients, fusion, agreement, deletion test)"),
                       ("tests.test_train_utils", "Step 3 helpers (Youden threshold, block-bootstrap CI)"),
                       ("tests.test_text_explainer", "text explainable-AI layer (wording follows the numbers)"),
                       ("tests.test_project_integrity", "project integrity")):
        def run(mod=mod):
            code, out = run_cmd(["-m", mod], timeout=900)
            passed, failed = out.count("PASS"), out.count("FAIL")
            last = [l for l in out.splitlines() if l.strip()][-1:] or [""]
            return ("PASS", f"{passed} tests passed") if code == 0 else ("FAIL", f"{failed} failed; {last[0][:120]}")
        check("A", f"tests: {label}", run)


# ============================================================================================ level B
def load_step2():
    import numpy as np
    d = np.load(OUT / "step2_patches.npz", allow_pickle=False)
    meta = json.loads((OUT / "step2_meta.json").read_text())
    months = meta["months"]
    scene = {m: np.load(OUT / f"step2_scene_{m}.npz", allow_pickle=False) for m in months}
    return d, meta, months, scene


def level_b(run_data):
    print("\nLEVEL B - data (Steps 1-2)")
    if run_data:
        if not (ROOT / "dataset").is_dir():
            record("B", "run Steps 1-2", "SKIP", "no dataset/ folder next to this script")
        else:
            try:
                import rasterio  # noqa: F401
            except ImportError:
                record("B", "run Steps 1-2", "SKIP", "rasterio not installed (pip install rasterio)")
            else:
                for mod in ("src.step1_data_audit", "src.step2_build_dataset"):
                    def run(mod=mod):
                        code, out = run_cmd(["-m", mod])
                        return ("PASS", "ran successfully") if code == 0 else ("FAIL", out.strip().splitlines()[-1][:200])
                    check("B", f"run {mod}", run)

    if skip_if_missing("B", "Step 2 consistency checks",
                       ["step2_patches.npz", "step2_meta.json", "step2_scene_2025-04.npz"],
                       "run `python verify_changes.py --run-data` or `python -m src.step2_build_dataset`"):
        return
    import numpy as np
    from src import xai_utils as U
    d, meta, months, scene = load_step2()
    X, valid, split, y, pm, hrm = d["X"].astype("float32"), d["valid"].astype(bool), d["split"], d["y"], d["meta"], d["mask"].astype(bool)

    def scene_keys():
        miss = [m for m in months if "valid_risk" not in scene[m].files]
        if miss:
            return "FAIL", f"scene files for {miss} have no valid_risk (written by the old Step 2) - re-run Step 2"
        return "PASS", "scene files contain valid, valid_in and valid_risk"
    check("B", "Step 2 saves the land mask (valid_risk)", scene_keys)

    def land_mask():
        for m in months:
            s = scene[m]
            if int((s["valid_risk"] & s["water"]).sum()) != 0:
                return "FAIL", "valid_risk overlaps water"
        zeros = bool((X[~valid] == 0).all())
        return ("PASS", "land mask excludes water; masked patch pixels are exactly 0") if zeros else ("FAIL", "masked pixels are not 0")
    check("B", "land mask correct", land_mask)

    def windows_match():
        P, H, W = meta["patch"], meta["shape"][0], meta["shape"][1]
        ctx = SimpleNamespace(PATCH=P, H=H, W=W)
        tot = bad = 0
        for mi, m in enumerate(months):
            s = {k: scene[m][k] for k in scene[m].files}
            rs, cs, wins = U.scene_windows(s, ctx, stride=32)
            look = {(int(r), int(c)): k for k, (r, c) in enumerate(zip(rs, cs))}
            for i in np.where(pm[:, 0] == mi)[0]:
                key = (int(pm[i, 1]), int(pm[i, 2]))
                if key in look:
                    tot += 1
                    bad += int(not np.allclose(wins[look[key]], X[i], atol=1e-3))
        if tot == 0:
            return "WARN", "no scene window coincides with a training patch to compare"
        return ("PASS", f"{tot}/{tot} scene windows identical to the training patch at the same place (old code: 0 identical)") \
            if bad == 0 else ("FAIL", f"{bad}/{tot} scene windows differ from training patches")
    check("B", "scene inputs == training inputs", windows_match)

    def cutoff():
        frac = (hrm & valid).sum((1, 2)) / np.maximum(valid.sum((1, 2)), 1)
        thr = meta["patch_frac_threshold"]
        expect = float(np.quantile(frac[split == 0], 0.70))
        return ("PASS", f"label cutoff {thr:.4f} = 70th percentile of TRAIN patches only") if abs(thr - expect) < 1e-6 \
            else ("FAIL", f"cutoff {thr:.4f} != train-only {expect:.4f}")
    check("B", "label cutoff learned from training patches only", cutoff)

    def leakage():
        b = pm[:, 3]
        sets = [set(b[split == k]) for k in (0, 1, 2)]
        ok = sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])
        return ("PASS", f"train/val/test patches: {[int((split == k).sum()) for k in (0, 1, 2)]}, no shared spatial blocks") if ok \
            else ("FAIL", "spatial blocks shared between splits")
    check("B", "no spatial leakage between splits", leakage)


# ============================================================================================ level C
def level_c():
    print("\nLEVEL C - full-pipeline outputs (after `python run_all.py`)")
    hint = "run `python run_all.py` (needs TensorFlow) first"

    # ---- Step 3
    if not skip_if_missing("C", "Step 3 metrics", ["step3_metrics.json"], hint):
        M3 = jload("step3_metrics.json")

        def s3():
            probs = []
            if "f1" not in M3.get("pixel_test", {}):
                probs.append("pixel_test.f1 missing")
            if "patch_test_ci95_block_bootstrap" not in M3:
                probs.append("no bootstrap confidence intervals")
            if "Youden" not in M3.get("threshold_source", ""):
                probs.append(f"threshold source is '{M3.get('threshold_source')}', not Youden")
            if not 0 < M3.get("threshold", 0) < 1:
                probs.append("threshold outside (0,1)")
            return ("FAIL", "; ".join(probs) + " - output is from the OLD code, re-run Step 3") if probs else \
                ("PASS", f"threshold {M3['threshold']:.3f} (Youden), pixel F1 {M3['pixel_test']['f1']:.2f}, CIs present")
        check("C", "Step 3 produced the new metrics", s3)

        def vs_baseline():
            a, b = M3["patch_test"]["auc"], M3["baseline_logreg_test"]["auc"]
            ci = (M3.get("patch_test_ci95_block_bootstrap") or {}).get("auc")
            rng = f" (95% CI {ci['low']:.2f}-{ci['high']:.2f})" if ci else ""
            return ("WARN", f"CNN AUC {a:.2f}{rng} does not beat the logistic baseline {b:.2f}: explanations describe a proxy formula") \
                if a <= b + 0.02 else ("PASS", f"CNN AUC {a:.2f}{rng} vs baseline {b:.2f}")
        check("C", "model vs simple baseline (honesty check)", vs_baseline)

    # ---- Step 4
    if not skip_if_missing("C", "Step 4 Grad-CAM sanity", ["step4_metrics.json"], hint):
        M4 = jload("step4_metrics.json")

        def s4():
            rho = M4.get("sanity_spearman_trained_vs_random")
            tr = (M4["focus_test_all"].get("high_risk") or {}).get("median")
            rn = (M4["focus_randomised_model"].get("high_risk") or {}).get("median")
            if rho is None or tr is None or rn is None:
                return "WARN", "sanity numbers unavailable"
            ok = abs(rho) < 0.3 and tr > rn
            return ("PASS" if ok else "FAIL",
                    f"trained-vs-random-model Spearman {rho:+.2f} (want ~0), high-risk enrichment trained {tr:.2f} vs random {rn:.2f}")
        check("C", "Grad-CAM depends on learned weights (randomisation test)", s4)

    # ---- Step 5
    if not skip_if_missing("C", "Step 5 SHAP checks", ["step5_metrics.json"], hint):
        M5 = jload("step5_metrics.json")

        def comp():
            c = M5["completeness"]
            rel = c.get("relative_mean_abs_err")
            if rel is None:
                return "FAIL", "no relative_mean_abs_err (old code) - re-run Step 5"
            s = f"SHAP misses {rel:.1%} of the effect (n_samples={M5['n_samples']}), bias {c.get('mean_signed_err', 0):+.3f}, R2 {c.get('r2_sum_phi_vs_target')}"
            return ("PASS" if rel <= 0.10 else "WARN"), s + ("" if rel <= 0.10 else " - raise SHAP_SAMPLES")
        check("C", "SHAP additivity (completeness)", comp)

        def lib():
            x = M5.get("shap_library_crosscheck") or {}
            if "pearson" in x:
                return ("PASS" if x["pearson"] >= 0.9 else "FAIL"), f"Pearson vs shap {x.get('shap_version')} = {x['pearson']:.3f}"
            return "WARN", f"cross-check did not run: {x.get('error', 'no result')}"
        check("C", "SHAP agrees with the `shap` library", lib)

        def san():
            s = M5["sanity_randomised_model"]
            if s.get("status") != "completed":
                return "WARN", str(s.get("status"))
            r = s["spearman_trained_vs_random"]
            return ("PASS" if r < 0.5 else "FAIL"), f"trained-vs-random SHAP Spearman {r:.2f} (want low)"
        check("C", "SHAP depends on learned weights (randomisation test)", san)

        def pred():
            d = M5.get("prediction_check_max_abs_difference")
            if d is None:
                return "FAIL", "missing (old Step 5)"
            return ("PASS" if d < 1e-3 else "FAIL"), f"Step 5 vs Step 4 probabilities differ by at most {d:.2e}"
        check("C", "same model behind Grad-CAM and SHAP", pred)

    # ---- Step 7
    if not skip_if_missing("C", "Step 7 validation checks", ["step7_metrics.json"], hint):
        M7 = jload("step7_metrics.json")

        def stab():
            st = M7["stability"]
            noisy, floor = st["shap"]["top20_iou"]["mean"], st["shap_noise_floor_same_input_new_seed"]["top20_iou"]["mean"]
            if noisy > floor * 1.15:
                return "FAIL", f"SHAP noisy-vs-clean IoU {noisy:.2f} exceeds its own noise floor {floor:.2f}: shared random seed (old code)"
            ok = noisy >= 0.9 * floor
            return ("PASS" if ok else "WARN"), f"SHAP clean-vs-noisy IoU {noisy:.2f} vs Monte-Carlo noise floor {floor:.2f}" + \
                ("" if ok else " - input noise degrades SHAP beyond its sampling noise")
        check("C", "stability test is no longer inflated", stab)

        def fid():
            res = []
            for k, v in M7["fidelity_deletion"].items():
                s = v["predicted_high"]
                res.append((k, s["mean_gain"], s["wilcoxon_p"]))
            bad = [k for k, g, p in res if k in ("gradcam", "shap") and not (g > 0 and p < 0.05)]
            txt = ", ".join(f"{k} {g:+.2f}" for k, g, _ in res)
            return ("FAIL", f"no significant fidelity gain for {bad}") if bad else ("PASS", f"deleting top pixels beats random: {txt}")
        check("C", "deletion test: explanations are faithful", fid)

        def agree():
            a = M7["spatial_agreement"]["predicted_high"]["gradcam~shap"]["top20_iou"]["mean"]
            ch = M7["spatial_agreement"]["chance_top20_iou"]
            return ("PASS" if a > 2 * ch else "WARN"), f"Grad-CAM~SHAP top-20% IoU {a:.2f} (chance {ch:.2f})"
        check("C", "Grad-CAM and SHAP agree above chance", agree)

    # ---- Step 9 (text layer)
    if not skip_if_missing("C", "Step 9 text layer", ["step8_explanations_test.csv", "step8_report.json"], hint):
        def text_layer():
            from src import text_explainer as T
            table, report = T.load_patch_table(OUT), T.load_report(OUT)
            extras = T.load_extras(OUT, table)
            n = 0
            for i, r in table.items():
                for s in T.STYLES:
                    t = T.explain_patch(r, s, extras.get(i))["text"]
                    assert t and "None" not in t, f"patch {i} style {s}"
                    n += 1
            for m in report["scene"]:
                assert T.explain_scene(report, m, "plain")["text"]
            return "PASS", f"{n} patch explanations ({len(table)} patches x {len(T.STYLES)} styles) + {len(report['scene'])} scene texts generated"
        check("C", "text layer explains every test patch", text_layer)

        def resid():
            from src import text_explainer as T
            table = T.load_patch_table(OUT)
            ex = T.load_extras(OUT, table)
            sh = sorted(e["resid_share"] for e in ex.values() if "resid_share" in e)
            if not sh:
                return "SKIP", "step5_shap_patches.npz not available"
            med, bad = sh[len(sh) // 2], sum(x >= 1.0 for x in sh)
            ok = med <= 0.15 and bad / len(sh) <= 0.05
            return ("PASS" if ok else "WARN"), (f"SHAP leaves a median {med:.0%} of each patch's score change unattributed; "
                                                f"{bad}/{len(sh)} patches have a residual >= 100% (their NDVI/NDWI split is flagged 'do not rely')"
                                                + ("" if ok else " - raise SHAP_SAMPLES"))
        check("C", "per-patch SHAP additivity (affects trust in the text)", resid)

    # ---- Step 8
    if not skip_if_missing("C", "Step 8 report", ["step8_report.md", "step8_report.json", "step8_explanations_test.csv"], hint):
        def rep():
            md = (OUT / "step8_report.md").read_text(encoding="utf-8")
            n = max(sum(1 for _ in open(OUT / "step8_explanations_test.csv", encoding="utf-8")) - 1, 0)
            has_ci = "block-bootstrap" in md
            return ("PASS" if has_ci else "WARN"), f"report + {n} per-patch explanations written" + ("" if has_ci else "; no confidence-interval line (old Step 8)")
        check("C", "Step 8 final report", rep)


# ============================================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--run-data", action="store_true", help="run Steps 1-2 on dataset/ before checking level B (needs rasterio)")
    ap.add_argument("--skip-tests", action="store_true", help="skip the unit-test runs in level A")
    ap.add_argument("--json", metavar="FILE", help="also write the results as JSON")
    a = ap.parse_args()

    print(f"Verifying: {ROOT}")
    if a.level in ("A", "all"):
        level_a(a.skip_tests)
    if a.level in ("B", "all"):
        level_b(a.run_data)
    if a.level in ("C", "all"):
        level_c()

    n = {k: sum(1 for r in RESULTS if r[2] == k) for k in ICON}
    print("\n" + "=" * 70)
    print(f"SUMMARY: {n['PASS']} passed | {n['WARN']} warnings | {n['FAIL']} failed | {n['SKIP']} skipped")
    for lvl, name, st, det in RESULTS:
        if st == "FAIL":
            print(f"  FAILED [{lvl}] {name}: {det}")
    if n["FAIL"] == 0 and n["SKIP"]:
        print("Skipped checks need Steps 2-8 outputs: run `python verify_changes.py --run-data`, then `python run_all.py`, then this again.")
    print("=" * 70)
    if a.json:
        Path(a.json).write_text(json.dumps([dict(level=l, check=c, status=s, detail=d) for l, c, s, d in RESULTS], indent=1), encoding="utf-8")
    sys.exit(1 if n["FAIL"] else 0)


if __name__ == "__main__":
    main()
