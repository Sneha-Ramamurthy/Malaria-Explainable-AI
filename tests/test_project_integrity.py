"""Cheap structural checks that would have caught the original packaging errors (no TensorFlow needed)."""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_pipeline_module_in_run_all_exists():
    mods = re.findall(r'"module": "(src\.[A-Za-z0-9_]+)"', (ROOT / "run_all.py").read_text())
    assert len(mods) == 9
    for m in mods:
        assert (ROOT / (m.replace(".", "/") + ".py")).exists(), f"run_all.py refers to missing module {m}"


def test_all_python_files_parse():
    for f in list(ROOT.glob("src/*.py")) + list(ROOT.glob("tests/*.py")) + [ROOT / "run_all.py"]:
        ast.parse(f.read_text(encoding="utf-8"), str(f))           # catches e.g. repeated keyword arguments


def test_scripts_that_define_main_have_a_correct_main_guard():
    f = ROOT / "src" / "step1_data_audit.py"
    assert '__name__ == "__main__"' in f.read_text() and "_name_ ==" not in f.read_text().replace("__name__", "")


def test_scene_consumers_use_the_land_mask_helper():
    for name in ("step6_cbam", "step7_validation", "step8_final_output"):
        text = (ROOT / "src" / f"{name}.py").read_text()
        assert 'sv = s["valid"]' not in text and 's["valid"] &' not in text, f"{name} still uses the water-inclusive mask"


def test_lock_file_is_utf8_without_conflicting_keras():
    raw = (ROOT / "requirements-lock.txt").read_bytes()
    assert not raw.startswith((b"\xff\xfe", b"\xfe\xff")) and b"\r" not in raw
    assert "keras-nightly" not in raw.decode().lower()


if __name__ == "__main__":
    import sys, traceback
    bad = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            try: v(); print("PASS ", k)
            except Exception: bad += 1; print("FAIL ", k); traceback.print_exc()
    sys.exit(1 if bad else 0)
