"""Tests for the TensorFlow-free Step 3 helpers (run: python -m tests.test_train_utils)."""
import numpy as np

from src.train_utils import block_bootstrap_ci, youden_threshold


def test_youden_threshold_finds_the_separating_cutoff():
    y = np.array([0] * 50 + [1] * 50); p = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    t = youden_threshold(y, p)
    assert 0.4 < t <= 0.6 + 1e-9 and ((p >= t) == y).all()


def test_youden_threshold_is_prevalence_independent_and_safe():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int); p = np.clip(0.3 + 0.4 * y + 0.15 * rng.normal(size=2000), 0, 1)
    assert 0.4 < youden_threshold(y, p) < 0.6                                 # ~midpoint despite 5% positives
    assert youden_threshold(np.zeros(10, int), rng.random(10)) == 0.5          # one class -> fallback
    assert youden_threshold(np.array([0, 1]), np.array([0.3, 0.3])) == 0.5     # constant predictions -> fallback
    assert 0 < youden_threshold(np.array([0, 1]), np.array([0.0, 1.0])) < 1


def test_block_bootstrap_ci_brackets_the_point_estimate_and_widens_with_less_data():
    rng = np.random.default_rng(1)
    def make(nb):
        blocks = np.repeat(np.arange(nb), 10); y = rng.integers(0, 2, nb * 10)
        p = np.clip(0.5 + 0.25 * (y - 0.5) * 2 + 0.25 * rng.normal(size=nb * 10), 0, 1)
        return y, p, blocks
    y, p, b = make(60)
    ci = block_bootstrap_ci(y, p, 0.5, b, n_boot=300)
    from sklearn.metrics import roc_auc_score
    assert ci["auc"]["low"] < roc_auc_score(y, p) < ci["auc"]["high"] and ci["n_boot_used"] > 250
    y2, p2, b2 = make(6)
    ci2 = block_bootstrap_ci(y2, p2, 0.5, b2, n_boot=300)
    assert (ci2["auc"]["high"] - ci2["auc"]["low"]) > (ci["auc"]["high"] - ci["auc"]["low"])


if __name__ == "__main__":
    import sys, traceback
    bad = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            try: v(); print("PASS ", k)
            except Exception: bad += 1; print("FAIL ", k); traceback.print_exc()
    sys.exit(1 if bad else 0)
