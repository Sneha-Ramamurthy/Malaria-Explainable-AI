"""Backend-free tests for Steps 5-6 numerics.  Run:  python -m pytest tests -q   (or: python tests/test_xai_core.py)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from src.shap_explain import expected_gradients, completeness, channel_contributions
from src import xai_utils as U


def _toy(seed=0):
    r = np.random.default_rng(seed); w1, w2 = r.normal(size=(8, 8)), r.normal(size=(8, 8))
    def vag(x):
        z = (w1[None] * x[..., 0] ** 2).sum((1, 2)) + (w2[None] * np.tanh(x[..., 1])).sum((1, 2)) - 3
        return z, np.stack([2 * w1[None] * x[..., 0], w2[None] * (1 - np.tanh(x[..., 1]) ** 2)], -1)
    return r, vag, w2


def test_completeness():
    r, vag, _ = _toy()
    X = r.normal(size=(40, 8, 8, 2)).astype(np.float32); bg = r.normal(size=(50, 8, 8, 2)).astype(np.float32)
    phi, base = expected_gradients(vag, X, bg, n_samples=300, batch=16)
    assert completeness(phi, vag(X)[0], base)["pearson_sum_phi_vs_target"] > 0.99


def test_zero_weight_feature_gets_zero():
    r, vag, w2 = _toy(); w2[:] = 0
    X = r.normal(size=(10, 8, 8, 2)).astype(np.float32); bg = r.normal(size=(20, 8, 8, 2)).astype(np.float32)
    phi, _ = expected_gradients(vag, X, bg, n_samples=50)
    assert np.abs(channel_contributions(phi)[:, 1]).max() < 1e-6


def test_agreement_identity_and_chance():
    r = np.random.default_rng(0); a = r.random((6, 64, 64)); v = np.ones_like(a, bool)
    rho, iou = U.patch_agreement(a, a, v, range(6)); assert rho.mean() > .999 and iou.mean() > .999
    rho, iou = U.patch_agreement(a, r.random(a.shape), v, range(6)); assert abs(iou.mean() - U.CHANCE_TOP20_IOU) < .03


if __name__ == "__main__":
    test_completeness(); test_zero_weight_feature_gets_zero(); test_agreement_identity_and_chance(); print("all passed")
