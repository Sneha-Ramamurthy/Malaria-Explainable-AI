"""Backend-free unit tests of the explainable-AI numerics (no TensorFlow / Keras / model needed).

Run either way, from the project root:
    python -m pytest tests -q
    python -m tests.test_xai_core

Every test uses a function whose correct answer is known analytically, so a failure means the XAI maths is wrong
(not that a trained model happens to behave oddly).
"""
from types import SimpleNamespace

import numpy as np

from src import validation_core as V
from src import xai_utils as U
from src.shap_explain import channel_contributions, completeness, evidence_map, expected_gradients


# ----------------------------------------------------------------------------- helpers
def _linear_fn(w):
    """f(x) = sum(w * x) -> (values (B,), grads (B,H,W,C)); the exact Shapley value of pixel i is w_i * (x_i - E[b_i])."""
    w = np.asarray(w, np.float32)
    return lambda x: ((x * w).reshape(len(x), -1).sum(1), np.broadcast_to(w, x.shape).copy())


def _quadratic_fn(x):
    """f(x) = sum(x^2): smooth, non-linear, gradient 2x."""
    return (x ** 2).reshape(len(x), -1).sum(1), 2 * x


def _data(n=6, h=4, w=4, c=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, h, w, c)).astype(np.float32), rng.normal(size=(8, h, w, c)).astype(np.float32)


# ----------------------------------------------------------------------------- SHAP / expected gradients
def test_expected_gradients_recovers_exact_shapley_for_linear_model():
    X, bg = _data()
    w = np.random.default_rng(1).normal(size=X.shape[1:]).astype(np.float32)
    phi, base = expected_gradients(_linear_fn(w), X, bg, n_samples=4000, seed=0)
    exact = w * (X - bg.mean(0))
    assert np.abs(phi - exact).max() < 0.25, "phi should converge to w*(x - E[b])"
    assert abs(base - float((bg * w).reshape(len(bg), -1).sum(1).mean())) < 1e-4


def test_expected_gradients_completeness_for_nonlinear_model():
    X, bg = _data()
    fn = _quadratic_fn
    phi, base = expected_gradients(fn, X, bg, n_samples=3000, seed=0)
    fx = fn(X)[0]
    c = completeness(phi, fx, base)
    assert c["relative_mean_abs_err"] < 0.05, c
    assert abs(c["mean_signed_err"]) < 0.1 * c["mean_abs_target"], "estimator must not be biased"
    assert c["r2_sum_phi_vs_target"] > 0.99, c


def test_expected_gradients_error_shrinks_with_more_samples():
    X, bg = _data()
    errs = []
    for n in (10, 100, 1000):
        phi, base = expected_gradients(_quadratic_fn, X, bg, n_samples=n, seed=3)
        errs.append(completeness(phi, _quadratic_fn(X)[0], base)["mean_abs_err"])
    assert errs[0] > errs[1] > errs[2], f"completeness error should fall as samples grow, got {errs}"


def test_expected_gradients_gives_zero_to_a_feature_the_model_ignores():
    X, bg = _data()
    w = np.zeros(X.shape[1:], np.float32)
    w[..., 0] = 1.0                                        # only channel 0 matters
    phi, _ = expected_gradients(_linear_fn(w), X, bg, n_samples=50, seed=0)
    assert np.all(phi[..., 1] == 0), "null feature must receive exactly zero attribution"
    assert np.abs(phi[..., 0]).max() > 0


def test_expected_gradients_is_reproducible_for_a_seed():
    X, bg = _data()
    a, _ = expected_gradients(_quadratic_fn, X, bg, n_samples=30, seed=5)
    b, _ = expected_gradients(_quadratic_fn, X, bg, n_samples=30, seed=5)
    c, _ = expected_gradients(_quadratic_fn, X, bg, n_samples=30, seed=6)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_expected_gradients_is_batch_size_invariant():
    X, bg = _data(n=10)
    a, ba = expected_gradients(_quadratic_fn, X, bg, n_samples=40, batch=4, seed=2)
    assert a.shape == X.shape and np.isfinite(a).all() and np.isfinite(ba)


def test_completeness_metric_is_perfect_for_perfect_attribution_and_flags_bias():
    phi = np.ones((5, 2, 2, 1), np.float32)               # sum = 4 per patch
    target_base = 1.0
    perfect = completeness(phi, np.full(5, 5.0) + np.arange(5) * 0, target_base)
    assert perfect["mean_abs_err"] < 1e-6 and perfect["mean_signed_err"] < 1e-6
    biased = completeness(phi, np.full(5, 7.0), target_base)          # sum(phi) under-explains by 2
    assert abs(biased["mean_signed_err"] + 2.0) < 1e-6 and biased["relative_mean_abs_err"] > 0.3


def test_channel_contributions_respect_valid_mask_and_add_up():
    phi = np.ones((2, 3, 3, 2), np.float32)
    valid = np.zeros((2, 3, 3), bool); valid[:, :2, :] = True          # 6 of 9 pixels
    assert np.allclose(channel_contributions(phi, valid), 6.0)
    assert np.allclose(channel_contributions(phi), 9.0)


def test_evidence_map_is_positive_part_masked_and_unit_scaled():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(4, 16, 16, 2)).astype(np.float32)
    valid = np.ones((4, 16, 16), bool); valid[:, :, :4] = False
    ev = evidence_map(phi, valid, sigma=1.0)
    assert ev.min() >= 0 and abs(ev.max() - 1.0) < 1e-5
    assert np.all(ev[~valid] == 0), "no evidence outside observed pixels"
    all_negative = evidence_map(-np.abs(phi), valid, 1.0)
    assert np.all(all_negative == 0), "pixels that only push risk DOWN must carry no 'evidence for high risk'"


# ----------------------------------------------------------------------------- map helpers
def test_minmax01_scales_valid_pixels_and_zeroes_invalid():
    a = np.array([[[1., 2.], [3., 100.]]]); v = np.array([[[True, True], [True, False]]])
    out = U.minmax01(a, v)
    assert out[0, 0, 0] == 0 and out[0, 1, 0] == 1 and out[0, 1, 1] == 0


def test_top_region_has_requested_size_and_ignores_non_positive_pixels():
    rng = np.random.default_rng(0)
    c = rng.random((64, 64)); v = np.ones((64, 64), bool)
    assert abs(U.top_region(c, v, 0.2).mean() - 0.2) < 0.01
    sparse = np.zeros((64, 64)); sparse[:5, :5] = 1.0                     # Grad-CAM-like: mostly exact zeros
    t = U.top_region(sparse, v, 0.2)
    assert t.sum() == 25 and not t[10:, 10:].any(), "zero-valued pixels must never be 'important'"


def test_chance_iou_constant_matches_simulation():
    rng = np.random.default_rng(0)
    ious = []
    for _ in range(300):
        a, b = rng.random((64, 64)), rng.random((64, 64))
        ta, tb = U.top_region(a, np.ones_like(a, bool)), U.top_region(b, np.ones_like(b, bool))
        ious.append((ta & tb).sum() / (ta | tb).sum())
    assert abs(np.mean(ious) - U.CHANCE_TOP20_IOU) < 0.01


def test_pair_agreement_identical_vs_independent_maps():
    rng = np.random.default_rng(1)
    a = rng.random((40, 32, 32)).astype(np.float32); b = rng.random((40, 32, 32)).astype(np.float32)
    v = np.ones((40, 32, 32), bool)
    rho, iou = V.pair_agreement(a, a, v)
    assert np.allclose(rho, 1) and np.allclose(iou, 1)
    rho, iou = V.pair_agreement(a, b, v)
    assert abs(np.nanmean(rho)) < 0.1 and abs(np.nanmean(iou) - V.CHANCE) < 0.02
    r2, i2 = U.patch_agreement(a, a, v, range(len(a)))                      # xai_utils version agrees
    assert np.allclose(r2, 1) and np.allclose(i2, 1)


# ----------------------------------------------------------------------------- fusion
def test_rank01_is_monotone_scaled_and_masked():
    a = np.array([[5., 1., 3., 3., 99.]]); v = np.array([[True, True, True, True, False]])
    r = V.rank01(a, v)
    assert r[0, 4] == 0 and r[0, 1] == 0 and r[0, 0] == 1
    assert r[0, 2] == r[0, 3], "ties get the same (average) rank"
    assert 0 < r[0, 2] < 1


def test_fuse_is_scale_free_and_stays_inside_valid_pixels():
    rng = np.random.default_rng(2)
    a = rng.random((3, 16, 16)).astype(np.float32); v = np.ones((3, 16, 16), bool); v[:, :3] = False
    f1 = V.fuse([a, a * 1000], v)                                           # same ordering, wildly different scale
    f2 = V.fuse([a], v)
    assert np.allclose(f1, f2, atol=1e-6), "rank fusion must not care about each method's scale"
    assert f1.min() >= 0 and f1.max() <= 1 + 1e-6 and np.all(f1[~v] == 0)


def test_agreement_count_counts_methods_per_pixel():
    v = np.ones((1, 10, 10), bool)
    a = np.arange(100, dtype=np.float32).reshape(1, 10, 10) + 1
    three = V.agreement_count([a, a, a], v, q=0.2)
    assert three.max() == 3 and (three == 3).sum() == 20 and (three == 0).sum() == 80
    mixed = V.agreement_count([a, a, np.flip(a)], v, q=0.2)                  # flipped map's top-20% is the opposite end
    assert (mixed == 2).sum() == 20 and (mixed == 1).sum() == 20 and (mixed == 0).sum() == 60


# ----------------------------------------------------------------------------- fidelity (deletion test)
def test_deletion_test_rewards_a_faithful_map_and_punishes_a_wrong_one():
    rng = np.random.default_rng(0)
    N, S = 12, 16
    X = (rng.random((N, S, S, 2)).astype(np.float32) + 1.0)
    valid = np.ones((N, S, S), bool)
    region = np.zeros((S, S), bool); region[:6, :6] = True                      # the only pixels the model uses
    logit_fn = lambda x: (x[:, region, 0]).sum(1).astype(np.float64)
    faithful = np.broadcast_to(region.astype(np.float32), (N, S, S)) + 0.01 * rng.random((N, S, S)).astype(np.float32)
    wrong = np.broadcast_to((~region).astype(np.float32), (N, S, S)) + 0.01 * rng.random((N, S, S)).astype(np.float32)
    fill = np.zeros(2, np.float32)
    D = V.deletion_test(logit_fn, X, valid, {"faithful": faithful, "wrong": wrong}, fill, (0.05, 0.1, 0.2), n_random=8, seed=0)
    idx = np.arange(N)
    good = V.summarise_deletion(D["faithful"]["top"], D["random"], D["faithful"]["bottom"], (0.05, 0.1, 0.2), idx)
    bad = V.summarise_deletion(D["wrong"]["top"], D["random"], D["wrong"]["bottom"], (0.05, 0.1, 0.2), idx)
    assert good["mean_gain"] > 0 and good["share_top_gt_random"] == 1.0
    assert bad["mean_gain"] < 0 and bad["aopc_top"] < bad["aopc_random"]
    assert good["aopc_top"] > good["aopc_random"] > good["aopc_bottom"] - 1e-9


def test_delete_pixels_only_touches_requested_fraction_of_valid_pixels():
    X = np.ones((1, 10, 10, 2), np.float32); valid = np.zeros((1, 10, 10), bool); valid[0, :5] = True
    m = np.arange(100, dtype=np.float32).reshape(1, 10, 10)
    orders = V.deletion_orders(m, valid, np.random.default_rng(0))
    out = V.delete_pixels(X, orders, 0.2, np.array([0., 0.], np.float32), "top")
    assert (out[0, ..., 0] == 0).sum() == 10                                    # 20% of the 50 valid pixels
    assert np.all(out[0, 5:] == 1), "invalid pixels are never deleted"


def test_add_noise_only_changes_valid_pixels():
    X = np.zeros((2, 8, 8, 2), np.float32); valid = np.zeros((2, 8, 8), bool); valid[:, :4] = True
    Xn = V.add_noise(X, valid, sd=np.array([0.1, 0.2]), level=1.0, seed=0)
    assert np.all(Xn[:, 4:] == 0) and np.abs(Xn[:, :4]).max() > 0


# ----------------------------------------------------------------------------- scenes
def _ctx(P=8, H=20, W=21):
    return SimpleNamespace(PATCH=P, H=H, W=W)


def test_scene_valid_prefers_land_mask_and_falls_back_to_valid():
    both = {"valid": np.ones((3, 3), bool), "valid_risk": np.zeros((3, 3), bool)}
    assert not U.scene_valid(both).any()
    assert U.scene_valid({"valid": np.ones((3, 3), bool)}).all()


def test_scene_windows_use_land_mask_include_edges_and_stitch_back():
    ctx = _ctx()
    ndvi = np.random.default_rng(0).random((20, 21)).astype(np.float32); ndwi = ndvi * 0.5
    land = np.ones((20, 21), bool); land[:, 10:13] = False                      # a river
    sc = {"ndvi": ndvi, "ndwi": ndwi, "valid": np.ones((20, 21), bool), "valid_risk": land}
    rs, cs, wins = U.scene_windows(sc, ctx, stride=4)
    assert wins.shape[1:] == (8, 8, 2) and rs.max() == ctx.H - 8 and cs.max() == ctx.W - 8, "last row/col is edge-aligned"
    r, c = int(rs[0]), int(cs[0])
    assert np.all(wins[0][~land[r:r + 8, c:c + 8]] == 0), "non-land pixels must be zeroed exactly as in training patches"
    out = U.stitch(rs, cs, np.ones(wins.shape[:3], np.float32), ctx)
    assert np.nanmin(out) == 1 and np.nanmax(out) == 1                         # averaging ones gives ones
    out2 = U.stitch(rs[:1], cs[:1], np.ones((1, 8, 8), np.float32), ctx)
    assert np.isnan(out2[-1, -1]) and out2[0, 0] == 1                          # uncovered pixels stay NaN


# ----------------------------------------------------------------------------- final-output helpers
def test_decision_helpers():
    assert V.risk_level(0.2, 0.15) == "HIGH" and V.risk_level(0.1, 0.15) == "ELEVATED" and V.risk_level(0.01, 0.15) == "LOW"
    assert abs(V.decision_margin(0.5, 0.5)) < 1e-9 and V.decision_margin(0.9, 0.5) > 0 > V.decision_margin(0.1, 0.5)
    assert V.confidence_band(0.1) == "borderline" and V.confidence_band(-1.0) == "moderate" and V.confidence_band(3) == "strong"
    assert V.support_level([0.5, 0.5, 0.0])[0] == "strong" and V.support_level([0.5, 0.0, 0.0])[0] == "partial"
    assert V.support_level([0.0, 0.0, np.nan])[0] == "weak"


def test_explain_text_reports_signs_and_does_not_crash_on_missing_values():
    r = dict(level="HIGH", prob=0.4, thr=0.15, margin=1.2, band="moderate", ndvi_phi=2.0, ndwi_phi=-0.5, base=-1.0,
             enrich=dict(vegetated=2.0, moist=np.nan, near_water=1.0, high_risk=2.5), iou_gs=0.3, iou_gc=0.1, iou_sc=0.12,
             support="partial", n_pairs=1, del_top=3.0, del_rand=0.5)
    txt = V.explain_text(r)
    assert "NDVI is the dominant driver and pushes the risk UP" in txt and "partial support" in txt
    r.update(del_top=np.nan, enrich={})
    assert "Fidelity" not in V.explain_text(r)


# ----------------------------------------------------------------------------- tiny runner (works without pytest)
if __name__ == "__main__":
    import sys
    import traceback
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except Exception:                                                      # noqa: BLE001
            failed += 1; print(f"FAIL  {name}"); traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
