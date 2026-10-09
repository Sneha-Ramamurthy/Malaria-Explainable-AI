"""Step 5 - SHAP expected-gradient explanations for the risk classifier."""

import os
import json
import hashlib
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras

from scipy.stats import spearmanr
from .config import OUT_DIR, ROOT
from .gradcam import load_model
from .models import build_model
from .shap_explain import (
    LogitGradFn,
    expected_gradients,
    completeness,
    channel_contributions,
    evidence_map,
    shap_library_crosscheck,
    FEATURES,
)
from . import xai_utils as U


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


CACHE = env_flag("CACHE", False)
SHAP_LIB = env_flag("SHAP_LIB", True)
N_SAMPLES = int(os.environ.get("SHAP_SAMPLES", "100"))
SCENE_SAMPLES = int(os.environ.get("SCENE_SAMPLES", "30"))
N_BG = 100
SIGMA = 1.5
SCENE_STRIDE = 32

if N_SAMPLES < 1 or SCENE_SAMPLES < 1:
    raise ValueError("SHAP_SAMPLES and SCENE_SAMPLES must be positive integers.")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(value):
    """Convert a scalar to a JSON-safe float or None."""
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def safe_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def json_safe(value):
    """Recursively replace NumPy objects and non-finite values for JSON."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return safe_float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    z = np.clip(z, -80, 80)
    return 1.0 / (1.0 + np.exp(-z))


def cache_matches(path, shapes):
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return all(
                name in data and data[name].shape == shape
                for name, shape in shapes.items()
            )
    except (OSError, ValueError, KeyError):
        return False


def correlation(x, y):
    """Return a finite Spearman correlation or None."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(x, y)
    return safe_float(result.statistic)


def overlay(ax, base_img, explanation, cmap, vmin, vmax, title):
    ax.imshow(base_img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.imshow(
        np.ma.masked_where(
            ~np.isfinite(explanation) | (explanation < 0.15),
            explanation,
        ),
        cmap="jet",
        alpha=0.55,
        vmin=0,
        vmax=1,
    )
    ax.set_title(title, fontsize=8)
    ax.axis("off")


# ---------------------------------------------------------------------
# 1. Load and validate inputs

ctx = U.load_context()

X = np.asarray(ctx.X, dtype=np.float32)
y = np.asarray(ctx.y).astype(int)
split = np.asarray(ctx.split).astype(int)
valid = np.asarray(ctx.valid, dtype=bool)
hrm = np.asarray(ctx.hrm, dtype=bool)
meta = np.asarray(ctx.meta)

THR = float(ctx.THR)
MONTHS = list(ctx.MONTHS)
PATCH, H, W = int(ctx.PATCH), int(ctx.H), int(ctx.W)

if X.ndim != 4 or X.shape[1:] != (PATCH, PATCH, 2):
    raise ValueError(f"Unexpected patch array shape: {X.shape}")

if len(X) == 0 or len(y) != len(X) or len(split) != len(X):
    raise ValueError("Patch data, labels and split arrays are empty or misaligned.")

if valid.shape != X.shape[:3] or hrm.shape != X.shape[:3]:
    raise ValueError("Patch validity/high-risk masks do not match X.")

if meta.shape[0] != len(X):
    raise ValueError("Patch metadata does not match the patch count.")

if not np.isfinite(X).all():
    raise ValueError("Patch inputs contain NaN or infinite values.")

if not 0 < THR < 1:
    raise ValueError(f"Invalid probability threshold: {THR}")

tr_idx = np.flatnonzero(split == 0)
idx_te = np.flatnonzero(split == 2)

if len(tr_idx) == 0:
    raise ValueError("No training patches (split == 0).")

if len(idx_te) == 0:
    raise ValueError("No test patches (split == 2).")

if not np.any(valid[tr_idx]):
    raise ValueError("Training patches contain no valid pixels.")

masks = U.env_masks(ctx)

model_path = ROOT / "models" / "mobilenetv2_cbam.keras"
if not model_path.exists():
    raise FileNotFoundError(f"Trained model not found: {model_path}")

gradcam_path = OUT_DIR / "step4_gradcam_patches.npz"
if not gradcam_path.exists():
    raise FileNotFoundError(
        "Missing step4_gradcam_patches.npz. Complete Step 4 first."
    )

with np.load(gradcam_path, allow_pickle=False) as data:
    cam = np.asarray(data["cam"], dtype=np.float32)
    prob_step4 = np.asarray(data["prob"], dtype=np.float32)

if cam.shape != X.shape[:3] or prob_step4.shape != (len(X),):
    raise ValueError("Step 4 Grad-CAM cache does not match Step 2 patches.")

model = load_model(model_path)
fn = LogitGradFn(model)

LOGIT_THR = float(np.log(THR / (1.0 - THR)))
pf = OUT_DIR / "step5_shap_patches.npz"

# ---------------------------------------------------------------------
# 2. Patch-level expected gradients

rng = np.random.RandomState(0)
n_bg = min(N_BG, len(tr_idx))
bg_idx = np.sort(rng.choice(tr_idx, size=n_bg, replace=False))
bg = X[bg_idx].astype(np.float32, copy=False)

# Reuse a cache only if its dimensions, background selection, sample
# count and predictions agree with the current inputs and model.
patch_cache_ok = CACHE and cache_matches(
    pf,
    {
        "phi": X.shape,
        "logit": (len(X),),
        "prob": (len(X),),
        "base_value": (),
        "bg_idx": bg_idx.shape,
        "n_samples": (),
    },
)

if patch_cache_ok:
    try:
        with np.load(pf, allow_pickle=False) as data:
            patch_cache_ok = (
                np.array_equal(data["bg_idx"], bg_idx)
                and int(data["n_samples"].item()) == N_SAMPLES
            )
            if patch_cache_ok:
                phi = np.asarray(data["phi"], dtype=np.float32)
                base = float(data["base_value"].item())
                logit = np.asarray(data["logit"], dtype=np.float32)
                cached_prob = np.asarray(data["prob"], dtype=np.float32)
    except (OSError, ValueError, KeyError):
        patch_cache_ok = False

if patch_cache_ok:
    current_logit = np.asarray(fn.logit(X), dtype=np.float32)
    patch_cache_ok = (
        np.allclose(logit, current_logit, rtol=1e-3, atol=1e-3)
        and np.allclose(
            cached_prob, sigmoid(current_logit), rtol=1e-3, atol=1e-3
        )
    )

if not patch_cache_ok:
    logit = np.asarray(fn.logit(X), dtype=np.float32)
    phi, base = expected_gradients(
        fn, X, bg, n_samples=N_SAMPLES, batch=64, seed=0
    )
    phi = np.asarray(phi, dtype=np.float32)

p_chk = sigmoid(logit).astype(np.float32)
prob_diff = float(np.max(np.abs(p_chk - prob_step4)))
comp = completeness(phi, logit, base)

print(
    f"SHAP patches: {len(X)} | background: {n_bg} | "
    f"base logit: {base:.3f} | "
    f"max probability difference vs Step 4: {prob_diff:.5f}"
)
print("Completeness:", comp)

fc = channel_contributions(phi, valid)
resid = logit - base - fc.sum(axis=1)
ev = evidence_map(phi, valid, SIGMA)

np.savez_compressed(
    pf,
    phi=phi.astype(np.float32),
    base_value=np.float32(base),
    logit=logit.astype(np.float32),
    prob=p_chk.astype(np.float32),
    feature_sum=fc.astype(np.float32),
    evidence=ev.astype(np.float16),
    bg_idx=bg_idx,
    n_samples=np.int32(N_SAMPLES),
)

# Optional independent SHAP-library cross-check.
xc = None
if SHAP_LIB:
    try:
        xc = shap_library_crosscheck(
            fn,
            bg[:min(50, len(bg))],
            X[idx_te[:min(16, len(idx_te))]],
            nsamples=100,
        )
    except Exception as exc:
        xc = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    print("SHAP library cross-check:", xc)

# ---------------------------------------------------------------------
# 3. Global patch statistics and Grad-CAM agreement

pred_hi = idx_te[logit[idx_te] >= LOGIT_THR]
pred_lo = idx_te[logit[idx_te] < LOGIT_THR]


def feature_summary(indices):
    if len(indices) == 0:
        return {
            "n": 0,
            "share_NDVI": None,
            "share_NDWI": None,
            "mean_abs_NDVI": None,
            "mean_abs_NDWI": None,
            "mean_signed_NDVI": None,
            "mean_signed_NDWI": None,
        }

    a = np.abs(phi[indices] * valid[indices][..., None]).sum(axis=(1, 2))
    totals = a.sum(axis=1, keepdims=True)
    fractions = np.divide(
        a, totals, out=np.zeros_like(a), where=totals > 0
    )

    return {
        "n": int(len(indices)),
        "share_NDVI": safe_mean(fractions[:, 0]),
        "share_NDWI": safe_mean(fractions[:, 1]),
        "mean_abs_NDVI": safe_mean(a[:, 0]),
        "mean_abs_NDWI": safe_mean(a[:, 1]),
        "mean_signed_NDVI": safe_mean(fc[indices, 0]),
        "mean_signed_NDWI": safe_mean(fc[indices, 1]),
    }


G = {
    "all_test": feature_summary(idx_te),
    "predicted_high": feature_summary(pred_hi),
    "predicted_low": feature_summary(pred_lo),
}
print("Feature importance:", G)

st_all = U.focus_stats(ev, ctx, masks, idx_te)
st_pos = U.focus_stats(ev, ctx, masks, pred_hi)
S_all = U.summarise(st_all)
S_pos = U.summarise(st_pos)

rho, iou = U.patch_agreement(ev, cam, valid, idx_te)
rho_p, iou_p = U.patch_agreement(ev, cam, valid, pred_hi)

AG = {
    "all_test": {
        "spearman": U.describe(rho),
        "top20_iou": U.describe(iou),
    },
    "predicted_high": {
        "spearman": U.describe(rho_p),
        "top20_iou": U.describe(iou_p),
    },
    "chance_top20_iou": U.CHANCE_TOP20_IOU,
}

print(
    "SHAP vs Grad-CAM:",
    f"Spearman={safe_mean(rho)}, IoU={safe_mean(iou)},",
    f"chance IoU={U.CHANCE_TOP20_IOU:.3f}",
)

# ---------------------------------------------------------------------
# 4. Randomised-model sanity check
#
# Do not assume metadata fields exist. If they are absent, report that
# this check was skipped rather than stopping the entire SHAP workflow.

sanity = {"status": "skipped: model reconstruction metadata unavailable"}

try:
    model_config = ctx.M3.get("model_config")
    norm_mean = ctx.M3.get("norm_mean")
    norm_var = ctx.M3.get("norm_var")

    if (
        isinstance(model_config, dict)
        and norm_mean is not None
        and norm_var is not None
    ):
        keras.utils.set_random_seed(123)
        rnd = build_model(norm_mean, norm_var, **model_config)
        fn_r = LogitGradFn(rnd)

        phi_r, _ = expected_gradients(
            fn_r,
            X[idx_te],
            bg,
            n_samples=max(20, N_SAMPLES // 2),
            batch=64,
            seed=1,
        )
        ev_r = evidence_map(
            np.asarray(phi_r, dtype=np.float32),
            valid[idx_te],
            SIGMA,
        )

        rho_r = []
        for a, b, v in zip(ev[idx_te], ev_r, valid[idx_te]):
            if v.sum() > 100:
                r = correlation(a[v], b[v])
                if r is not None:
                    rho_r.append(r)

        ev_r_full = np.zeros_like(ev)
        ev_r_full[idx_te] = ev_r
        S_rnd = U.summarise(
            U.focus_stats(ev_r_full, ctx, masks, idx_te)
        )

        trained_high = S_all.get("high_risk")
        random_high = S_rnd.get("high_risk")

        sanity = {
            "status": "completed",
            "spearman_trained_vs_random": safe_mean(rho_r),
            "high_risk_enrichment_trained": (
                trained_high.get("median") if trained_high else None
            ),
            "high_risk_enrichment_random": (
                random_high.get("median") if random_high else None
            ),
        }
    else:
        print("Skipping randomised-model check: metadata keys are missing.")
except Exception as exc:
    sanity = {
        "status": "failed",
        "error": f"{type(exc).__name__}: {str(exc)[:250]}",
    }
    print("Randomised-model check could not complete:", sanity["error"])

print("Randomised-model sanity check:", sanity)

# ---------------------------------------------------------------------
# 5. Scene-level SHAP maps

region = U.region_map(ctx)
scene_ev = {}
scene_stats = {}

for month in MONTHS:
    scene = ctx.scene[month]
    cache_file = OUT_DIR / f"step5_scene_shap_{month}.npz"

    scene_valid = np.asarray(
        scene["valid_risk"] if "valid_risk" in scene.files else scene["valid"],
        dtype=bool,
    )
    scene_for_shap = {k: scene[k] for k in scene.files}
    scene_for_shap["valid"] = scene_valid

    expected_shapes = {
        "evidence": (H, W),
        "ndvi_phi": (H, W),
        "ndwi_phi": (H, W),
    }
    use_scene_cache = CACHE and cache_matches(cache_file, expected_shapes)

    # Scene caches are only used when explicitly requested. Shape checks
    # cannot prove the model/data/settings match, so default CACHE=False.
    if use_scene_cache:
        print(
            f"Using existing scene cache for {month}; "
            "ensure it belongs to the current model and inputs."
        )
        with np.load(cache_file, allow_pickle=False) as data:
            evs = np.asarray(data["evidence"], dtype=np.float32)
            pn = np.asarray(data["ndvi_phi"], dtype=np.float32)
            pw = np.asarray(data["ndwi_phi"], dtype=np.float32)
    else:
        rs, cs, wins = U.scene_windows(
            scene_for_shap, ctx, stride=SCENE_STRIDE
        )

        if len(wins) == 0:
            print(f"WARNING: no valid SHAP windows for {month}.")
            evs = np.full((H, W), np.nan, dtype=np.float32)
            pn = np.full((H, W), np.nan, dtype=np.float32)
            pw = np.full((H, W), np.nan, dtype=np.float32)
        else:
            ph, _ = expected_gradients(
                fn,
                wins,
                bg,
                n_samples=SCENE_SAMPLES,
                batch=128,
                seed=2,
            )
            ph = np.asarray(ph, dtype=np.float32)

            wv = np.stack([
                scene_valid[r:r + PATCH, c:c + PATCH]
                for r, c in zip(rs, cs)
            ])

            evw = evidence_map(ph, wv, SIGMA)
            evs = U.stitch(rs, cs, evw, ctx)
            sg = U.stitch(rs, cs, ph * wv[..., None], ctx)
            pn, pw = sg[..., 0], sg[..., 1]

        np.savez_compressed(
            cache_file,
            evidence=evs.astype(np.float32),
            ndvi_phi=pn.astype(np.float32),
            ndwi_phi=pw.astype(np.float32),
        )

    scene_ev[month] = evs

    scene_prob_path = OUT_DIR / f"step3_scene_prob_{month}.npz"
    scene_cam_path = OUT_DIR / f"step4_scene_cam_{month}.npz"

    if not scene_prob_path.exists() or not scene_cam_path.exists():
        raise FileNotFoundError(
            f"Missing Step 3/4 scene output for {month}; "
            "complete those steps first."
        )

    with np.load(scene_prob_path, allow_pickle=False) as data:
        prb = np.asarray(data["prob"], dtype=np.float32)
    with np.load(scene_cam_path, allow_pickle=False) as data:
        ref = np.asarray(data["cam"], dtype=np.float32)

    water = np.asarray(scene["water"], dtype=bool)
    hr = np.asarray(scene["hr"], dtype=bool)

    for name, arr in (
        ("probability", prb),
        ("Grad-CAM", ref),
        ("water", water),
        ("proxy high-risk", hr),
        ("validity", scene_valid),
    ):
        if arr.shape != (H, W):
            raise ValueError(
                f"{month}: {name} map has shape {arr.shape}; expected {(H, W)}."
            )

    ok = scene_valid & ~water & (region[month] == 3)

    # Only evaluate pixels where every required map is finite.
    stats_mask = ok & np.isfinite(evs) & np.isfinite(prb) & np.isfinite(ref)
    stats = U.scene_agreement(
        evs, ref, prb, hr, stats_mask, THR
    ) or {}

    valid_ndvi = ok & np.isfinite(pn)
    valid_ndwi = ok & np.isfinite(pw)
    stats["mean_ndvi_phi_test"] = (
        safe_mean(pn[valid_ndvi]) if np.any(valid_ndvi) else None
    )
    stats["mean_ndwi_phi_test"] = (
        safe_mean(pw[valid_ndwi]) if np.any(valid_ndwi) else None
    )

    # Add a finite correlation calculated with explicit guards.
    stats["spearman_vs_prob_safe"] = correlation(
        evs[stats_mask], prb[stats_mask]
    )
    scene_stats[month] = stats
    print(f"{month} scene statistics:", stats)

    U.save_geotiff(
        OUT_DIR / f"step5_shap_{month}.tif",
        [evs, pn, pw],
        [
            "SHAP evidence for high risk (0-1)",
            "NDVI contribution (logit)",
            "NDWI contribution (logit)",
        ],
        ctx,
    )

# ---------------------------------------------------------------------
# 6. Save metrics

metrics = {
    "method": "expected gradients on pre-sigmoid logit",
    "n_samples": N_SAMPLES,
    "scene_samples": SCENE_SAMPLES,
    "n_background": n_bg,
    "base_value_logit": safe_float(base),
    "completeness": comp,
    "feature_importance": G,
    "focus_test_predicted_high": S_pos,
    "focus_test_all": S_all,
    "agreement_with_gradcam": AG,
    "sanity_randomised_model": sanity,
    "shap_library_crosscheck": xc,
    "prediction_check_max_abs_difference": prob_diff,
    "scene": scene_stats,
}

with open(OUT_DIR / "step5_metrics.json", "w", encoding="utf-8") as f:
    json.dump(json_safe(metrics), f, indent=2, allow_nan=False)

# ---------------------------------------------------------------------
# 7. Patch-level figures

sel = U.pick_examples(ctx, p_chk)
if len(sel) == 0:
    sel = idx_te[:min(3, len(idx_te))]

if len(sel):
    fig, axes = plt.subplots(
        5, len(sel),
        figsize=(max(5, 2.3 * len(sel)), 12),
        squeeze=False,
    )

    for j, i in enumerate(sel):
        v = valid[i]
        vals = np.abs(phi[i][v])
        lim = max(
            float(np.percentile(vals, 99)) if vals.size else 0.0,
            1e-9,
        )

        axes[0, j].imshow(
            np.ma.masked_where(~v, X[i, ..., 0]),
            cmap="YlGn", vmin=0, vmax=0.7,
        )
        axes[0, j].set_title(
            f"NDVI | p={p_chk[i]:.2f}\n"
            f"true={'HIGH' if y[i] else 'low'}",
            fontsize=8,
        )

        axes[1, j].imshow(
            np.ma.masked_where(~v, X[i, ..., 1]),
            cmap="BrBG", vmin=-1, vmax=1,
        )
        axes[1, j].set_title("NDWI", fontsize=8)

        axes[2, j].imshow(
            np.ma.masked_where(~v, phi[i, ..., 0]),
            cmap="bwr", vmin=-lim, vmax=lim,
        )
        axes[2, j].set_title(
            f"SHAP NDVI sum={fc[i, 0]:+.2f}", fontsize=8
        )

        axes[3, j].imshow(
            np.ma.masked_where(~v, phi[i, ..., 1]),
            cmap="bwr", vmin=-lim, vmax=lim,
        )
        axes[3, j].set_title(
            f"SHAP NDWI sum={fc[i, 1]:+.2f}", fontsize=8
        )

        overlay(
            axes[4, j],
            np.ma.masked_where(~v, X[i, ..., 0]),
            ev[i], "YlGn", 0, 0.7, "SHAP evidence on NDVI",
        )
        for ax in axes[:, j]:
            ax.axis("off")

    fig.suptitle(
        "SHAP: red pushes risk up; blue pushes risk down", y=0.995
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "step5_shap_patches.png", dpi=75)
    plt.close(fig)

# ---------------------------------------------------------------------
# 8. Global importance and dependence figures

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
xs = np.arange(2)
width = 0.25

for offset, (label, key) in enumerate((
    ("all test", "all_test"),
    ("pred. high", "predicted_high"),
    ("pred. low", "predicted_low"),
)):
    values = G[key]
    axes[0].bar(
        xs + (offset - 1) * width,
        [
            values["mean_abs_NDVI"] or 0,
            values["mean_abs_NDWI"] or 0,
        ],
        width,
        label=label,
    )

axes[0].set_xticks(xs, FEATURES)
axes[0].set_ylabel("Mean absolute SHAP contribution (logit)")
axes[0].set_title("Feature importance")
axes[0].legend()

for offset, (label, key) in enumerate((
    ("pred. high", "predicted_high"),
    ("pred. low", "predicted_low"),
)):
    values = G[key]
    axes[1].bar(
        xs + (offset - 0.5) * width * 1.5,
        [
            values["mean_signed_NDVI"] or 0,
            values["mean_signed_NDWI"] or 0,
        ],
        width * 1.4,
        label=label,
    )

axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_xticks(xs, FEATURES)
axes[1].set_title("Mean signed contribution")
axes[1].legend()

valid_test = valid[idx_te]
for channel, label in ((0, "NDVI"), (1, "NDWI")):
    values = X[idx_te][..., channel][valid_test]
    contributions = phi[idx_te][..., channel][valid_test]

    if len(values):
        rng_plot = np.random.RandomState(3)
        chosen = rng_plot.choice(
            len(values), min(25000, len(values)), replace=False
        )
        xv, pv = values[chosen], contributions[chosen]
        axes[2].scatter(xv, pv, s=2, alpha=0.12, c="black",
                        label=f"{label} points")
        break

axes[2].axhline(0, color="gray", lw=0.8)
axes[2].set_xlabel("Input feature value")
axes[2].set_ylabel("SHAP contribution (logit)")
axes[2].set_title("Feature dependence sample")
axes[2].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "step5_shap_global.png", dpi=80)
plt.close(fig)

# ---------------------------------------------------------------------
# 9. Monthly scene figures

for month in MONTHS:
    scene = ctx.scene[month]
    scene_valid = np.asarray(
        scene["valid_risk"] if "valid_risk" in scene.files else scene["valid"],
        dtype=bool,
    )
    water = np.asarray(scene["water"], dtype=bool)
    ok = scene_valid & ~water
    sc = scene_ev[month]

    with np.load(
        OUT_DIR / f"step5_scene_shap_{month}.npz",
        allow_pickle=False,
    ) as data:
        pn = np.asarray(data["ndvi_phi"], dtype=np.float32)
        pw = np.asarray(data["ndwi_phi"], dtype=np.float32)

    finite_phi = np.concatenate([
        np.abs(pn[ok & np.isfinite(pn)]),
        np.abs(pw[ok & np.isfinite(pw)]),
    ])
    lim = max(
        float(np.percentile(finite_phi, 99)) if finite_phi.size else 0.0,
        1e-6,
    )

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    overlay(
        axes[0],
        np.ma.masked_where(
            ~scene_valid, np.asarray(scene["ndvi"], dtype=np.float32)
        ),
        sc, "YlGn", 0, 0.7, f"{month}: SHAP evidence",
    )

    axes[1].imshow(
        np.ma.masked_where(~ok | ~np.isfinite(pn), pn),
        cmap="bwr", vmin=-lim, vmax=lim,
    )
    axes[1].set_title("NDVI contribution (logit)")

    axes[2].imshow(
        np.ma.masked_where(~ok | ~np.isfinite(pw), pw),
        cmap="bwr", vmin=-lim, vmax=lim,
    )
    axes[2].set_title("NDWI contribution (logit)")

    with np.load(
        OUT_DIR / f"step3_scene_prob_{month}.npz",
        allow_pickle=False,
    ) as data:
        prb = np.asarray(data["prob"], dtype=np.float32)

    pred = np.isfinite(prb) & (prb >= THR)
    axes[3].imshow(
        np.ma.masked_invalid(sc), cmap="magma", vmin=0, vmax=1
    )

    if np.any(pred & ok):
        axes[3].contour(
            pred & ok, levels=[0.5], colors="cyan", linewidths=0.6
        )

    proxy = np.asarray(scene["hr"], dtype=bool) & scene_valid
    if np.any(proxy):
        axes[3].contour(
            proxy, levels=[0.5], colors="lime", linewidths=0.4
        )

    axes[3].set_title(
        "Cyan = predicted high risk; green = proxy high risk",
        fontsize=9,
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        OUT_DIR / f"step5_shap_scene_{month}.png", dpi=55
    )
    plt.close(fig)

print("Step 5 SHAP completed.")
