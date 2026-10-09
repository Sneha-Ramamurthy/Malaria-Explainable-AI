"""Step 4 - Grad-CAM heatmaps, focus analysis and sanity checks.

Run:
    python -m src.step4_gradcam

Uses the trained model and Step 2/3 outputs.
Grad-CAM explains model behaviour; it does not establish
actual malaria transmission risk.
"""

import os
import json

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import keras
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import rasterio

from rasterio.transform import Affine
from scipy.stats import wilcoxon, spearmanr

from .config import OUT_DIR, ROOT
from .gradcam import load_model, GradCAM
from .models import build_model


# ------------------------------------------------------------
# 1. Configuration and input checks
# ------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}. "
            "Run the earlier pipeline steps first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_npz(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}. "
            "Run the earlier pipeline steps first."
        )
    return np.load(path, allow_pickle=False)


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def safe_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


CACHE = env_flag("CACHE", False)

M3 = load_json(OUT_DIR / "step3_metrics.json")
M2 = load_json(OUT_DIR / "step2_meta.json")

THR = float(M3["threshold"])
MONTHS = M2["months"]
PATCH = int(M2["patch"])
H, W = map(int, M2["shape"])

if not 0 < THR < 1:
    raise ValueError(f"Invalid classification threshold: {THR}")

model_path = ROOT / "models" / "mobilenetv2_cbam.keras"

if not model_path.exists():
    raise FileNotFoundError(
        f"Trained model not found: {model_path}. "
        "Run Step 3 before Step 4."
    )

model = load_model(model_path)
gc = GradCAM(model, "last_conv")

with load_npz(OUT_DIR / "step2_patches.npz") as d:
    required = {"X", "y", "split", "valid", "mask", "meta"}
    missing = required - set(d.files)

    if missing:
        raise ValueError(
            f"Step 2 patch file is missing arrays: {sorted(missing)}"
        )

    X = d["X"].astype("float32")
    y = d["y"].astype("int32")
    split = d["split"].astype("int32")
    valid = d["valid"].astype(bool)
    hrm = d["mask"].astype(bool)
    meta = d["meta"].astype("int64")

if X.ndim != 4 or X.shape[1:] != (PATCH, PATCH, 2):
    raise ValueError(f"Unexpected patch array shape: {X.shape}")

if not (
    len(X) == len(y) == len(split) == len(valid)
    == len(hrm) == len(meta)
):
    raise ValueError("Step 2 arrays have inconsistent lengths.")

if valid.shape != X.shape[:3] or hrm.shape != X.shape[:3]:
    raise ValueError("Patch masks do not match the patch dimensions.")

if not np.isfinite(X).all():
    raise ValueError("Patch inputs contain NaN or infinite values.")

if not np.isin(split, [0, 1, 2]).all():
    raise ValueError("Unexpected split labels; expected 0, 1 and 2.")

te = split == 2

if not te.any():
    raise ValueError("No test patches were found in the Step 2 dataset.")

scene = {}

for month in MONTHS:
    path = OUT_DIR / f"step2_scene_{month}.npz"
    scene[month] = load_npz(path)

    required_scene = {"ndvi", "ndwi", "water", "hr"}
    missing = required_scene - set(scene[month].files)

    if missing:
        raise ValueError(
            f"{path.name} is missing arrays: {sorted(missing)}"
        )

    if "valid_risk" in scene[month].files:
        scene_valid = scene[month]["valid_risk"].astype(bool)
    elif "valid" in scene[month].files:
        scene_valid = scene[month]["valid"].astype(bool)
    else:
        raise ValueError(
            f"{path.name} needs a valid or valid_risk mask."
        )

    if scene_valid.shape != (H, W):
        raise ValueError(
            f"Unexpected scene mask shape for {month}: "
            f"{scene_valid.shape}"
        )


def get_scene_valid(month):
    """Prefer the land/risk-validity mask introduced in Step 2."""
    s = scene[month]

    if "valid_risk" in s.files:
        return s["valid_risk"].astype(bool)

    return s["valid"].astype(bool)


# ------------------------------------------------------------
# 2. Grad-CAM for patches
# ------------------------------------------------------------

patch_cache = OUT_DIR / "step4_gradcam_patches.npz"
use_patch_cache = CACHE and patch_cache.exists()

if use_patch_cache:
    with np.load(patch_cache, allow_pickle=False) as saved:
        cam = saved["cam"].astype("float32")
        prob = saved["prob"].astype("float32")
        empty = saved["empty"].astype(bool)

    if not (
        len(cam) == len(X)
        and len(prob) == len(X)
        and len(empty) == len(X)
    ):
        raise ValueError(
            "Cached Grad-CAM patch results do not match "
            "the current Step 2 dataset. Rerun without CACHE."
        )

    if not np.isfinite(cam).all() or not np.isfinite(prob).all():
        raise ValueError(
            "Cached Grad-CAM results contain invalid values. "
            "Rerun without CACHE."
        )

else:
    cam, logits, empty = gc(X)
    logits = np.asarray(logits, dtype="float32")

    # Numerically stable sigmoid.
    logits = np.clip(logits, -80, 80)
    prob = (1.0 / (1.0 + np.exp(-logits))).astype("float32")

    np.savez_compressed(
        patch_cache,
        cam=cam.astype("float16"),
        prob=prob,
        empty=empty,
    )

print(
    f"Patch CAMs: {len(cam)} | "
    f"empty CAMs: {int(empty.sum())} "
    f"({empty.mean():.1%})"
)


# ------------------------------------------------------------
# 3. Environmental focus analysis
# ------------------------------------------------------------

water_dil = {
    month: cv2.dilate(
        scene[month]["water"].astype("uint8"),
        np.ones((7, 7), dtype="uint8"),
    ).astype(bool)
    for month in MONTHS
}


def patch_slice(array, index):
    month_index, row, col = map(int, meta[index, :3])
    return array[row:row + PATCH, col:col + PATCH]


near_water = np.stack([
    patch_slice(
        water_dil[MONTHS[int(meta[i, 0])]],
        i,
    )
    for i in range(len(X))
])

train_valid = valid[split == 0]
train_ndwi = X[split == 0][..., 1][train_valid]

if train_ndwi.size:
    ndwi_p75 = float(np.percentile(train_ndwi, 75))
else:
    ndwi_p75 = 0.0
    print("WARNING: no valid training pixels for NDWI percentile.")

moist = (X[..., 1] >= ndwi_p75) & valid
vegd = (X[..., 0] >= 0.35) & valid


def focus_stats(camset, indices):
    """Measure CAM enrichment within selected environmental masks."""
    rows = {
        key: []
        for key in (
            "high_risk",
            "near_water",
            "moist",
            "vegetated",
            "invalid_mass",
        )
    }

    for i in indices:
        c = np.asarray(camset[i], dtype="float32")
        v = valid[i]

        if v.sum() < 100 or not np.isfinite(c).all():
            continue

        valid_cam = c[v]

        if valid_cam.size == 0 or valid_cam.max() <= 0:
            continue

        cutoff = np.quantile(valid_cam, 0.80)
        top = (c >= cutoff) & (c > 0) & v

        if top.sum() < 20:
            continue

        masks = (
            ("high_risk", hrm[i]),
            ("near_water", near_water[i]),
            ("moist", moist[i]),
            ("vegetated", vegd[i]),
        )

        for key, mask in masks:
            baseline = float(mask[v].mean())
            if baseline > 0.02:
                rows[key].append(
                    float(mask[top].mean() / baseline)
                )

        invalid_fraction = float((~v).mean())

        if invalid_fraction > 0.02 and c.sum() > 0:
            rows["invalid_mass"].append(
                float(
                    (c[~v].sum() / c.sum())
                    / invalid_fraction
                )
            )

    return {
        key: np.asarray(values, dtype="float64")
        for key, values in rows.items()
    }


def summarise(stats):
    summary = {}

    for key, values in stats.items():
        values = values[np.isfinite(values)]

        if len(values) < 5:
            summary[key] = None
            continue

        try:
            p_value = float(wilcoxon(values - 1).pvalue)
        except (ValueError, FloatingPointError):
            p_value = float("nan")

        summary[key] = {
            "n": int(len(values)),
            "median": safe_float(np.median(values)),
            "mean": safe_float(values.mean()),
            "share_above_1": safe_float((values > 1).mean()),
            "wilcoxon_p_vs_1": safe_float(p_value),
        }

    return summary


idx_all = np.flatnonzero(te)
idx_pos = np.flatnonzero(te & (prob >= THR))

st_all = focus_stats(cam, idx_all)
st_pos = focus_stats(cam, idx_pos)

S_all = summarise(st_all)
S_pos = summarise(st_pos)


# ------------------------------------------------------------
# 4. Randomised-model sanity check
# ------------------------------------------------------------

keras.utils.set_random_seed(123)

rnd = build_model(
    M3["norm_mean"],
    M3["norm_var"],
    **M3["model_config"],
)

if rnd.output_shape[-1] != 1:
    raise ValueError("Randomised model has an unexpected output shape.")

cam_r, _, _ = gc(X[te]) if False else (None, None, None)

# Use the randomised model, not the trained GradCAM instance.
rnd_gc = GradCAM(rnd, "last_conv")
cam_r, _, _ = rnd_gc(X[te], batch=128)

st_rnd = focus_stats(
    dict(zip(idx_all, cam_r)),
    idx_all,
)

S_rnd = summarise(st_rnd)

correlations = []

for trained_cam, random_cam, index in zip(cam[te], cam_r, idx_all):
    v = valid[index]

    if (
        v.sum() > 100
        and trained_cam[v].std() > 0
        and random_cam[v].std() > 0
    ):
        result = spearmanr(trained_cam[v], random_cam[v]).statistic

        if np.isfinite(result):
            correlations.append(float(result))

rho = float(np.mean(correlations)) if correlations else None

print("\nFOCUS: test patches predicted high-risk")
print("Enrichment > 1 means the CAM overlaps a class more than its baseline share.")

for key, value in S_pos.items():
    print(
        f"  {key:13s}",
        None if value is None else value,
    )

print("SANITY: trained vs randomised CAM mean Spearman rho =", rho)


# ------------------------------------------------------------
# 5. Scene-level Grad-CAM
# ------------------------------------------------------------

region = {
    month: np.zeros((H, W), dtype="uint8")
    for month in MONTHS
}

for i in range(len(X)):
    month_index, row, col = map(int, meta[i, :3])

    if month_index < 0 or month_index >= len(MONTHS):
        continue

    month = MONTHS[month_index]
    row_end = min(row + PATCH, H)
    col_end = min(col + PATCH, W)

    region[month][row:row_end, col:col_end] = np.maximum(
        region[month][row:row_end, col:col_end],
        split[i] + 1,
    )

affine = Affine(*M2["transform"])
scene_cam = {}
scene_stats = {}


for month in MONTHS:
    s = scene[month]
    sv = get_scene_valid(month)

    cache_path = OUT_DIR / f"step4_scene_cam_{month}.npz"
    tif_path = OUT_DIR / f"step4_gradcam_{month}.tif"

    use_cache = CACHE and cache_path.exists()

    if use_cache:
        with np.load(cache_path, allow_pickle=False) as saved:
            sc = saved["cam"].astype("float32")

        if sc.shape != (H, W):
            raise ValueError(
                f"Cached scene CAM has the wrong shape for {month}."
            )

    else:
        img = np.stack(
            [
                s["ndvi"].astype("float32"),
                s["ndwi"].astype("float32"),
            ],
            axis=-1,
        )

        img[~sv] = 0.0

        rows, cols, windows = [], [], []

        # Include the final edge-aligned window where necessary.
        row_starts = list(range(0, max(H - PATCH + 1, 1), 32))
        col_starts = list(range(0, max(W - PATCH + 1, 1), 32))

        if H >= PATCH and (not row_starts or row_starts[-1] != H - PATCH):
            row_starts.append(H - PATCH)

        if W >= PATCH and (not col_starts or col_starts[-1] != W - PATCH):
            col_starts.append(W - PATCH)

        for row in row_starts:
            for col in col_starts:
                window_valid = sv[
                    row:row + PATCH,
                    col:col + PATCH,
                ]

                if window_valid.shape != (PATCH, PATCH):
                    continue

                if window_valid.mean() < 0.60:
                    continue

                rows.append(row)
                cols.append(col)
                windows.append(
                    img[row:row + PATCH, col:col + PATCH]
                )

        acc = np.zeros((H, W), dtype="float32")
        cnt = np.zeros((H, W), dtype="float32")

        if windows:
            windows = np.stack(windows).astype("float32")
            window_cams, _, _ = gc(windows, batch=128)

            for row, col, window_cam in zip(rows, cols, window_cams):
                acc[
                    row:row + PATCH,
                    col:col + PATCH,
                ] += window_cam

                cnt[
                    row:row + PATCH,
                    col:col + PATCH,
                ] += 1

            sc = np.full((H, W), np.nan, dtype="float32")
            covered = cnt > 0
            sc[covered] = acc[covered] / cnt[covered]

        else:
            print(
                f"WARNING: no valid sliding windows for {month}; "
                "scene CAM will contain only NaNs."
            )
            sc = np.full((H, W), np.nan, dtype="float32")

        np.savez_compressed(
            cache_path,
            cam=sc.astype("float16"),
        )

        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=H,
            width=W,
            count=1,
            dtype="float32",
            crs=M2["crs"],
            transform=affine,
            nodata=-9999,
        ) as dst:
            dst.write(
                np.nan_to_num(
                    sc,
                    nan=-9999.0,
                    posinf=-9999.0,
                    neginf=-9999.0,
                ).astype("float32"),
                1,
            )

    scene_cam[month] = sc

    probability_path = OUT_DIR / f"step3_scene_prob_{month}.npz"

    with load_npz(probability_path) as saved:
        prb = saved["prob"].astype("float32")

    if prb.shape != (H, W):
        raise ValueError(
            f"Step 3 probability map has the wrong shape for {month}."
        )

    ok = (
        sv
        & ~s["water"].astype(bool)
        & np.isfinite(sc)
        & (region[month] == 3)
    )

    if ok.sum() > 100:
        top_cutoff = np.quantile(sc[ok], 0.80)
        top = ok & (sc >= top_cutoff)
    else:
        top = np.zeros((H, W), dtype=bool)

    pred = np.isfinite(prb) & (prb >= THR)

    cam_prob_corr = None

    if ok.sum() > 100 and np.std(sc[ok]) > 0 and np.std(prb[ok]) > 0:
        cam_prob_corr = safe_float(
            spearmanr(sc[ok], prb[ok]).statistic
        )

    scene_stats[month] = {
        "test_px": int(ok.sum()),
        "top20cam_inside_pred_highrisk": (
            safe_float(pred[top].mean()) if top.any() else None
        ),
        "pred_highrisk_share_of_test_land": (
            safe_float(pred[ok].mean()) if ok.any() else None
        ),
        "top20cam_inside_proxy_highrisk": (
            safe_float(s["hr"][top].mean()) if top.any() else None
        ),
        "proxy_highrisk_share_of_test_land": (
            safe_float(s["hr"][ok].mean()) if ok.any() else None
        ),
        "spearman_cam_vs_prob": cam_prob_corr,
    }

    print(month, scene_stats[month])


# ------------------------------------------------------------
# 6. Save metrics
# ------------------------------------------------------------

metrics = {
    "layer": "last_conv",
    "threshold": THR,
    "empty_cam_fraction": safe_float(empty.mean()),
    "focus_test_predicted_high": S_pos,
    "focus_test_all": S_all,
    "focus_randomised_model": S_rnd,
    "sanity_spearman_trained_vs_random": rho,
    "scene": scene_stats,
}

with open(
    OUT_DIR / "step4_metrics.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(metrics, f, indent=2, allow_nan=False)


# ------------------------------------------------------------
# 7. Patch figures
# ------------------------------------------------------------

positive = idx_all[
    (y[idx_all] == 1)
    & (prob[idx_all] >= THR)
]

positive = positive[np.argsort(-prob[positive])][:5]

negative = idx_all[y[idx_all] == 0]
negative = negative[np.argsort(prob[negative])][:2]

false_negative = idx_all[
    (y[idx_all] == 1)
    & (prob[idx_all] < THR)
][:1]

selected = np.concatenate(
    [positive, false_negative, negative]
)

if len(selected) == 0:
    selected = idx_all[np.argsort(-prob[idx_all])[:1]]

fig, axes = plt.subplots(
    5,
    len(selected),
    figsize=(2.5 * len(selected), 12),
    squeeze=False,
)

def heat(ax, base, cam_map, cmap, vmin, vmax, title):
    ax.imshow(base, cmap=cmap, vmin=vmin, vmax=vmax)

    overlay = np.ma.masked_where(
        ~np.isfinite(cam_map) | (cam_map < 0.15),
        cam_map,
    )

    ax.imshow(
        overlay,
        cmap="jet",
        alpha=0.55,
        vmin=0,
        vmax=1,
    )

    ax.set_title(title, fontsize=8)
    ax.axis("off")


for col, i in enumerate(selected):
    v = valid[i]

    ndvi = np.ma.masked_where(~v, X[i, ..., 0])
    ndwi = np.ma.masked_where(~v, X[i, ..., 1])

    title = (
        f"p={prob[i]:.2f} | "
        f"true={'HIGH' if y[i] else 'low'}"
    )

    axes[0, col].imshow(ndvi, cmap="YlGn", vmin=0, vmax=0.7)
    axes[0, col].set_title("NDVI\n" + title, fontsize=8)

    axes[1, col].imshow(ndwi, cmap="BrBG", vmin=-1, vmax=1)
    axes[1, col].set_title("NDWI", fontsize=8)

    heat(
        axes[2, col],
        ndvi,
        cam[i],
        "YlGn",
        0,
        0.7,
        "Grad-CAM over NDVI",
    )

    heat(
        axes[3, col],
        ndwi,
        cam[i],
        "BrBG",
        -1,
        1,
        "Grad-CAM over NDWI",
    )

    axes[4, col].imshow(hrm[i], cmap="Reds", vmin=0, vmax=1)

    if cam[i].max() > 0:
        axes[4, col].contour(
            cam[i],
            levels=[0.5],
            colors="cyan",
            linewidths=1.2,
        )

    axes[4, col].set_title(
        "Proxy high-risk mask\n+ CAM 0.5 contour",
        fontsize=8,
    )

    for row in range(5):
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig(
    OUT_DIR / "step4_gradcam_patches.png",
    dpi=75,
)
plt.close()


# ------------------------------------------------------------
# 8. Scene figures
# ------------------------------------------------------------

for month in MONTHS:
    s = scene[month]
    sv = get_scene_valid(month)
    sc = scene_cam[month]

    with load_npz(OUT_DIR / f"step3_scene_prob_{month}.npz") as saved:
        prb = saved["prob"].astype("float32")

    pred = np.isfinite(prb) & (prb >= THR)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    heat(
        axes[0],
        np.ma.masked_where(~sv, s["ndvi"].astype("float32")),
        np.nan_to_num(sc, nan=0.0),
        "YlGn",
        0,
        0.7,
        f"{month}: Grad-CAM over NDVI",
    )

    heat(
        axes[1],
        np.ma.masked_where(~sv, s["ndwi"].astype("float32")),
        np.nan_to_num(sc, nan=0.0),
        "BrBG",
        -1,
        1,
        f"{month}: Grad-CAM over NDWI",
    )

    axes[2].imshow(
        np.ma.masked_invalid(sc),
        cmap="magma",
        vmin=0,
        vmax=1,
    )

    pred_contour = pred & sv & ~s["water"].astype(bool)

    if pred_contour.any() and pred_contour.all() is False:
        axes[2].contour(
            pred_contour.astype("float32"),
            levels=[0.5],
            colors="cyan",
            linewidths=0.6,
        )

    proxy = s["hr"].astype(bool) & sv

    if proxy.any() and not proxy.all():
        axes[2].contour(
            proxy.astype("float32"),
            levels=[0.5],
            colors="lime",
            linewidths=0.4,
        )

    axes[2].set_title(
        "Scene CAM: cyan = predicted high-risk; "
        "green = proxy high-risk",
        fontsize=9,
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        OUT_DIR / f"step4_gradcam_scene_{month}.png",
        dpi=55,
    )
    plt.close()


# ------------------------------------------------------------
# 9. Focus comparison figures
# ------------------------------------------------------------

names = ["high_risk", "near_water", "moist", "vegetated"]

labels = [
    "proxy\nhigh-risk",
    "near\nwater",
    "moist\n(NDWI top 25%)",
    "vegetated\n(NDVI >= 0.35)",
]


def medians(summary):
    return [
        (
            summary[key]["median"]
            if summary.get(key) is not None
            else np.nan
        )
        for key in names
    ]


fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

x = np.arange(len(names))
width = 0.38

axes[0].bar(
    x - width / 2,
    medians(S_all),
    width,
    label="trained model",
)

axes[0].bar(
    x + width / 2,
    medians(S_rnd),
    width,
    label="randomised model",
)

axes[0].axhline(1, color="black", linestyle="--")
axes[0].set_xticks(x, labels)
axes[0].set_ylabel("Median enrichment (1 = baseline)")
axes[0].set_title("Where does the top 20% of Grad-CAM fall?")
axes[0].legend()

trained_values = st_all["high_risk"]
random_values = st_rnd["high_risk"]

axes[1].hist(
    trained_values[np.isfinite(trained_values)],
    bins=25,
    alpha=0.7,
    label="trained",
)

axes[1].hist(
    random_values[np.isfinite(random_values)],
    bins=25,
    alpha=0.5,
    label="randomised",
)

axes[1].axvline(1, color="black", linestyle="--")
axes[1].set_title("High-risk enrichment per test patch")
axes[1].legend()

plt.tight_layout()
plt.savefig(
    OUT_DIR / "step4_focus_analysis.png",
    dpi=80,
)
plt.close()


# ------------------------------------------------------------
# 10. Completion
# ------------------------------------------------------------

print("\nStep 4 finished.")
print("Metrics:", OUT_DIR / "step4_metrics.json")
print("Patch figure:", OUT_DIR / "step4_gradcam_patches.png")
print("Focus figure:", OUT_DIR / "step4_focus_analysis.png")
print("Scene figures and GeoTIFFs saved for each month.")
print("Reminder: proxy-risk explanations are not clinical or")
print("epidemiological validation of malaria transmission.")
