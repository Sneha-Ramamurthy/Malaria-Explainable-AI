"""STEP 2 - Decode rasters, separate sea/inland water,
build proxy risk labels, and tile patches.

Run: python -m src.step2_build_dataset
"""

import json

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from .config import OUT_DIR
from .data_loader import index_files, load_rgb
from .decode import decode_ndvi, decode_ndwi, decode_scl


# ---------------- Configuration ----------------

DS = 2
PATCH, STRIDE, BLOCK = 64, 16, 128
POS_QUANTILE = 0.70
MONTHS = ["2025-04", "2025-05"]

KM_PX = 0.765
D0_KM = 2.0
RISK_WEIGHTS = dict(prox=0.4, moist=0.3, veg=0.3)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- 1. Find and validate inputs ----------------

files = index_files()

if not files:
    raise ValueError(
        "No input data files found. Check the dataset folder."
    )

audit_months = sorted({month for _, month in files})

for month in MONTHS:
    missing = [
        layer
        for layer in ("SCL", "NDVI", "NDWI")
        if (layer, month) not in files
    ]

    if missing:
        raise ValueError(
            f"Month {month} is missing required input layers: {missing}"
        )


def scl_ds(month):
    """Load and downsample the scene classification layer."""
    rgb, profile = load_rgb(files[("SCL", month)])
    scl, _ = decode_scl(rgb)
    return scl[::DS, ::DS], profile


# ---------------- 2. Build water composite ----------------

usable = []

for month in audit_months:
    if ("SCL", month) not in files:
        continue

    scl, _ = scl_ds(month)

    if scl.max() > 0:
        usable.append(month)

if not usable:
    raise ValueError(
        "No usable SCL data found. Check source images and decoding."
    )

water_cnt = None
valid_cnt = None

for month in usable:
    scl, _ = scl_ds(month)

    month_water = (scl == 2).astype(np.uint16)
    month_valid = np.isin(scl, (1, 2, 3)).astype(np.uint16)

    if water_cnt is None:
        water_cnt = np.zeros_like(month_water, dtype=np.uint16)
        valid_cnt = np.zeros_like(month_valid, dtype=np.uint16)

    if scl.shape != water_cnt.shape:
        raise ValueError(
            f"SCL dimensions differ for month {month}: {scl.shape}"
        )

    water_cnt += month_water
    valid_cnt += month_valid

water_freq = water_cnt / np.maximum(valid_cnt, 1)

# Identify the largest connected component of persistent water.
lab, n_components = ndi.label(water_freq > 0.5)

if n_components > 0:
    sizes = ndi.sum(
        np.ones_like(lab),
        lab,
        range(1, n_components + 1),
    )
    largest_label = 1 + int(np.argmax(sizes))
    sea = lab == largest_label
else:
    sea = np.zeros_like(water_freq, dtype=bool)

sea = ndi.binary_dilation(sea, iterations=2)

perm_inland = (water_freq > 0.3) & ~sea

print(
    f"Usable months: {len(usable)} | "
    f"Sea pixels: {int(sea.sum())} | "
    f"Inland permanent-water pixels: {int(perm_inland.sum())}"
)


# ---------------- 3. Decode monthly scene layers ----------------

H, Wd = sea.shape
scene = {}

for month in MONTHS:
    ndvi_rgb, _ = load_rgb(files[("NDVI", month)])
    ndwi_rgb, _ = load_rgb(files[("NDWI", month)])

    ndvi, ndvi_error = decode_ndvi(ndvi_rgb)
    ndwi = decode_ndwi(ndwi_rgb)

    if ndvi.ndim != 2 or ndwi.ndim != 2:
        raise ValueError(
            f"Decoded NDVI/NDWI must be 2D arrays for {month}."
        )

    ndvi = cv2.resize(
        ndvi.astype(np.float32),
        (Wd, H),
        interpolation=cv2.INTER_AREA,
    )
    ndwi = cv2.resize(
        ndwi.astype(np.float32),
        (Wd, H),
        interpolation=cv2.INTER_AREA,
    )

    scl, _ = scl_ds(month)

    if scl.shape != (H, Wd):
        raise ValueError(
            f"SCL shape mismatch for {month}: {scl.shape}"
        )

    if not np.isfinite(ndvi).all() or not np.isfinite(ndwi).all():
        raise ValueError(
            f"Decoded NDVI/NDWI contains NaN or infinite values in {month}."
        )

    print(
        month,
        "maximum NDVI colour-match error:",
        float(np.max(ndvi_error)),
    )

    valid_in = np.isin(scl, (1, 2, 3)) & ~sea

    water = ((scl == 2) | perm_inland) & ~sea

    # Pixels used to calculate the proxy land-risk score.
    valid_risk = valid_in & ~water

    # Distance from each pixel to water, in approximate kilometres.
    dist = (
        cv2.distanceTransform(
            (~water).astype(np.uint8) * 255,
            cv2.DIST_L2,
            5,
        )
        * KM_PX
    )

    scene[month] = {
        "ndvi": ndvi,
        "ndwi": ndwi,
        "scl": scl,
        "valid_in": valid_in,
        "water": water,
        "valid_risk": valid_risk,
        "dist": dist,
    }


# ---------------- 4. Calculate proxy risk scores ----------------

ndwi_pools = [
    scene[month]["ndwi"][scene[month]["valid_risk"]]
    for month in MONTHS
]

ndwi_pools = [
    values for values in ndwi_pools if values.size > 0
]

if not ndwi_pools:
    raise ValueError(
        "No valid land pixels available for risk calculation."
    )

pool = np.concatenate(ndwi_pools)
pool = pool[np.isfinite(pool)]

if pool.size == 0:
    raise ValueError("No finite NDWI values available for risk calculation.")

p5, p95 = np.percentile(pool, [5, 95])

if p95 <= p5:
    print(
        "Warning: NDWI values have little or no spread; "
        "the moisture score may be uninformative."
    )

for month in MONTHS:
    data = scene[month]

    prox = np.exp(-data["dist"] / D0_KM)

    moist = np.clip(
        (data["ndwi"] - p5) / (p95 - p5 + 1e-6),
        0,
        1,
    )

    veg = np.exp(
        -(((data["ndvi"] - 0.5) / 0.2) ** 2)
    )

    score = (
        RISK_WEIGHTS["prox"] * prox
        + RISK_WEIGHTS["moist"] * moist
        + RISK_WEIGHTS["veg"] * veg
    )

    score = cv2.GaussianBlur(
        score.astype(np.float32),
        (0, 0),
        3.0,
    )

    data["risk"] = np.where(
        data["valid_risk"],
        score,
        np.nan,
    ).astype(np.float32)

allscore = np.concatenate([
    scene[month]["risk"][scene[month]["valid_risk"]]
    for month in MONTHS
])

allscore = allscore[np.isfinite(allscore)]

if allscore.size == 0:
    raise ValueError(
        "No finite risk scores available. "
        "Check NDVI, NDWI and valid-land masks."
    )

THR = float(np.percentile(allscore, 80))

for month in MONTHS:
    data = scene[month]

    data["hr"] = (
        data["valid_risk"] & (data["risk"] >= THR)
    )

    valid_count = int(data["valid_risk"].sum())
    high_risk_count = int(data["hr"].sum())

    high_risk_share = (
        high_risk_count / valid_count if valid_count else 0.0
    )

    print(
        f"{month} | Valid land pixels: {valid_count} | "
        f"High-risk share: {high_risk_share:.3f}"
    )


# ---------------- 5. Extract patches ----------------

X, M, R, V, meta = [], [], [], [], []

for month in MONTHS:
    data = scene[month]

    # Invalid land-risk pixels are zeroed in the model inputs.
    img = (
        np.stack([data["ndvi"], data["ndwi"]], axis=-1)
        * data["valid_risk"][..., None]
    )

    for br in range(H // BLOCK):
        for bc in range(Wd // BLOCK):

            for dr in range(0, BLOCK - PATCH + 1, STRIDE):
                for dc in range(0, BLOCK - PATCH + 1, STRIDE):

                    r = br * BLOCK + dr
                    c = bc * BLOCK + dc

                    rows = slice(r, r + PATCH)
                    cols = slice(c, c + PATCH)

                    valid_patch = data["valid_risk"][rows, cols]

                    if valid_patch.mean() < 0.6:
                        continue

                    X.append(img[rows, cols])
                    M.append(data["hr"][rows, cols])
                    R.append(
                        np.nan_to_num(
                            data["risk"][rows, cols],
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                    )
                    V.append(valid_patch)

                    block_id = br * (Wd // BLOCK) + bc

                    # Columns: month index, row, column, spatial block ID.
                    meta.append(
                        (MONTHS.index(month), r, c, block_id)
                    )

if not X:
    raise ValueError(
        "No patches were generated. Check image dimensions, "
        "valid-data coverage, PATCH, BLOCK and STRIDE."
    )

X = np.asarray(X, dtype=np.float16)
M = np.asarray(M, dtype=np.uint8)
R = np.asarray(R, dtype=np.float16)
V = np.asarray(V, dtype=np.uint8)
meta = np.asarray(meta, dtype=np.int32)

# Each patch label is based only on valid land pixels.
valid_counts = V.sum(axis=(1, 2))
high_risk_counts = (M * V).sum(axis=(1, 2))

frac = np.divide(
    high_risk_counts,
    valid_counts,
    out=np.zeros(len(valid_counts), dtype=np.float32),
    where=valid_counts > 0,
)

FRAC_THR = float(np.quantile(frac, POS_QUANTILE))
y = (frac > FRAC_THR).astype(np.uint8)

# Keep all patches from a spatial block in the same split.
rng = np.random.RandomState(42)
blocks = np.unique(meta[:, 3])
rng.shuffle(blocks)

n_blocks = len(blocks)
train_end = int(0.6 * n_blocks)
val_end = int(0.8 * n_blocks)

train_blocks = blocks[:train_end]
val_blocks = blocks[train_end:val_end]

split = np.where(
    np.isin(meta[:, 3], train_blocks),
    0,
    np.where(np.isin(meta[:, 3], val_blocks), 1, 2),
).astype(np.uint8)

print(
    f"Patch high-risk-fraction cutoff: {FRAC_THR:.3f} | "
    f"Patches: {len(X)} | "
    f"Positive labels: {y.mean():.3f} | "
    f"Train/validation/test: "
    f"{(split == 0).sum()}/"
    f"{(split == 1).sum()}/"
    f"{(split == 2).sum()}"
)

for split_id, split_name in enumerate(("train", "validation", "test")):
    selected = split == split_id
    positive_count = int(y[selected].sum())
    total_count = int(selected.sum())

    print(
        f"  {split_name}: positives {positive_count} / {total_count}"
    )

# ---------------- 6. Save patch dataset ----------------

np.savez_compressed(
    OUT_DIR / "step2_patches.npz",
    X=X,
    mask=M,
    risk=R,
    valid=V,
    y=y,
    split=split,
    meta=meta,
    months=np.array(MONTHS),
)


# ---------------- 7. Save scene-level data ----------------

for month in MONTHS:
    data = scene[month]

    np.savez_compressed(
        OUT_DIR / f"step2_scene_{month}.npz",
        ndvi=data["ndvi"].astype(np.float16),
        ndwi=data["ndwi"].astype(np.float16),
        valid=data["valid_in"],
        valid=data["valid_risk"],
        water=data["water"],
        risk=data["risk"].astype(np.float16),
        hr=data["hr"],
        sea=sea,
    )

profile0 = scl_ds(usable[0])[1]
transform = profile0["transform"]

metadata = {
    "months": MONTHS,
    "ds": DS,
    "shape": [H, Wd],
    "crs": str(profile0["crs"]),
    "transform": [
        transform.a * DS,
        transform.b,
        transform.c,
        transform.d,
        transform.e * DS,
        transform.f,
    ],
    "patch": PATCH,
    "stride": STRIDE,
    "risk_threshold": THR,
    "patch_frac_threshold": FRAC_THR,
    "ndwi_p5_p95": [float(p5), float(p95)],
    "weights": RISK_WEIGHTS,
    "d0_km": D0_KM,
}

with open(OUT_DIR / "step2_meta.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)


# ---------------- 8. Save scene visualisation ----------------

fig, axes = plt.subplots(2, 5, figsize=(26, 10))

for row, month in enumerate(MONTHS):
    data = scene[month]

    category = np.zeros((H, Wd, 3), dtype=np.float32)
    category[~data["valid_in"]] = (0.1, 0.1, 0.1)
    category[data["valid_in"]] = (0.85, 0.8, 0.6)
    category[sea] = (0.05, 0.1, 0.5)
    category[data["water"]] = (0.2, 0.8, 1.0)

    invalid = ~data["valid_in"]

    panels = [
        (
            np.ma.masked_where(invalid, data["ndvi"]),
            "YlGn",
            "NDVI (decoded)",
            (0, 0.7),
        ),
        (
            np.ma.masked_where(invalid, data["ndwi"]),
            "BrBG",
            "NDWI (decoded)",
            (-1, 1),
        ),
        (
            category,
            None,
            "Sea / inland water / land / no-data",
            None,
        ),
        (
            np.ma.masked_invalid(data["risk"]),
            "inferno",
            "Proxy risk score",
            None,
        ),
        (
            None,
            None,
            "High-risk mask (top 20%)",
            None,
        ),
    ]

    for ax, (image, cmap, title, limits) in zip(axes[row], panels):
        if image is None:
            display = np.zeros((H, Wd, 3), dtype=np.float32)
            display[~data["valid_in"]] = 0.1
            display[data["valid_risk"]] = (0.75, 0.75, 0.75)
            display[sea] = (0.05, 0.1, 0.5)
            display[data["water"]] = (0.2, 0.8, 1.0)
            display[data["hr"]] = (0.9, 0.1, 0.1)
            ax.imshow(display)
        elif cmap is None:
            ax.imshow(image)
        else:
            kwargs = {}
            if limits is not None:
                kwargs["vmin"], kwargs["vmax"] = limits

            ax.imshow(image, cmap=cmap, **kwargs)

        ax.set_title(f"{month}  {title}", fontsize=10)
        ax.axis("off")

plt.tight_layout()
plt.savefig(
    OUT_DIR / "step2_decoded_and_risk.png",
    dpi=55,
)
plt.close(fig)


# ---------------- 9. Save patch examples ----------------

n_examples = min(6, len(X))

indices = np.random.RandomState(1).choice(
    len(X),
    n_examples,
    replace=False,
)

fig, axes = plt.subplots(
    3,
    n_examples,
    figsize=(max(3, 2.5 * n_examples), 7.5),
    squeeze=False,
)

for column, index in enumerate(indices):
    axes[0, column].imshow(
        X[index][..., 0].astype(float),
        cmap="YlGn",
        vmin=0,
        vmax=0.7,
    )
    axes[0, column].set_title(
        f"NDVI  y={y[index]}",
        fontsize=9,
    )

    axes[1, column].imshow(
        X[index][..., 1].astype(float),
        cmap="BrBG",
        vmin=-1,
        vmax=1,
    )
    axes[1, column].set_title("NDWI", fontsize=9)

    axes[2, column].imshow(M[index], cmap="Reds")
    axes[2, column].set_title("High-risk mask", fontsize=9)

    for row in range(3):
        axes[row, column].axis("off")

plt.tight_layout()
plt.savefig(
    OUT_DIR / "step2_patch_examples.png",
    dpi=60,
)
plt.close(fig)

print("Step 2 completed successfully.")
