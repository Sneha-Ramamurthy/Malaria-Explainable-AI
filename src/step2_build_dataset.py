"""STEP 2 - decode rasters, separate sea/inland water, build proxy risk labels, tile patches.
Run:  python -m src.step2_build_dataset
"""
import json
import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from .config import OUT_DIR
from .data_loader import index_files, load_rgb
from .decode import decode_ndvi, decode_ndwi, decode_scl

DS = 2                       # down-sample factor (1786x2500 -> 893x1250)
PATCH, STRIDE, BLOCK = 64, 16, 128
POS_QUANTILE = 0.70          # patch is 'high-risk' if its high-risk-pixel fraction is in the top 30%
MONTHS = ["2025-04", "2025-05"]   # only months that have NDVI + NDWI and usable coverage
KM_PX = 0.765                # approx. km per down-sampled pixel at ~19 N
D0_KM = 2.0                  # distance-decay scale for water proximity
W = dict(prox=0.4, moist=0.3, veg=0.3)

OUT_DIR.mkdir(parents=True, exist_ok=True)

files = index_files()
if not files:
    raise ValueError("No input data files found. Check the dataset folder.")

audit_months = sorted({m for _, m in files})

# Check the months required by this pipeline.
for m in MONTHS:
    missing = [
        layer for layer in ("SCL", "NDVI", "NDWI")
        if (layer, m) not in files
    ]
    if missing:
        raise ValueError(
            f"Month {m} is missing required input layers: {missing}"
        )


def scl_ds(month):
    rgb, prof = load_rgb(files[("SCL", month)])
    return decode_scl(rgb)[0][::DS, ::DS], prof


# ---------- 1. sea mask from a multi-month water composite ----------
usable = [m for m in audit_months if scl_ds(m)[0].max() > 0]
water_cnt = valid_cnt = 0
for m in usable:
    s, prof = scl_ds(m)
    water_cnt = water_cnt + (s == 2)
    valid_cnt = valid_cnt + np.isin(s, (1, 2, 3))
water_freq = water_cnt / np.maximum(valid_cnt, 1)
lab, n = ndi.label(water_freq > 0.5)
sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
sea = lab == (1 + int(np.argmax(sizes)))
sea = ndi.binary_dilation(sea, iterations=2)             # include coastal fringe
perm_inland = (water_freq > 0.3) & ~sea
print(f"usable months for composite: {len(usable)} | sea px: {sea.sum()} | inland permanent-water px: {perm_inland.sum()}")

# ---------- 2. per-month layers ----------
H, Wd = sea.shape
scene = {}
for m in MONTHS:
    ndvi, e1 = decode_ndvi(load_rgb(files[("NDVI", m)])[0])
    ndwi = decode_ndwi(load_rgb(files[("NDWI", m)])[0])
    ndvi = cv2.resize(ndvi, (Wd, H), interpolation=cv2.INTER_AREA)
    ndwi = cv2.resize(ndwi, (Wd, H), interpolation=cv2.INTER_AREA)
    scl, _ = scl_ds(m)
    print(m, "max NDVI colour-match error:", float(e1.max()))
    valid_in = np.isin(scl, (1, 2, 3)) & ~sea
    water = ((scl == 2) | perm_inland) & ~sea
    valid_risk = valid_in & ~water
    dist = cv2.distanceTransform((~water).astype(np.uint8) * 255, cv2.DIST_L2, 5) * KM_PX
    scene[m] = dict(ndvi=ndvi, ndwi=ndwi, scl=scl, valid_in=valid_in, water=water,
                    valid_risk=valid_risk, dist=dist)

# ---------- 3. proxy risk score (knowledge-based pseudo-label) ----------
pool = np.concatenate([scene[m]["ndwi"][scene[m]["valid_risk"]] for m in MONTHS])
p5, p95 = np.percentile(pool, [5, 95])
for m in MONTHS:
    s = scene[m]
    prox = np.exp(-s["dist"] / D0_KM)
    moist = np.clip((s["ndwi"] - p5) / (p95 - p5 + 1e-6), 0, 1)
    veg = np.exp(-(((s["ndvi"] - 0.5) / 0.2) ** 2))
    score = W["prox"] * prox + W["moist"] * moist + W["veg"] * veg
    score = cv2.GaussianBlur(score.astype(np.float32), (0, 0), 3.0)
    s["risk"] = np.where(s["valid_risk"], score, np.nan).astype(np.float32)
allscore = np.concatenate([scene[m]["risk"][scene[m]["valid_risk"]] for m in MONTHS])
THR = float(np.percentile(allscore, 80))
for m in MONTHS:
    s = scene[m]
    s["hr"] = (np.nan_to_num(s["risk"], nan=-1) >= THR)
    print(m, f"valid-land px {s['valid_risk'].sum()} | high-risk share {s['hr'].sum()/s['valid_risk'].sum():.3f}")

# ---------- 4. patches (block-wise so no patch crosses a split boundary) ----------
X, M, R, V, meta = [], [], [], [], []
for m in MONTHS:
    s = scene[m]
    img = np.stack([s["ndvi"], s["ndwi"]], -1) * s["valid_in"][..., None]
    for br in range(H // BLOCK):
        for bc in range(Wd // BLOCK):
            for dr in range(0, BLOCK - PATCH + 1, STRIDE):
                for dc in range(0, BLOCK - PATCH + 1, STRIDE):
                    r, c = br * BLOCK + dr, bc * BLOCK + dc
                    vin = s["valid_in"][r:r + PATCH, c:c + PATCH]
                    if vin.mean() < 0.6:
                        continue
                    X.append(img[r:r + PATCH, c:c + PATCH])
                    M.append(s["hr"][r:r + PATCH, c:c + PATCH])
                    R.append(np.nan_to_num(s["risk"][r:r + PATCH, c:c + PATCH]))
                    V.append(vin)
                    meta.append((MONTHS.index(m), r, c, br * (Wd // BLOCK) + bc))
X = np.asarray(X, np.float16); M = np.asarray(M, np.uint8); R = np.asarray(R, np.float16)
V = np.asarray(V, np.uint8); meta = np.asarray(meta, np.int32)
frac = M.sum((1, 2)) / np.maximum(V.sum((1, 2)), 1)
FRAC_THR = float(np.quantile(frac, POS_QUANTILE))
y = (frac > FRAC_THR).astype(np.uint8)
# random spatial-block split (same block -> same split in both months, so no leakage)
rng = np.random.RandomState(42)
blocks = np.unique(meta[:, 3]); rng.shuffle(blocks)
n = len(blocks); tr_b, va_b = blocks[:int(.6 * n)], blocks[int(.6 * n):int(.8 * n)]
split = np.where(np.isin(meta[:, 3], tr_b), 0, np.where(np.isin(meta[:, 3], va_b), 1, 2)).astype(np.uint8)
print(f"patch high-risk-fraction cutoff {FRAC_THR:.3f} | patches: {len(X)} | class balance (high-risk=1): {y.mean():.3f} | train/val/test = "
      f"{(split==0).sum()}/{(split==1).sum()}/{(split==2).sum()}")
for k, nm in enumerate(("train", "val", "test")):
    print(f"  {nm}: positives {y[split==k].sum()} / {(split==k).sum()}")
np.savez_compressed(OUT_DIR / "step2_patches.npz", X=X, mask=M, risk=R, valid=V, y=y,
                    split=split, meta=meta, months=np.array(MONTHS))

# ---------- 5. save scene-level fields for later overlays ----------
for m in MONTHS:
    s = scene[m]
    np.savez_compressed(OUT_DIR / f"step2_scene_{m}.npz", ndvi=s["ndvi"].astype(np.float16),
                        ndwi=s["ndwi"].astype(np.float16), valid=s["valid_in"], water=s["water"],
                        risk=s["risk"].astype(np.float16), hr=s["hr"], sea=sea)
prof0 = scl_ds(usable[0])[1]
tr = prof0["transform"]
json.dump(dict(months=MONTHS, ds=DS, shape=[H, Wd], crs=str(prof0["crs"]),
               transform=[tr.a * DS, tr.b, tr.c, tr.d, tr.e * DS, tr.f], patch=PATCH, stride=STRIDE,
               risk_threshold=THR, patch_frac_threshold=FRAC_THR, ndwi_p5_p95=[float(p5), float(p95)], weights=W, d0_km=D0_KM),
          open(OUT_DIR / "step2_meta.json", "w"), indent=1)

# ---------- 6. figure ----------
fig, ax = plt.subplots(2, 5, figsize=(26, 10))
for row, m in enumerate(MONTHS):
    s = scene[m]
    cat = np.zeros((H, Wd, 3))
    cat[~s["valid_in"]] = (0.1, 0.1, 0.1); cat[s["valid_in"]] = (0.85, 0.8, 0.6)
    cat[sea] = (0.05, 0.1, 0.5); cat[s["water"]] = (0.2, 0.8, 1.0)
    inv = ~s["valid_in"]
    panels = [(np.ma.masked_where(inv, s["ndvi"]), "YlGn", "NDVI (decoded)", (0, 0.7)),
              (np.ma.masked_where(inv, s["ndwi"]), "BrBG", "NDWI (decoded)", (-1, 1)),
              (cat, None, "sea / inland water / land / no-data", None),
              (np.ma.masked_invalid(s["risk"]), "inferno", "proxy risk score", None),
              (None, None, "high-risk mask (top 20%)", None)]
    for a, (img, cm, t, lim) in zip(ax[row], panels):
        if img is None:
            o = np.zeros((H, Wd, 3)); o[~s["valid_in"]] = .1; o[s["valid_risk"]] = (.75, .75, .75)
            o[sea] = (.05, .1, .5); o[s["water"]] = (.2, .8, 1); o[s["hr"]] = (.9, .1, .1)
            a.imshow(o)
        elif cm is None:
            a.imshow(img)
        else:
            a.imshow(img, cmap=cm, vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1])
        a.set_title(f"{m}  {t}", fontsize=10); a.axis("off")
plt.tight_layout(); plt.savefig(OUT_DIR / "step2_decoded_and_risk.png", dpi=55)

# patch examples
idx = np.random.RandomState(1).choice(len(X), 6, replace=False)
fig, ax = plt.subplots(3, 6, figsize=(15, 7.5))
for j, i in enumerate(idx):
    ax[0, j].imshow(X[i][..., 0].astype(float), cmap="YlGn", vmin=0, vmax=.7); ax[0, j].set_title(f"NDVI  y={y[i]}", fontsize=9)
    ax[1, j].imshow(X[i][..., 1].astype(float), cmap="BrBG", vmin=-1, vmax=1); ax[1, j].set_title("NDWI", fontsize=9)
    ax[2, j].imshow(M[i], cmap="Reds"); ax[2, j].set_title("high-risk mask", fontsize=9)
    for k in range(3): ax[k, j].axis("off")
plt.tight_layout(); plt.savefig(OUT_DIR / "step2_patch_examples.png", dpi=60)
print("done")
