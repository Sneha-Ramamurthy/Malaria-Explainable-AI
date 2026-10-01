"""STEP 4 - Grad-CAM: patch heatmaps, scene heatmaps, focus analysis and sanity check.
Run:  python -m src.step4_gradcam
"""
import os, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras, rasterio
from rasterio.transform import Affine
from scipy.stats import wilcoxon, spearmanr
from .config import OUT_DIR, ROOT
from .gradcam import load_model, GradCAM
from .models import build_model

M3 = json.load(open(OUT_DIR / "step3_metrics.json")); M2 = json.load(open(OUT_DIR / "step2_meta.json"))
THR, MONTHS, PATCH = M3["threshold"], M2["months"], M2["patch"]
H, W = M2["shape"]
model = load_model(ROOT / "models" / "mobilenetv2_cbam.keras")
gc = GradCAM(model, "last_conv")

d = np.load(OUT_DIR / "step2_patches.npz")
X, y, split, valid, hrm, meta = (d[k] for k in ("X", "y", "split", "valid", "mask", "meta"))
X = X.astype("float32"); valid = valid.astype(bool); hrm = hrm.astype(bool)
scene = {m: np.load(OUT_DIR / f"step2_scene_{m}.npz") for m in MONTHS}

# ------------------------------------------------------------------ 1. CAMs for every patch
CACHE = bool(os.environ.get("CACHE"))
if CACHE:
    L = np.load(OUT_DIR / "step4_gradcam_patches.npz"); cam = L["cam"].astype("float32"); prob = L["prob"]; empty = L["empty"]
    logit = np.log(prob / (1 - prob))
else:
    cam, logit, empty = gc(X); prob = 1 / (1 + np.exp(-logit))
te = split == 2
print(f"CAMs computed: {len(cam)} patches | empty (no positive evidence): {empty.sum()} ({empty.mean():.1%})")
if not CACHE:
    np.savez_compressed(OUT_DIR / "step4_gradcam_patches.npz", cam=cam.astype(np.float16), prob=prob.astype(np.float32),
                        empty=empty)

# environmental masks per patch (from scene-level layers)
water_dil = {m: cv2.dilate(scene[m]["water"].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) for m in MONTHS}  # ~2.3 km
def sl(a, i): mi, r, c = meta[i, 0], meta[i, 1], meta[i, 2]; return a[r:r + PATCH, c:c + PATCH]
near_water = np.stack([sl(water_dil[MONTHS[meta[i, 0]]], i) for i in range(len(X))])
ndwi_p75 = float(np.percentile(X[split == 0][..., 1][valid[split == 0]], 75))
moist = (X[..., 1] >= ndwi_p75) & valid
vegd = (X[..., 0] >= 0.35) & valid

def focus_stats(camset, idx):
    """Enrichment = share of top-20%-CAM pixels that are in a class / share of valid pixels in that class."""
    rows = {k: [] for k in ("high_risk", "near_water", "moist", "vegetated", "invalid_mass")}
    for i in idx:
        c, v = camset[i], valid[i]
        if v.sum() < 100 or c.max() <= 0: continue
        top = (c >= np.quantile(c, 0.8)) & (c > 0) & v
        if top.sum() < 20: continue
        for k, m in (("high_risk", hrm[i]), ("near_water", near_water[i]), ("moist", moist[i]), ("vegetated", vegd[i])):
            base = m[v].mean()
            rows[k].append(m[top].mean() / base if base > 0.02 else np.nan)
        rows["invalid_mass"].append((c[~v].sum() / c.sum()) / max((~v).mean(), 1e-6) if (~v).mean() > 0.02 else np.nan)
    return {k: np.array(v, float) for k, v in rows.items()}

def summarise(st):
    out = {}
    for k, a in st.items():
        a = a[~np.isnan(a)]
        if len(a) < 5: out[k] = None; continue
        try: p = float(wilcoxon(a - 1).pvalue)
        except ValueError: p = float("nan")
        out[k] = dict(n=int(len(a)), median=float(np.median(a)), mean=float(a.mean()),
                      share_above_1=float((a > 1).mean()), wilcoxon_p_vs_1=p)
    return out

idx_all = np.where(te)[0]; idx_pos = np.where(te & (prob >= 1 / (1 + np.exp(-np.log(THR / (1 - THR))))))[0]
st_all, st_pos = focus_stats(cam, idx_all), focus_stats(cam, idx_pos)
S_all, S_pos = summarise(st_all), summarise(st_pos)

# ------------------------------------------------------------------ 2. sanity check: randomised model
keras.utils.set_random_seed(123)
rnd = build_model(M3["norm_mean"], M3["norm_var"], **M3["model_config"])
cam_r, _, _ = GradCAM(rnd, "last_conv")(X[te])
st_rnd = focus_stats(dict(zip(idx_all, cam_r)), idx_all)
S_rnd = summarise(st_rnd)
rho = []
for a, b, i in zip(cam[te], cam_r, idx_all):
    v = valid[i]
    if a.max() > 0 and b.max() > 0 and v.sum() > 100 and a[v].std() > 0 and b[v].std() > 0:
        rho.append(spearmanr(a[v], b[v])[0])
rho = float(np.nanmean(rho))
print(f"\nFOCUS (test patches predicted high-risk, n={len(idx_pos)}); enrichment >1 = model looks at it more than chance")
for k, v in S_pos.items(): print(f"  {k:13s}", None if v is None else {a: round(b, 3) for a, b in v.items()})
print("SANITY  trained vs randomised-model Grad-CAM: mean Spearman rho =", round(rho, 3))
print("        high-risk enrichment median  trained", None if S_all['high_risk'] is None else round(S_all['high_risk']['median'], 2),
      "| randomised", None if S_rnd['high_risk'] is None else round(S_rnd['high_risk']['median'], 2))

# ------------------------------------------------------------------ 3. scene-level Grad-CAM (sliding window)
region = {m: np.zeros((H, W), np.uint8) for m in MONTHS}
for i in range(len(X)):
    mm = MONTHS[meta[i, 0]]; r, c = meta[i, 1], meta[i, 2]
    region[mm][r:r + PATCH, c:c + PATCH] = np.maximum(region[mm][r:r + PATCH, c:c + PATCH], split[i] + 1)
tr = M2["transform"]; affine = Affine(*tr); scene_cam, scene_stats = {}, {}
for m in MONTHS:
    s = scene[m]; sv = s["valid"]
    cf = OUT_DIR / f"step4_scene_cam_{m}.npz"
    if CACHE and cf.exists():
        sc = np.load(cf)["cam"].astype("float32"); scene_cam[m] = sc
    else:
        img = np.stack([s["ndvi"], s["ndwi"]], -1).astype("float32") * sv[..., None]
        rs, cs, wins = [], [], []
        for r in range(0, H - PATCH + 1, 32):
            for c in range(0, W - PATCH + 1, 32):
                if sv[r:r + PATCH, c:c + PATCH].mean() >= 0.6:
                    rs.append(r); cs.append(c); wins.append(img[r:r + PATCH, c:c + PATCH])
        cw, _, _ = gc(np.stack(wins), batch=256)
        acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
        for r, c, k in zip(rs, cs, cw): acc[r:r + PATCH, c:c + PATCH] += k; cnt[r:r + PATCH, c:c + PATCH] += 1
        sc = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32); scene_cam[m] = sc
        np.savez_compressed(OUT_DIR / f"step4_scene_cam_{m}.npz", cam=sc.astype(np.float16))
        with rasterio.open(OUT_DIR / f"step4_gradcam_{m}.tif", "w", driver="GTiff", height=H, width=W, count=1,
                           dtype="float32", crs=M2["crs"], transform=affine, nodata=-9999) as dst:
            dst.write(np.nan_to_num(sc, nan=-9999).astype("float32"), 1)      # open in QGIS

    prb = np.load(OUT_DIR / f"step3_scene_prob_{m}.npz")["prob"].astype("float32")
    ok = sv & ~s["water"] & ~np.isnan(sc) & (region[m] == 3)              # unseen (test) blocks only
    top = ok & (sc >= np.nanquantile(sc[ok], 0.8)) if ok.sum() > 100 else ok
    pred = np.nan_to_num(prb, nan=0) >= THR
    scene_stats[m] = dict(test_px=int(ok.sum()),
        top20cam_inside_pred_highrisk=float(pred[top].mean()) if top.sum() else None,
        pred_highrisk_share_of_test_land=float(pred[ok].mean()) if ok.sum() else None,
        top20cam_inside_proxy_highrisk=float(s["hr"][top].mean()) if top.sum() else None,
        proxy_highrisk_share_of_test_land=float(s["hr"][ok].mean()) if ok.sum() else None,
        spearman_cam_vs_prob=float(spearmanr(sc[ok], prb[ok])[0]) if ok.sum() > 100 else None)
    print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in scene_stats[m].items()})

json.dump(dict(layer="last_conv", threshold=THR, empty_cam_fraction=float(empty.mean()),
               focus_test_predicted_high=S_pos, focus_test_all=S_all, focus_randomised_model=S_rnd,
               sanity_spearman_trained_vs_random=rho, scene=scene_stats),
          open(OUT_DIR / "step4_metrics.json", "w"), indent=1)

# ------------------------------------------------------------------ 4. figures
def heat(ax, base, c, cmap, vmin, vmax, title):
    ax.imshow(base, cmap=cmap, vmin=vmin, vmax=vmax); ax.imshow(np.ma.masked_less(c, 0.15), cmap="jet", alpha=0.55, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8); ax.axis("off")

tp = idx_all[(y[idx_all] == 1) & (prob[idx_all] >= prob[te].mean())]; tp = tp[np.argsort(-prob[tp])][:5]
fn = idx_all[(y[idx_all] == 1) & ~np.isin(idx_all, tp)][:1]
tn = idx_all[(y[idx_all] == 0)]; tn = tn[np.argsort(prob[tn])][:2]
sel = np.concatenate([tp, fn, tn])
fig, ax = plt.subplots(5, len(sel), figsize=(2.3 * len(sel), 12))
for j, i in enumerate(sel):
    v = valid[i]; t = f"p={prob[i]:.2f}  true={'HIGH' if y[i] else 'low'}"
    ax[0, j].imshow(np.ma.masked_where(~v, X[i][..., 0]), cmap="YlGn", vmin=0, vmax=.7); ax[0, j].set_title("NDVI\n" + t, fontsize=8)
    ax[1, j].imshow(np.ma.masked_where(~v, X[i][..., 1]), cmap="BrBG", vmin=-1, vmax=1); ax[1, j].set_title("NDWI", fontsize=8)
    heat(ax[2, j], np.ma.masked_where(~v, X[i][..., 0]), cam[i], "YlGn", 0, .7, "Grad-CAM on NDVI")
    heat(ax[3, j], np.ma.masked_where(~v, X[i][..., 1]), cam[i], "BrBG", -1, 1, "Grad-CAM on NDWI")
    ax[4, j].imshow(hrm[i], cmap="Reds", vmin=0, vmax=1.5)
    if cam[i].max() > 0: ax[4, j].contour(cam[i], levels=[0.5], colors="cyan", linewidths=1.2)
    ax[4, j].set_title("proxy high-risk mask\n+ CAM>0.5 (cyan)", fontsize=8)
    for a in ax[:, j]: a.axis("off")
plt.tight_layout(); plt.savefig(OUT_DIR / "step4_gradcam_patches.png", dpi=75); plt.close()

for m in MONTHS:
    s = scene[m]; sv = s["valid"]; prb = np.load(OUT_DIR / f"step3_scene_prob_{m}.npz")["prob"].astype("float32")
    sc = scene_cam[m]; pred = np.nan_to_num(prb, nan=0) >= THR
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.5))
    heat(ax[0], np.ma.masked_where(~sv, s["ndvi"].astype("float32")), np.nan_to_num(sc), "YlGn", 0, .7, f"{m}  Grad-CAM over NDVI")
    heat(ax[1], np.ma.masked_where(~sv, s["ndwi"].astype("float32")), np.nan_to_num(sc), "BrBG", -1, 1, "Grad-CAM over NDWI")
    ax[2].imshow(np.ma.masked_invalid(sc), cmap="magma", vmin=0, vmax=1)
    ax[2].contour(pred & sv & ~s["water"], levels=[0.5], colors="cyan", linewidths=0.6)
    ax[2].contour(s["hr"] & sv, levels=[0.5], colors="lime", linewidths=0.4)
    ax[2].set_title("Grad-CAM (mean of windows)  cyan = predicted high-risk, green = proxy high-risk", fontsize=9)
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT_DIR / f"step4_gradcam_scene_{m}.png", dpi=55); plt.close()

names = ["high_risk", "near_water", "moist", "vegetated"]; lab = ["proxy\nhigh-risk", "near\nwater", "moist\n(NDWI top 25%)", "vegetated\n(NDVI ≥ 0.35)"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
xs = np.arange(4); wd = .38
med = lambda S: [S[k]["median"] if S.get(k) else np.nan for k in names]
ax[0].bar(xs - wd / 2, med(S_all), wd, label="trained model"); ax[0].bar(xs + wd / 2, med(S_rnd), wd, label="randomised model")
ax[0].axhline(1, color="k", ls="--"); ax[0].set_xticks(xs, lab); ax[0].set_ylabel("median enrichment (1 = chance)")
ax[0].set_title("Where does the top-20% of Grad-CAM fall? (test patches)"); ax[0].legend()
ax[1].hist(st_all["high_risk"][~np.isnan(st_all["high_risk"])], bins=25, alpha=.7, label="trained")
ax[1].hist(st_rnd["high_risk"][~np.isnan(st_rnd["high_risk"])], bins=25, alpha=.5, label="randomised")
ax[1].axvline(1, color="k", ls="--"); ax[1].set_title("High-risk enrichment per patch"); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT_DIR / "step4_focus_analysis.png", dpi=80); plt.close()
print("done")
