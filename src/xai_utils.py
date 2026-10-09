"""Shared helpers for the XAI steps (5 SHAP, 6 CBAM, later 7 fusion / 8 final explanation).

Deliberately NumPy / SciPy / OpenCV only (no TensorFlow import) so everything here is unit-testable
and reusable by the fusion step without loading a model.

Conventions (same as Step 4):
  * all patch arrays are aligned with outputs/step2_patches.npz  (N,64,64,...)
  * a "map" is a heatmap in [0,1] (1 = most important); 0 outside the valid (observed) pixels
  * "top-20%" regions are the 20% most important valid pixels of a patch
"""
import json
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.stats import spearmanr, wilcoxon

from .config import OUT_DIR

CHANCE_TOP20_IOU = 0.2 / (2 - 0.2)      # expected IoU of two independent random 20% regions (~0.111)


# ----------------------------------------------------------------------------- context
def load_context():
    """Everything Steps 5-8 need from the outputs of Steps 2-4."""
    M2 = json.load(open(OUT_DIR / "step2_meta.json"))
    M3 = json.load(open(OUT_DIR / "step3_metrics.json"))
    d = np.load(OUT_DIR / "step2_patches.npz")
    months = M2["months"]
    return SimpleNamespace(
        M2=M2, M3=M3, THR=M3["threshold"], MONTHS=months, PATCH=M2["patch"], H=M2["shape"][0], W=M2["shape"][1],
        X=d["X"].astype("float32"), y=d["y"], split=d["split"], meta=d["meta"],
        valid=d["valid"].astype(bool), hrm=d["mask"].astype(bool),
        scene={m: np.load(OUT_DIR / f"step2_scene_{m}.npz") for m in months})


def env_masks(ctx):
    """Per-patch environmental masks used to ask 'does the explanation point at water / moisture / vegetation?'
    (identical definitions to Step 4)."""
    P = ctx.PATCH
    wd = {m: cv2.dilate(ctx.scene[m]["water"].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
          for m in ctx.MONTHS}                                                   # ~2.3 km around water
    near = np.stack([wd[ctx.MONTHS[mi]][r:r + P, c:c + P] for mi, r, c, _ in ctx.meta])
    tr = ctx.split == 0
    p75 = float(np.percentile(ctx.X[tr][..., 1][ctx.valid[tr]], 75))
    return dict(high_risk=ctx.hrm, near_water=near,
                moist=(ctx.X[..., 1] >= p75) & ctx.valid,
                vegetated=(ctx.X[..., 0] >= 0.35) & ctx.valid)


# ----------------------------------------------------------------------------- map helpers
def minmax01(a, v=None):
    """Per-patch min-max to [0,1] over valid pixels; invalid pixels -> 0. a: (N,H,W), v: (N,H,W) bool or None."""
    a = np.asarray(a, np.float32)
    out = np.zeros_like(a)
    for i in range(len(a)):
        m = np.ones(a[i].shape, bool) if v is None else v[i]
        if m.sum() == 0:
            continue
        lo, hi = a[i][m].min(), a[i][m].max()
        if hi - lo > 1e-12:
            out[i] = np.where(m, (a[i] - lo) / (hi - lo), 0)
    return out


def max01(a, v=None):
    """Divide each map by its max (keeps 0 = no evidence). Used for non-negative evidence maps."""
    a = np.asarray(a, np.float32).copy()
    if v is not None:
        a = a * v
    mx = a.reshape(len(a), -1).max(1)
    return a / np.maximum(mx, 1e-12)[:, None, None]


def smooth(a, sigma):
    """Gaussian-smooth a stack of 2-D maps (N,H,W)."""
    if sigma <= 0:
        return a
    return np.stack([cv2.GaussianBlur(np.ascontiguousarray(m, np.float32), (0, 0), sigma) for m in a])


def top_region(c, v, q=0.2):
    """Boolean mask of the top-q fraction of valid, strictly positive pixels of one map."""
    if v.sum() < 20 or c.max() <= 0:
        return np.zeros_like(v, bool)
    return (c >= np.quantile(c[v], 1 - q)) & (c > 0) & v


# ----------------------------------------------------------------------------- explanation <-> label/environment
def focus_stats(maps, ctx, masks, idx, q=0.2):
    """Enrichment = share of the top-q map pixels that fall in a class / share of valid pixels in that class.
    >1 means the explanation looks at that class more than chance."""
    rows = {k: [] for k in list(masks) + ["invalid_mass"]}
    for i in idx:
        c, v = maps[i], ctx.valid[i]
        if v.sum() < 100 or c.max() <= 0:
            continue
        top = top_region(c, v, q)
        if top.sum() < 20:
            continue
        for k, m in masks.items():
            base = m[i][v].mean()
            rows[k].append(m[i][top].mean() / base if base > 0.02 else np.nan)
        rows["invalid_mass"].append((c[~v].sum() / max(c.sum(), 1e-12)) / max((~v).mean(), 1e-6)
                                    if (~v).mean() > 0.02 else np.nan)
    return {k: np.array(v, float) for k, v in rows.items()}


def summarise(stats):
    out = {}
    for k, a in stats.items():
        a = a[~np.isnan(a)]
        if len(a) < 5:
            out[k] = None
            continue
        try:
            p = float(wilcoxon(a - 1).pvalue)
        except ValueError:
            p = float("nan")
        out[k] = dict(n=int(len(a)), median=float(np.median(a)), mean=float(a.mean()),
                      share_above_1=float((a > 1).mean()), wilcoxon_p_vs_1=p)
    return out


def patch_agreement(a, b, valid, idx, q=0.2):
    """Spatial agreement of two explanation maps per patch.
    Returns (spearman rho array, top-q IoU array); patches where either map is empty are skipped."""
    rho, iou = [], []
    for i in idx:
        v = valid[i]
        if v.sum() < 100 or a[i].max() <= 0 or b[i].max() <= 0:
            continue
        x, y = a[i][v], b[i][v]
        if x.std() > 0 and y.std() > 0:
            rho.append(spearmanr(x, y)[0])
        ta, tb = top_region(a[i], v, q), top_region(b[i], v, q)
        u = (ta | tb).sum()
        if u:
            iou.append((ta & tb).sum() / u)
    return np.array(rho, float), np.array(iou, float)


def describe(arr):
    arr = np.asarray(arr, float)
    arr = arr[~np.isnan(arr)]
    if not len(arr):
        return None
    return dict(n=int(len(arr)), mean=float(arr.mean()), median=float(np.median(arr)),
                p25=float(np.percentile(arr, 25)), p75=float(np.percentile(arr, 75)))


# ----------------------------------------------------------------------------- scenes

def scene_windows(sc, ctx, stride=32):
    """Build valid sliding windows from a scene."""
    sv = np.asarray(sc["valid"], dtype=bool)
    img = np.stack(
        [sc["ndvi"], sc["ndwi"]], axis=-1
    ).astype(np.float32) * sv[..., None]

    P = ctx.PATCH
    rs, cs, wins = [], [], []

    if ctx.H < P or ctx.W < P:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.empty((0, P, P, 2), dtype=np.float32),
        )

    row_starts = list(range(0, ctx.H - P + 1, stride))
    col_starts = list(range(0, ctx.W - P + 1, stride))

    if not row_starts or row_starts[-1] != ctx.H - P:
        row_starts.append(ctx.H - P)

    if not col_starts or col_starts[-1] != ctx.W - P:
        col_starts.append(ctx.W - P)

    for r in row_starts:
        for c in col_starts:
            window_valid = sv[r:r + P, c:c + P]

            if window_valid.mean() >= 0.6:
                rs.append(r)
                cs.append(c)
                wins.append(img[r:r + P, c:c + P])

    if not wins:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.empty((0, P, P, 2), dtype=np.float32),
        )

    return (
        np.asarray(rs, dtype=int),
        np.asarray(cs, dtype=int),
        np.stack(wins).astype(np.float32),
    )



def stitch(rs, cs, maps, ctx):
    """Average overlapping window maps (n,P,P[,k]) back onto the scene grid. NaN where no window covers."""
    P = ctx.PATCH
    extra = maps.shape[3:]
    acc = np.zeros((ctx.H, ctx.W) + extra, np.float32)
    cnt = np.zeros((ctx.H, ctx.W), np.float32)
    for r, c, m in zip(rs, cs, maps):
        acc[r:r + P, c:c + P] += m
        cnt[r:r + P, c:c + P] += 1
    cnt_b = cnt.reshape(cnt.shape + (1,) * len(extra))
    return np.where(cnt_b > 0, acc / np.maximum(cnt_b, 1), np.nan).astype(np.float32)


def region_map(ctx):
    """Per month: 0 none / 1 train / 2 val / 3 test blocks (as in Steps 3/4)."""
    P = ctx.PATCH
    reg = {m: np.zeros((ctx.H, ctx.W), np.uint8) for m in ctx.MONTHS}
    for i in range(len(ctx.X)):
        m = ctx.MONTHS[ctx.meta[i, 0]]; r, c = ctx.meta[i, 1], ctx.meta[i, 2]
        reg[m][r:r + P, c:c + P] = np.maximum(reg[m][r:r + P, c:c + P], ctx.split[i] + 1)
    return reg


def scene_agreement(sc_map, ref_map, prob, hr, ok, thr, q=0.2):
    """Scene-level stats on the unseen (test-block) land pixels `ok`."""
    sel = ok & ~np.isnan(sc_map)
    if sel.sum() < 100:
        return None
    top = sel & (sc_map >= np.nanquantile(sc_map[sel], 1 - q))
    pred = np.nan_to_num(prob, nan=0) >= thr
    out = dict(test_px=int(sel.sum()),
               top20_inside_pred_highrisk=float(pred[top].mean()),
               pred_highrisk_share_of_test_land=float(pred[sel].mean()),
               top20_inside_proxy_highrisk=float(hr[top].mean()),
               proxy_highrisk_share_of_test_land=float(hr[sel].mean()),
               spearman_vs_prob=float(spearmanr(sc_map[sel], prob[sel])[0]))
    if ref_map is not None:
        s2 = sel & ~np.isnan(ref_map)
        if s2.sum() > 100:
            t2 = s2 & (ref_map >= np.nanquantile(ref_map[s2], 1 - q))
            t1 = s2 & (sc_map >= np.nanquantile(sc_map[s2], 1 - q))
            out["spearman_vs_gradcam"] = float(spearmanr(sc_map[s2], ref_map[s2])[0])
            out["top20_iou_vs_gradcam"] = float((t1 & t2).sum() / max((t1 | t2).sum(), 1))
    return out


def save_geotiff(path, bands, descriptions, ctx, nodata=-9999.0):
    """Write float32 GeoTIFF (QGIS-ready) on the project grid. bands: list of (H,W) arrays (NaN -> nodata)."""
    import rasterio
    from rasterio.transform import Affine
    with rasterio.open(path, "w", driver="GTiff", height=ctx.H, width=ctx.W, count=len(bands), dtype="float32",
                       crs=ctx.M2["crs"], transform=Affine(*ctx.M2["transform"]), nodata=nodata) as dst:
        for k, (b, d) in enumerate(zip(bands, descriptions), 1):
            dst.write(np.nan_to_num(np.asarray(b, np.float32), nan=nodata).astype("float32"), k)
            dst.set_band_description(k, d)


def pick_examples(ctx, prob, n_tp=5, n_fn=1, n_tn=2):
    """Same example selection as the Step 4 figure so panels are comparable across steps."""
    te = np.where(ctx.split == 2)[0]; y = ctx.y
    tp = te[(y[te] == 1) & (prob[te] >= prob[te].mean())]
    tp = tp[np.argsort(-prob[tp])][:n_tp]
    fn = te[(y[te] == 1) & ~np.isin(te, tp)][:n_fn]
    tn = te[y[te] == 0]
    tn = tn[np.argsort(prob[tn])][:n_tn]
    return np.concatenate([tp, fn, tn]).astype(int)
