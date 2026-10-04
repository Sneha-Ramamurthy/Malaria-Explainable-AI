"""Backend-free numerics for Step 7 (explanation validation & fusion) and Step 8 (final explainable output).

NumPy / SciPy only (no TensorFlow import), in the same spirit as xai_utils.py, so everything here is unit-testable
without loading a model.  Conventions are unchanged from Steps 4-6:
  * patch arrays are aligned with outputs/step2_patches.npz      (N,64,64,...)
  * an explanation "map" is a heatmap in [0,1] (1 = most important), 0 outside the valid (observed) pixels
  * the "top-20% region" of a map = the 20% most important valid pixels of that patch
"""
import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon

from . import xai_utils as U

CHANCE = U.CHANCE_TOP20_IOU          # expected IoU of two independent random 20% regions (~0.111)


# ============================================================================ fusion
def rank01(a, v=None):
    """Rank-normalise to [0,1] over valid pixels (average rank on ties); invalid / non-finite pixels -> 0.
    Rank-normalising puts Grad-CAM (sparse, many exact zeros), SHAP (smooth evidence) and CBAM (near-flat sigmoid)
    on one comparable scale before they are averaged.  a: (N,H,W) or (H,W); v: bool mask of the same shape or None."""
    a = np.asarray(a, np.float32)
    two_d = a.ndim == 2
    if two_d:
        a = a[None]
        v = None if v is None else np.asarray(v)[None]
    out = np.zeros_like(a)
    for i in range(len(a)):
        m = np.isfinite(a[i]) if v is None else (np.asarray(v[i], bool) & np.isfinite(a[i]))
        n = int(m.sum())
        if n < 2:
            continue
        r = rankdata(a[i][m], method="average")
        out[i][m] = ((r - 1) / (n - 1)).astype(np.float32)
    return out[0] if two_d else out


def fuse(maps, valid, weights=None):
    """Weighted mean of the rank-normalised maps, re-scaled to [0,1] over valid pixels.
    maps: list of (N,H,W) or (H,W) arrays.  Returns the same shape; 0 outside `valid`."""
    w = np.ones(len(maps)) if weights is None else np.asarray(weights, np.float64)
    acc = sum(wi * rank01(m, valid) for wi, m in zip(w, maps)) / w.sum()
    acc = np.asarray(acc, np.float32)
    if acc.ndim == 2:
        return U.minmax01(acc[None], np.asarray(valid, bool)[None])[0]
    return U.minmax01(acc, np.asarray(valid, bool))


def top_masks(maps, valid, q=0.2):
    """Boolean top-q region of every map. maps: list of (N,H,W) -> list of (N,H,W) bool."""
    return [np.stack([U.top_region(m[i], valid[i], q) for i in range(len(m))]) for m in maps]


def agreement_count(maps, valid, q=0.2):
    """(N,H,W) uint8 = for every pixel, how many of the methods have it inside their own top-q region (0..len(maps))."""
    return np.sum(top_masks(maps, valid, q), axis=0).astype(np.uint8)


def top_scene(m, ok, q=0.2):
    """Scene version of top_region: pixels of `ok` in the top-q of the map (NaN ignored)."""
    sel = ok & np.isfinite(m)
    if sel.sum() < 100:
        return np.zeros(m.shape, bool)
    return sel & (m >= np.quantile(m[sel], 1 - q))


# ============================================================================ spatial agreement / overlap
def pair_agreement(a, b, valid, q=0.2):
    """Per-patch Spearman rho and top-q IoU between two explanation maps, ALIGNED with the patch index
    (NaN where a patch is skipped: <100 valid pixels or an empty / constant map).  a, b: (N,H,W)."""
    n = len(a)
    rho = np.full(n, np.nan); iou = np.full(n, np.nan)
    for i in range(n):
        v = valid[i]
        if v.sum() < 100 or a[i].max() <= 0 or b[i].max() <= 0:
            continue
        x, y = a[i][v], b[i][v]
        if x.std() > 0 and y.std() > 0:
            rho[i] = spearmanr(x, y)[0]
        ta, tb = U.top_region(a[i], v, q), U.top_region(b[i], v, q)
        u = (ta | tb).sum()
        if u:
            iou[i] = (ta & tb).sum() / u
    return rho, iou


def chance_iou(a, p, n):
    """Expected IoU of a random region of `a` pixels with a fixed region of `p` pixels inside `n` pixels."""
    inter = a * p / max(n, 1)
    return float(inter / max(a + p - inter, 1e-12))


def region_overlap(m, valid, mask, q=0.2):
    """Per patch: IoU between the map's top-q region and a boolean `mask` (e.g. the high-risk region), plus the IoU
    expected by chance for a random region of that size.  Both NaN where the mask is empty / the patch is skipped."""
    n = len(m)
    iou = np.full(n, np.nan); chance = np.full(n, np.nan)
    for i in range(n):
        v = valid[i]
        if v.sum() < 100 or m[i].max() <= 0 or mask[i][v].sum() == 0:
            continue
        t = U.top_region(m[i], v, q)
        if t.sum() == 0:
            continue
        mk = mask[i] & v
        iou[i] = (t & mk).sum() / max((t | mk).sum(), 1)
        chance[i] = chance_iou(int(t.sum()), int(mk.sum()), int(v.sum()))
    return iou, chance


# ============================================================================ stability
def add_noise(X, valid, sd, level, seed=0):
    """Gaussian noise (per-feature sd * level) added to the VALID pixels only. X (N,H,W,C); sd (C,)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=X.shape).astype(np.float32) * (np.asarray(sd, np.float32) * float(level))
    return (X + noise * np.asarray(valid, np.float32)[..., None]).astype(np.float32)


# ============================================================================ perturbation / fidelity
def deletion_orders(m, valid, rng):
    """Per patch: flat indices of the valid pixels sorted from most to least important (random tie-break, which matters
    for Grad-CAM where many pixels are exactly 0)."""
    orders = []
    for i in range(len(m)):
        idx = np.flatnonzero(np.asarray(valid[i], bool).ravel())
        val = np.asarray(m[i], np.float64).ravel()[idx]
        orders.append(idx[np.lexsort((rng.random(len(idx)), -val))])
    return orders


def delete_pixels(X, orders, frac, fill, which="top", rng=None):
    """Copy of X in which `frac` of every patch's valid pixels is replaced by `fill` (C,).
    which = 'top' (most important first) | 'bottom' (least important first) | 'random'."""
    Xd = X.copy()
    for i, o in enumerate(orders):
        k = int(round(frac * len(o)))
        if k == 0:
            continue
        if which == "top":
            sel = o[:k]
        elif which == "bottom":
            sel = o[len(o) - k:]
        else:
            sel = rng.choice(o, k, replace=False)
        r, c = np.unravel_index(sel, X.shape[1:3])
        Xd[i, r, c, :] = fill
    return Xd


def deletion_test(logit_fn, X, valid, maps, fill, fracs, n_random=5, seed=0):
    """Faithfulness by perturbation: replace the most important pixels by the training-mean value (= 0 after the model's
    Normalization layer) and measure how much the logit DROPS.  Controls: random pixels of the same count
    (averaged over `n_random` draws; shared by all methods) and the LEAST important pixels.
    maps: {name: (N,H,W)}.  Returns {'base': (N,), 'fractions': [...], 'random': (F,N),
                                     name: {'top': (F,N), 'bottom': (F,N)}}."""
    rng = np.random.default_rng(seed)
    fill = np.asarray(fill, np.float32)
    base = np.asarray(logit_fn(X), np.float64)
    orders = {k: deletion_orders(m, valid, rng) for k, m in maps.items()}
    any_order = next(iter(orders.values()))
    out = {"base": base, "fractions": list(map(float, fracs))}
    out["random"] = np.stack([np.mean([base - np.asarray(logit_fn(delete_pixels(X, any_order, f, fill, "random", rng)))
                                       for _ in range(n_random)], axis=0) for f in fracs])
    for k in maps:
        out[k] = dict(
            top=np.stack([base - np.asarray(logit_fn(delete_pixels(X, orders[k], f, fill, "top"))) for f in fracs]),
            bottom=np.stack([base - np.asarray(logit_fn(delete_pixels(X, orders[k], f, fill, "bottom"))) for f in fracs]))
    return out


def summarise_deletion(top, rand, bottom, fracs, idx):
    """top / rand / bottom: (F,N) logit drops; idx: patch positions to evaluate.
    AOPC = drop averaged over the deletion fractions.  gain = AOPC(top) - AOPC(random): >0 means the method's
    important pixels matter more to the model than arbitrary pixels (paired Wilcoxon over patches)."""
    idx = np.asarray(idx, int)
    t, r, b = top[:, idx], rand[:, idx], bottom[:, idx]
    at, ar, ab = t.mean(0), r.mean(0), b.mean(0)
    gain = at - ar
    try:
        p = float(wilcoxon(gain).pvalue)
    except ValueError:
        p = float("nan")
    return dict(n=int(len(idx)), fractions=list(map(float, fracs)),
                aopc_top=float(at.mean()), aopc_random=float(ar.mean()), aopc_bottom=float(ab.mean()),
                mean_gain=float(gain.mean()), median_gain=float(np.median(gain)),
                share_top_gt_random=float((gain > 0).mean()), wilcoxon_p=p,
                mean_drop_top_by_frac=t.mean(1).tolist(), mean_drop_random_by_frac=r.mean(1).tolist(),
                mean_drop_bottom_by_frac=b.mean(1).tolist())


# ============================================================================ final-output helpers (Step 8)
def risk_level(prob, thr):
    """Categorical level RELATIVE to the decision threshold (the model's threshold is tuned, not 0.5)."""
    if prob >= thr:
        return "HIGH"
    return "ELEVATED" if prob >= 0.5 * thr else "LOW"


def logit_of(p, eps=1e-7):
    p = float(np.clip(p, eps, 1 - eps))
    return float(np.log(p / (1 - p)))


def decision_margin(prob, thr):
    """Distance of the prediction from the decision threshold, in logit units (+ = above threshold)."""
    return logit_of(prob) - logit_of(thr)


def confidence_band(margin):
    a = abs(margin)
    return "borderline" if a < 0.5 else ("moderate" if a < 1.5 else "strong")


def support_level(ious, k=2.0, chance=CHANCE):
    """How consistently do the methods point at the same region?  ious = pairwise top-20% IoUs.
    A pair 'agrees' when its IoU >= k * chance.   >=2 agreeing pairs -> strong | 1 -> partial | 0 -> weak."""
    n = sum(1 for x in ious if np.isfinite(x) and x >= k * chance)
    return ("strong" if n >= 2 else "partial" if n == 1 else "weak"), n


def _f(x, fmt="{:.2f}", na="n/a"):
    return na if x is None or not np.isfinite(x) else fmt.format(x)


def explain_text(r):
    """Plain-language explanation for one patch from a record `r` (keys used below)."""
    lines = []
    lines.append(f"Predicted malaria risk: {r['level']}  (P(high-risk) = {_f(r['prob'])}, decision threshold {_f(r['thr'], '{:.3f}')}; "
                 f"margin {_f(r['margin'], '{:+.2f}')} logit -> {r['band']}).")
    nd, nw = r["ndvi_phi"], r["ndwi_phi"]
    dom, other = (("NDVI", "NDWI") if abs(nd) >= abs(nw) else ("NDWI", "NDVI"))
    dv = nd if dom == "NDVI" else nw
    lines.append(f"Why: starting from a base of {_f(r['base'], '{:+.2f}')} logit, NDVI contributed {_f(nd, '{:+.2f}')} and NDWI "
                 f"{_f(nw, '{:+.2f}')} (SHAP); {dom} is the dominant driver and pushes the risk {'UP' if dv > 0 else 'DOWN'}.")
    e = r.get("enrich", {})
    parts = [f"{_f(e.get(k), '{:.1f}')}x {lab}" for k, lab in (("vegetated", "vegetated pixels"), ("moist", "moist (high-NDWI) pixels"),
                                                              ("near_water", "pixels near water"), ("high_risk", "the high-risk zone"))
             if e.get(k) is not None and np.isfinite(e.get(k))]
    if parts:
        lines.append("Where: the top-20% of the fused explanation is " + ", ".join(parts) + " relative to the patch average (1.0 = chance).")
    lines.append(f"Agreement: Grad-CAM~SHAP IoU {_f(r['iou_gs'])}, Grad-CAM~CBAM {_f(r['iou_gc'])}, SHAP~CBAM {_f(r['iou_sc'])} "
                 f"(chance {CHANCE:.2f}) -> {r['support']} support ({r['n_pairs']}/3 method pairs agree).")
    if r.get("del_top") is not None and np.isfinite(r["del_top"]):
        lines.append(f"Fidelity: deleting that region lowers the logit by {_f(r['del_top'], '{:+.2f}')} vs "
                     f"{_f(r['del_rand'], '{:+.2f}')} for a random region of the same size.")
    return "\n".join(lines)
