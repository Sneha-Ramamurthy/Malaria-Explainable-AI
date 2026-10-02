"""STEP 5 (document section 3) - SHAP: Input -> Feature Contribution -> Risk Prediction.

Run:  python -m src.step5_shap                 (~5-10 min on CPU)
Env:  CACHE=1           reuse saved SHAP values (figures/metrics only)
      SHAP_SAMPLES=100  expected-gradient samples per patch      SCENE_SAMPLES=30  per scene window
      SHAP_LIB=0        skip the optional cross-check against the `shap` library

Outputs (outputs/):
  step5_shap_patches.npz          phi (N,64,64,2) logit-units, base_value, logit, prob, per-feature sums, evidence map
  step5_scene_shap_<month>.npz    stitched scene maps (evidence, NDVI phi, NDWI phi)
  step5_shap_<month>.tif          GeoTIFF (3 bands: evidence 0-1, NDVI contribution, NDWI contribution) for QGIS
  step5_shap_patches.png / step5_shap_flow.png / step5_shap_global.png / step5_shap_scene_<month>.png
  step5_metrics.json
"""
import os, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
from scipy.stats import spearmanr

from .config import OUT_DIR, ROOT
from .gradcam import load_model
from .models import build_model
from .shap_explain import (LogitGradFn, expected_gradients, completeness, channel_contributions,
                           evidence_map, shap_library_crosscheck, FEATURES)
from . import xai_utils as U

CACHE = bool(os.environ.get("CACHE"))
N_SAMPLES = int(os.environ.get("SHAP_SAMPLES", 100))
SCENE_SAMPLES = int(os.environ.get("SCENE_SAMPLES", 30))
N_BG = 100
SIGMA = 1.5

ctx = U.load_context()
X, y, split, valid, hrm, meta = ctx.X, ctx.y, ctx.split, ctx.valid, ctx.hrm, ctx.meta
THR, MONTHS, PATCH, H, W = ctx.THR, ctx.MONTHS, ctx.PATCH, ctx.H, ctx.W
te = split == 2
masks = U.env_masks(ctx)
model = load_model(ROOT / "models" / "mobilenetv2_cbam.keras")
fn = LogitGradFn(model)

# Grad-CAM from Step 4 (patch-aligned) for the cross-method comparison
g4 = np.load(OUT_DIR / "step4_gradcam_patches.npz")
cam = g4["cam"].astype("float32"); prob = g4["prob"]
LOGIT_THR = float(np.log(THR / (1 - THR)))

# ------------------------------------------------------------------ 1. SHAP for every patch
rng = np.random.RandomState(0)
tr_idx = np.where(split == 0)[0]
bg = X[rng.choice(tr_idx, N_BG, replace=False)]                        # background = training patches
pf = OUT_DIR / "step5_shap_patches.npz"
if CACHE and pf.exists():
    L = np.load(pf); phi, base, logit = L["phi"], float(L["base_value"]), L["logit"]
else:
    logit = fn.logit(X)
    phi, base = expected_gradients(fn, X, bg, n_samples=N_SAMPLES, batch=64, seed=0)
comp = completeness(phi, logit, base)
p_chk = 1 / (1 + np.exp(-logit))
print(f"SHAP computed for {len(X)} patches | base value (mean background logit) {base:.3f} "
      f"| prob-check max|p-p_step4| = {np.abs(p_chk - prob).max():.4f}")
print("completeness (sum phi vs f(x)-base):", {k: (round(v, 3) if v is not None else None) for k, v in comp.items()})

fc = channel_contributions(phi, valid)                                  # (N,2) NDVI, NDWI contributions (logit units)
resid = logit - base - fc.sum(1)
ev = evidence_map(phi, valid, SIGMA)                                    # [0,1] evidence-for-high-risk map
if not (CACHE and pf.exists()):
    np.savez_compressed(pf, phi=phi.astype(np.float32), base_value=base, logit=logit.astype(np.float32),
                        prob=p_chk.astype(np.float32), feature_sum=fc.astype(np.float32), evidence=ev.astype(np.float16))

# optional library cross-check
xc = None
if os.environ.get("SHAP_LIB", "1") != "0":
    xc = shap_library_crosscheck(fn, bg[:50], X[np.where(te)[0][:16]], nsamples=100)
    print("shap-library cross-check:", xc)

# ------------------------------------------------------------------ 2. global analysis (test patches)
idx_te = np.where(te)[0]
pred_hi = idx_te[logit[idx_te] >= LOGIT_THR]; pred_lo = idx_te[logit[idx_te] < LOGIT_THR]

def share(idx):
    a = np.abs(phi[idx] * valid[idx][..., None]).sum((1, 2))             # (n,2) per-feature |phi| mass
    s = a / np.maximum(a.sum(1, keepdims=True), 1e-12)
    return dict(share_NDVI=float(s[:, 0].mean()), share_NDWI=float(s[:, 1].mean()),
                mean_abs_NDVI=float(a[:, 0].mean()), mean_abs_NDWI=float(a[:, 1].mean()),
                mean_signed_NDVI=float(fc[idx, 0].mean()), mean_signed_NDWI=float(fc[idx, 1].mean()))
G = dict(all_test=share(idx_te), predicted_high=share(pred_hi), predicted_low=share(pred_lo))
print("\nFEATURE IMPORTANCE share of |SHAP| (test):", {k: (round(v["share_NDVI"], 2), round(v["share_NDWI"], 2)) for k, v in G.items()}, "(NDVI, NDWI)")

# where does SHAP evidence point (enrichment) - same definition as Grad-CAM in Step 4
st_shap_all, st_shap_pos = (U.focus_stats(ev, ctx, masks, idx) for idx in (idx_te, pred_hi))
S_all, S_pos = U.summarise(st_shap_all), U.summarise(st_shap_pos)
print("FOCUS of SHAP evidence (test patches predicted high-risk): enrichment >1 = above chance")
for k, v in S_pos.items():
    print(f"  {k:13s}", None if v is None else {a: round(b, 3) for a, b in v.items()})

# agreement with Grad-CAM (same patches)
rho, iou = U.patch_agreement(ev, cam, valid, idx_te)
rho_p, iou_p = U.patch_agreement(ev, cam, valid, pred_hi)
AG = dict(all_test=dict(spearman=U.describe(rho), top20_iou=U.describe(iou)),
          predicted_high=dict(spearman=U.describe(rho_p), top20_iou=U.describe(iou_p)),
          chance_top20_iou=U.CHANCE_TOP20_IOU)
print(f"SHAP vs Grad-CAM: Spearman {rho.mean():.3f} | top-20% IoU {iou.mean():.3f} (chance {U.CHANCE_TOP20_IOU:.3f})")

# sanity: randomised model (SHAP must change when the weights are meaningless)
keras.utils.set_random_seed(123)
rnd = build_model(ctx.M3["norm_mean"], ctx.M3["norm_var"], **ctx.M3["model_config"])
fn_r = LogitGradFn(rnd)
phi_r, _ = expected_gradients(fn_r, X[idx_te], bg, n_samples=max(20, N_SAMPLES // 2), batch=64, seed=1)
ev_r = np.zeros_like(ev); ev_r[idx_te] = evidence_map(phi_r, valid[idx_te], SIGMA)
rho_r = [spearmanr(a[v], b[v])[0] for a, b, v in zip(ev[idx_te], ev_r[idx_te], valid[idx_te])
         if v.sum() > 100 and a[v].std() > 0 and b[v].std() > 0]
S_rnd = U.summarise(U.focus_stats(ev_r, ctx, masks, idx_te))
sanity = dict(spearman_trained_vs_random=float(np.nanmean(rho_r)),
              high_risk_enrichment_trained=S_all["high_risk"] and S_all["high_risk"]["median"],
              high_risk_enrichment_random=S_rnd["high_risk"] and S_rnd["high_risk"]["median"])
print("SANITY randomised model:", {k: (round(v, 3) if v is not None else None) for k, v in sanity.items()})

# ------------------------------------------------------------------ 3. scene-level SHAP (sliding windows)
region = U.region_map(ctx)
scene_ev, scene_stats = {}, {}
for m in MONTHS:
    s = ctx.scene[m]; cf = OUT_DIR / f"step5_scene_shap_{m}.npz"
    if CACHE and cf.exists():
        L = np.load(cf); evs, pn, pw = (L[k].astype("float32") for k in ("evidence", "ndvi_phi", "ndwi_phi"))
    else:
        rs, cs, wins = U.scene_windows(s, ctx, stride=32)
        ph, _ = expected_gradients(fn, wins, bg, n_samples=SCENE_SAMPLES, batch=128, seed=2)
        wv = np.stack([s["valid"][r:r + PATCH, c:c + PATCH] for r, c in zip(rs, cs)])
        evw = evidence_map(ph, wv, SIGMA)
        evs = U.stitch(rs, cs, evw, ctx)
        sg = U.stitch(rs, cs, ph * wv[..., None], ctx)
        pn, pw = sg[..., 0], sg[..., 1]
        np.savez_compressed(cf, evidence=evs.astype(np.float16), ndvi_phi=pn.astype(np.float32), ndwi_phi=pw.astype(np.float32))
    scene_ev[m] = evs
    U.save_geotiff(OUT_DIR / f"step5_shap_{m}.tif", [evs, pn, pw],
                   ["SHAP evidence for high risk (0-1)", "NDVI contribution (logit)", "NDWI contribution (logit)"], ctx)
    prb = np.load(OUT_DIR / f"step3_scene_prob_{m}.npz")["prob"].astype("float32")
    ref = np.load(OUT_DIR / f"step4_scene_cam_{m}.npz")["cam"].astype("float32")
    ok = s["valid"] & ~s["water"] & (region[m] == 3)
    scene_stats[m] = U.scene_agreement(evs, ref, prb, s["hr"], ok, THR) or {}
    scene_stats[m]["mean_ndvi_phi_test"] = float(np.nanmean(pn[ok])) if ok.any() else None
    scene_stats[m]["mean_ndwi_phi_test"] = float(np.nanmean(pw[ok])) if ok.any() else None
    print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in scene_stats[m].items()})

json.dump(dict(method="expected gradients (SHAP) on pre-sigmoid logit", n_samples=N_SAMPLES, scene_samples=SCENE_SAMPLES,
               n_background=N_BG, base_value_logit=base, completeness=comp,
               feature_importance=G, focus_test_predicted_high=S_pos, focus_test_all=S_all,
               agreement_with_gradcam=AG, sanity_randomised_model=sanity, shap_library_crosscheck=xc,
               scene=scene_stats),
          open(OUT_DIR / "step5_metrics.json", "w"), indent=1)

# ------------------------------------------------------------------ 4. figures
def overlay(ax, base_img, c, cmap, vmin, vmax, title):
    ax.imshow(base_img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.imshow(np.ma.masked_less(c, 0.15), cmap="jet", alpha=0.55, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8); ax.axis("off")

sel = U.pick_examples(ctx, prob)
fig, ax = plt.subplots(6, len(sel), figsize=(2.3 * len(sel), 14))
for j, i in enumerate(sel):
    v = valid[i]; lim = max(np.percentile(np.abs(phi[i][v]), 99), 1e-9)
    ax[0, j].imshow(np.ma.masked_where(~v, X[i][..., 0]), cmap="YlGn", vmin=0, vmax=.7)
    ax[0, j].set_title(f"NDVI\np={prob[i]:.2f} true={'HIGH' if y[i] else 'low'}", fontsize=8)
    ax[1, j].imshow(np.ma.masked_where(~v, X[i][..., 1]), cmap="BrBG", vmin=-1, vmax=1); ax[1, j].set_title("NDWI", fontsize=8)
    ax[2, j].imshow(np.ma.masked_where(~v, phi[i][..., 0]), cmap="bwr", vmin=-lim, vmax=lim)
    ax[2, j].set_title(f"SHAP NDVI  sum={fc[i,0]:+.2f}", fontsize=8)
    ax[3, j].imshow(np.ma.masked_where(~v, phi[i][..., 1]), cmap="bwr", vmin=-lim, vmax=lim)
    ax[3, j].set_title(f"SHAP NDWI  sum={fc[i,1]:+.2f}", fontsize=8)
    overlay(ax[4, j], np.ma.masked_where(~v, X[i][..., 0]), ev[i], "YlGn", 0, .7, "SHAP evidence on NDVI")
    ax[5, j].imshow(hrm[i], cmap="Reds", vmin=0, vmax=1.5)
    if ev[i].max() > 0: ax[5, j].contour(ev[i], levels=[0.5], colors="cyan", linewidths=1.2)
    ax[5, j].set_title("proxy high-risk + SHAP>0.5", fontsize=8)
    for a in ax[:, j]: a.axis("off")
fig.suptitle("SHAP (red = pushes risk UP, blue = pushes risk DOWN)", y=0.995)
plt.tight_layout(); plt.savefig(OUT_DIR / "step5_shap_patches.png", dpi=75); plt.close()

# Input -> Feature Contribution -> Risk Prediction (waterfall in logit space, prob on top)
fig, ax = plt.subplots(1, len(sel), figsize=(2.6 * len(sel), 4.2), sharey=True)
for j, i in enumerate(sel):
    a = ax[j]; steps = [("base", base), ("NDVI", fc[i, 0]), ("NDWI", fc[i, 1]), ("resid", resid[i])]
    cur = 0
    a.bar(0, base, color="#888")
    cur = base
    for k, (nm, val) in enumerate(steps[1:], 1):
        a.bar(k, val, bottom=cur, color="#c0392b" if val >= 0 else "#2980b9"); cur += val
    a.bar(4, logit[i], color="k", alpha=.8)
    a.axhline(LOGIT_THR, color="g", ls="--", lw=1)
    a.set_xticks(range(5), ["base", "NDVI", "NDWI", "resid.", "f(x)"], rotation=60, fontsize=8)
    a.set_title(f"P(high)={prob[i]:.2f}\ntrue={'HIGH' if y[i] else 'low'}", fontsize=8)
ax[0].set_ylabel("logit (log-odds of high risk)")
fig.suptitle("Input (NDVI, NDWI) -> Feature contribution (SHAP) -> Risk prediction;  green dashed = decision threshold", fontsize=10)
plt.tight_layout(); plt.savefig(OUT_DIR / "step5_shap_flow.png", dpi=80); plt.close()

# global figure: importance, direction, dependence
fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
xs = np.arange(2); wd = .27
for o, (nm, k) in enumerate((("all test", "all_test"), ("pred. high", "predicted_high"), ("pred. low", "predicted_low"))):
    ax[0].bar(xs + (o - 1) * wd, [G[k]["mean_abs_NDVI"], G[k]["mean_abs_NDWI"]], wd, label=nm)
ax[0].set_xticks(xs, FEATURES); ax[0].set_ylabel("mean sum|SHAP| per patch (logit)"); ax[0].set_title("Feature importance"); ax[0].legend()
for o, (nm, k) in enumerate((("pred. high", "predicted_high"), ("pred. low", "predicted_low"))):
    ax[1].bar(xs + (o - .5) * wd * 1.5, [G[k]["mean_signed_NDVI"], G[k]["mean_signed_NDWI"]], wd * 1.4, label=nm)
ax[1].axhline(0, color="k", lw=.8); ax[1].set_xticks(xs, FEATURES); ax[1].set_title("Mean signed contribution (direction)"); ax[1].legend()
pv = valid[idx_te]; sub = np.random.RandomState(3).choice(int(pv.sum()), min(25000, int(pv.sum())), replace=False)
for c_, a_, cm in ((0, ax[2], "YlGn"), (1, ax[3], "BrBG")):
    xv = X[idx_te][..., c_][pv][sub]; pvv = phi[idx_te][..., c_][pv][sub]
    a_.scatter(xv, pvv, s=2, alpha=.15, c="k")
    bins = np.quantile(xv, np.linspace(0, 1, 21)); mid = .5 * (bins[1:] + bins[:-1]); b_i = np.clip(np.digitize(xv, bins[1:-1]), 0, 19)
    a_.plot(mid, [np.median(pvv[b_i == b]) if (b_i == b).any() else np.nan for b in range(20)], "r-", lw=2, label="binned median")
    a_.axhline(0, color="gray", lw=.8); a_.set_xlabel(f"{FEATURES[c_]} value"); a_.set_ylabel("SHAP (logit)")
    a_.set_title(f"{FEATURES[c_]} dependence"); a_.legend()
plt.tight_layout(); plt.savefig(OUT_DIR / "step5_shap_global.png", dpi=80); plt.close()

for m in MONTHS:
    s = ctx.scene[m]; sv = s["valid"]; sc = np.nan_to_num(scene_ev[m]); ok = sv & ~s["water"]
    L = np.load(OUT_DIR / f"step5_scene_shap_{m}.npz"); pn, pw = L["ndvi_phi"], L["ndwi_phi"]
    lim = float(np.nanpercentile(np.abs(np.concatenate([pn[ok & ~np.isnan(pn)], pw[ok & ~np.isnan(pw)]])), 99))
    fig, ax = plt.subplots(1, 4, figsize=(26, 6.5))
    overlay(ax[0], np.ma.masked_where(~sv, s["ndvi"].astype("float32")), sc, "YlGn", 0, .7, f"{m}  SHAP evidence over NDVI")
    ax[1].imshow(np.ma.masked_where(~ok, pn), cmap="bwr", vmin=-lim, vmax=lim); ax[1].set_title("NDVI contribution (phi, logit)")
    ax[2].imshow(np.ma.masked_where(~ok, pw), cmap="bwr", vmin=-lim, vmax=lim); ax[2].set_title("NDWI contribution (phi, logit)")
    prb = np.load(OUT_DIR / f"step3_scene_prob_{m}.npz")["prob"].astype("float32"); pred = np.nan_to_num(prb, nan=0) >= THR
    ax[3].imshow(np.ma.masked_invalid(scene_ev[m]), cmap="magma", vmin=0, vmax=1)
    ax[3].contour(pred & ok, levels=[0.5], colors="cyan", linewidths=.6); ax[3].contour(s["hr"] & sv, levels=[0.5], colors="lime", linewidths=.4)
    ax[3].set_title("SHAP evidence | cyan = predicted high-risk, green = proxy high-risk", fontsize=9)
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT_DIR / f"step5_shap_scene_{m}.png", dpi=55); plt.close()
print("done")
