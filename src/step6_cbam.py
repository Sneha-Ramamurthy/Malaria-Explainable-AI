"""STEP 6 (document section 4) - CBAM attention analysis and comparison with Grad-CAM.

Run:  python -m src.step6_cbam                 (~2 min on CPU)
Env:  CACHE=1  reuse saved scene attention maps

Outputs (outputs/):
  step6_cbam_patches.npz            channel attention (N,C), spatial attention raw + [0,1] maps (N,64,64), energy map
  step6_scene_cbam_<month>.npz      stitched scene spatial-attention / energy maps
  step6_cbam_<month>.tif            GeoTIFF (2 bands: spatial attention 0-1, CBAM-output energy 0-1) for QGIS
  step6_cbam_patches.png / step6_cbam_channels.png / step6_cbam_scene_<month>.png / step6_cbam_focus.png
  step6_metrics.json
"""
import os, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
from scipy.stats import mannwhitneyu, spearmanr

from .config import OUT_DIR, ROOT
from .gradcam import load_model
from .models import build_model
from .cbam_attention import CBAMExtractor, to_map, channel_input_affinity, attention_weighted_share
from . import xai_utils as U

CACHE = bool(os.environ.get("CACHE"))
ctx = U.load_context()
X, y, split, valid, hrm, meta = ctx.X, ctx.y, ctx.split, ctx.valid, ctx.hrm, ctx.meta
THR, MONTHS, PATCH, H, W = ctx.THR, ctx.MONTHS, ctx.PATCH, ctx.H, ctx.W
te = split == 2; idx_te = np.where(te)[0]
masks = U.env_masks(ctx)
model = load_model(ROOT / "models" / "mobilenetv2_cbam.keras")
ex = CBAMExtractor(model)

g4 = np.load(OUT_DIR / "step4_gradcam_patches.npz")
cam = g4["cam"].astype("float32"); prob_s4 = g4["prob"]

# ------------------------------------------------------------------ 1. extract attention for every patch
A = ex(X)
prob = A["prob"]; C = A["ch"].shape[1]; h, w = A["sp_low"].shape[1:]
print(f"CBAM extracted for {len(X)} patches | channels C={C} | spatial attention grid {h}x{w} "
      f"| prob-check max|p-p_step4| = {np.abs(prob - prob_s4).max():.4f}")
sp, sp_range = to_map(A["sp"], valid)            # [0,1] spatial-attention map + raw dynamic range
en, _ = to_map(A["energy"], valid)               # [0,1] CBAM-output energy map
np.savez_compressed(OUT_DIR / "step6_cbam_patches.npz", channel_att=A["ch"].astype(np.float32),
                    spatial_raw=A["sp"].astype(np.float32), spatial_map=sp.astype(np.float16),
                    energy_map=en.astype(np.float16), prob=prob.astype(np.float32))
pred_hi = idx_te[prob[idx_te] >= THR]; pred_lo = idx_te[prob[idx_te] < THR]
raw = A["sp"]
print(f"spatial attention raw values: mean {raw.mean():.3f}, min {raw.min():.3f}, max {raw.max():.3f} | "
      f"mean per-patch dynamic range {sp_range.mean():.3f}  (small => attention is nearly flat; [0,1] maps exaggerate it)")

# ------------------------------------------------------------------ 2. channel attention analysis
ch = A["ch"]
aff, rsig = channel_input_affinity(A["feat"], X, valid, idx_te)           # (C,2)
share = attention_weighted_share(ch, aff)                                  # (N,2)
mw = []
for c in range(C):
    a, b = ch[pred_hi, c], ch[pred_lo, c]
    try: p = float(mannwhitneyu(a, b).pvalue)
    except ValueError: p = float("nan")
    mw.append(dict(channel=c, mean_high=float(a.mean()), mean_low=float(b.mean()), diff=float(a.mean() - b.mean()),
                   p_value=p, aff_NDVI=float(aff[c, 0]), aff_NDWI=float(aff[c, 1])))
bonf = 0.05 / C
n_sig = sum(1 for r in mw if r["p_value"] < bonf)
ch_sd = ch.std(1).mean()
CH = dict(n_channels=C, mean_attention=float(ch.mean()), mean_within_patch_std=float(ch_sd),
          note="within-patch std ~0 means channel attention barely discriminates between channels",
          n_channels_differing_high_vs_low_bonferroni=n_sig,
          top_channels_by_abs_diff=sorted(mw, key=lambda r: -abs(r["diff"]))[:5],
          attention_share_on_NDVI_tracking_channels=dict(pred_high=float(share[pred_hi, 0].mean()),
                                                         pred_low=float(share[pred_lo, 0].mean())),
          attention_share_on_NDWI_tracking_channels=dict(pred_high=float(share[pred_hi, 1].mean()),
                                                         pred_low=float(share[pred_lo, 1].mean())),
          channel_affinity_mean=dict(NDVI=float(aff[:, 0].mean()), NDWI=float(aff[:, 1].mean())))
print(f"\nCHANNEL ATTENTION: mean {ch.mean():.3f} | within-patch std {ch_sd:.3f} | channels differing high vs low (Bonferroni): {n_sig}/{C}")
print(f"  attention share on NDVI-tracking channels: high {share[pred_hi,0].mean():.2f} vs low {share[pred_lo,0].mean():.2f}")

# ------------------------------------------------------------------ 3. spatial attention vs Grad-CAM, labels, environment
res = {}
for name, M in (("spatial_attention", sp), ("cbam_energy", en)):
    rho, iou = U.patch_agreement(M, cam, valid, idx_te)
    rho_p, iou_p = U.patch_agreement(M, cam, valid, pred_hi)
    S_all = U.summarise(U.focus_stats(M, ctx, masks, idx_te))
    S_pos = U.summarise(U.focus_stats(M, ctx, masks, pred_hi))
    res[name] = dict(vs_gradcam_all_test=dict(spearman=U.describe(rho), top20_iou=U.describe(iou)),
                     vs_gradcam_predicted_high=dict(spearman=U.describe(rho_p), top20_iou=U.describe(iou_p)),
                     focus_all_test=S_all, focus_predicted_high=S_pos)
    print(f"{name:18s} vs Grad-CAM: Spearman {rho.mean():+.3f} | top-20% IoU {iou.mean():.3f} (chance {U.CHANCE_TOP20_IOU:.3f})")
    print("   focus (pred. high-risk test patches) enrichment medians:",
          {k: (round(v["median"], 2) if v else None) for k, v in S_pos.items()})

# sanity: randomised model
keras.utils.set_random_seed(123)
rnd = build_model(ctx.M3["norm_mean"], ctx.M3["norm_var"], **ctx.M3["model_config"])
A_r = CBAMExtractor(rnd)(X[idx_te], keep_features=False)
sp_r, rng_r = to_map(A_r["sp"], valid[idx_te])
rho_r = [spearmanr(a[v], b[v])[0] for a, b, v in zip(sp[idx_te], sp_r, valid[idx_te])
         if v.sum() > 100 and a[v].std() > 0 and b[v].std() > 0]
sp_r_full = np.zeros_like(sp); sp_r_full[idx_te] = sp_r
S_rnd = U.summarise(U.focus_stats(sp_r_full, ctx, masks, idx_te))
sanity = dict(spearman_trained_vs_random=float(np.nanmean(rho_r)),
              dynamic_range_trained=float(sp_range[idx_te].mean()), dynamic_range_random=float(rng_r.mean()),
              high_risk_enrichment_trained=res["spatial_attention"]["focus_all_test"]["high_risk"]["median"]
              if res["spatial_attention"]["focus_all_test"]["high_risk"] else None,
              high_risk_enrichment_random=S_rnd["high_risk"]["median"] if S_rnd["high_risk"] else None)
print("SANITY randomised model:", {k: (round(v, 3) if v is not None else None) for k, v in sanity.items()})

# ------------------------------------------------------------------ 4. scene-level spatial attention
region = U.region_map(ctx)
scene_sp, scene_stats = {}, {}
for m in MONTHS:
    s = ctx.scene[m]; cf = OUT_DIR / f"step6_scene_cbam_{m}.npz"
    if CACHE and cf.exists():
        L = np.load(cf); ssp, sen = L["spatial"].astype("float32"), L["energy"].astype("float32")
    else:
        rs, cs, wins = U.scene_windows(s, ctx, stride=32)
        Aw = ex(wins, batch=256, keep_features=False)
        wv = np.stack([s["valid"][r:r + PATCH, c:c + PATCH] for r, c in zip(rs, cs)])
        ssp = U.stitch(rs, cs, to_map(Aw["sp"], wv)[0], ctx)
        sen = U.stitch(rs, cs, to_map(Aw["energy"], wv)[0], ctx)
        np.savez_compressed(cf, spatial=ssp.astype(np.float16), energy=sen.astype(np.float16))
    scene_sp[m] = ssp
    U.save_geotiff(OUT_DIR / f"step6_cbam_{m}.tif", [ssp, sen],
                   ["CBAM spatial attention (0-1)", "CBAM output energy (0-1)"], ctx)
    prb = np.load(OUT_DIR / f"step3_scene_prob_{m}.npz")["prob"].astype("float32")
    ref = np.load(OUT_DIR / f"step4_scene_cam_{m}.npz")["cam"].astype("float32")
    ok = s["valid"] & ~s["water"] & (region[m] == 3)
    scene_stats[m] = dict(spatial_attention=U.scene_agreement(ssp, ref, prb, s["hr"], ok, THR) or {},
                          cbam_energy=U.scene_agreement(sen, ref, prb, s["hr"], ok, THR) or {})
    print(m, "spatial attention:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in scene_stats[m]["spatial_attention"].items()})

json.dump(dict(layers=dict(channel="cbam_channel_att", spatial="cbam_spatial_att", output="cbam_out"),
               feature_grid=[int(h), int(w)], spatial_attention_raw=dict(mean=float(raw.mean()), min=float(raw.min()),
               max=float(raw.max()), mean_patch_dynamic_range=float(sp_range.mean())),
               channel=CH, spatial_vs_gradcam=res, sanity_randomised_model=sanity, scene=scene_stats,
               chance_top20_iou=U.CHANCE_TOP20_IOU),
          open(OUT_DIR / "step6_metrics.json", "w"), indent=1)

# ------------------------------------------------------------------ 5. figures
def overlay(ax, base_img, c, cmap, vmin, vmax, title):
    ax.imshow(base_img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.imshow(np.ma.masked_less(c, 0.15), cmap="jet", alpha=0.55, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8); ax.axis("off")

sel = U.pick_examples(ctx, prob)
fig, ax = plt.subplots(6, len(sel), figsize=(2.3 * len(sel), 14))
for j, i in enumerate(sel):
    v = valid[i]; nd = np.ma.masked_where(~v, X[i][..., 0])
    ax[0, j].imshow(nd, cmap="YlGn", vmin=0, vmax=.7); ax[0, j].set_title(f"NDVI\np={prob[i]:.2f} true={'HIGH' if y[i] else 'low'}", fontsize=8)
    ax[1, j].imshow(np.ma.masked_where(~v, X[i][..., 1]), cmap="BrBG", vmin=-1, vmax=1); ax[1, j].set_title("NDWI", fontsize=8)
    overlay(ax[2, j], nd, sp[i], "YlGn", 0, .7, f"CBAM spatial att.\n(raw range {sp_range[i]:.2f})")
    overlay(ax[3, j], nd, en[i], "YlGn", 0, .7, "CBAM output energy")
    overlay(ax[4, j], nd, cam[i], "YlGn", 0, .7, "Grad-CAM (Step 4)")
    ax[5, j].imshow(hrm[i], cmap="Reds", vmin=0, vmax=1.5)
    if sp[i].max() > 0: ax[5, j].contour(sp[i], levels=[0.5], colors="cyan", linewidths=1.2)
    if cam[i].max() > 0: ax[5, j].contour(cam[i], levels=[0.5], colors="white", linewidths=1.0)
    ax[5, j].set_title("proxy mask | cyan=CBAM, white=CAM", fontsize=8)
    for a in ax[:, j]: a.axis("off")
plt.tight_layout(); plt.savefig(OUT_DIR / "step6_cbam_patches.png", dpi=75); plt.close()

order = idx_te[np.argsort(-prob[idx_te])]
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
im = ax[0].imshow(ch[order], aspect="auto", cmap="viridis"); ax[0].set_xlabel("channel"); ax[0].set_ylabel("test patches (sorted by P(high), top = highest)")
ax[0].set_title("Channel attention per patch"); plt.colorbar(im, ax=ax[0])
xs = np.arange(C); ax[1].bar(xs - .2, ch[pred_hi].mean(0), .4, label="predicted high"); ax[1].bar(xs + .2, ch[pred_lo].mean(0), .4, label="predicted low")
ax[1].set_xlabel("channel"); ax[1].set_ylabel("mean attention"); ax[1].set_title("Mean channel attention by predicted class"); ax[1].legend()
sc_ = ax[2].scatter(aff[:, 0], aff[:, 1], s=ch.mean(0) * 300, c=[r["diff"] for r in mw], cmap="coolwarm"); plt.colorbar(sc_, ax=ax[2], label="attention diff (high - low)")
for c in range(C): ax[2].annotate(str(c), (aff[c, 0], aff[c, 1]), fontsize=7)
ax[2].set_xlabel("|r| with NDVI"); ax[2].set_ylabel("|r| with NDWI"); ax[2].set_title("What each channel tracks (size = mean attention)")
plt.tight_layout(); plt.savefig(OUT_DIR / "step6_cbam_channels.png", dpi=80); plt.close()

names = ["high_risk", "near_water", "moist", "vegetated"]; lab = ["proxy\nhigh-risk", "near\nwater", "moist", "vegetated"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5)); xs = np.arange(4); wd = .27
for o, (nm, k) in enumerate((("spatial attention", "spatial_attention"), ("CBAM energy", "cbam_energy"))):
    ax[0].bar(xs + (o - .5) * wd * 1.2, [(res[k]["focus_all_test"][n] or {"median": np.nan})["median"] for n in names], wd, label=nm)
ax[0].bar(xs + 1.1 * wd * 1.2, [(S_rnd[n] or {"median": np.nan})["median"] for n in names], wd, label="randomised model (spatial att.)", alpha=.6)
ax[0].axhline(1, color="k", ls="--"); ax[0].set_xticks(xs, lab); ax[0].set_ylabel("median enrichment (1 = chance)")
ax[0].set_title("Where does the top-20% of CBAM attention fall? (test patches)"); ax[0].legend(fontsize=8)
rho_, iou_ = U.patch_agreement(sp, cam, valid, idx_te); ax[1].hist(rho_, bins=25); ax[1].axvline(0, color="k", ls="--")
ax[1].set_title(f"Per-patch Spearman: CBAM spatial att. vs Grad-CAM (mean {rho_.mean():+.2f})")
plt.tight_layout(); plt.savefig(OUT_DIR / "step6_cbam_focus.png", dpi=80); plt.close()

for m in MONTHS:
    s = ctx.scene[m]; sv = s["valid"]; ssp = scene_sp[m]; ref = np.load(OUT_DIR / f"step4_scene_cam_{m}.npz")["cam"].astype("float32")
    st = scene_stats[m]["spatial_attention"]
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.5))
    overlay(ax[0], np.ma.masked_where(~sv, s["ndvi"].astype("float32")), np.nan_to_num(ssp), "YlGn", 0, .7, f"{m}  CBAM spatial attention over NDVI")
    ax[1].imshow(np.ma.masked_invalid(ref), cmap="magma", vmin=0, vmax=1); ax[1].set_title("Grad-CAM (Step 4)")
    ax[2].imshow(np.ma.masked_invalid(ssp), cmap="magma", vmin=0, vmax=1)
    ax[2].set_title("CBAM spatial attention" + (f"  | test-block Spearman vs Grad-CAM {st['spearman_vs_gradcam']:+.2f}" if "spearman_vs_gradcam" in st else ""), fontsize=9)
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT_DIR / f"step6_cbam_scene_{m}.png", dpi=55); plt.close()
print("done")
