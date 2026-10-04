"""STEP 7 (document section 5) - Explanation validation and fusion.

Grad-CAM (Step 4), SHAP (Step 5) and CBAM (Step 6) are compared instead of trusting any single one:
  1. spatial agreement      pairwise Spearman rho + top-20% IoU between the three explanation maps (vs chance)
  2. overlap                top-20% region vs the high-risk region (patch: proxy mask; scene: predicted high-risk) and
                            vs the NDVI / NDWI environment (vegetated / moist / near-water enrichment)
  3. stability              small Gaussian noise on the input -> does the explanation stay the same?
                            (SHAP also gets a noise floor: same input, different Monte-Carlo seed)
  4. perturbation fidelity  delete the pixels a method calls important and measure the logit drop,
                            vs deleting random / least-important pixels (deletion test, AOPC)
  5. fusion                 rank-average of the maps (+ per-pixel "how many methods agree" count), validated with 1-4

Run:  python -m src.step7_validation                (~3-6 min on CPU; needs Steps 2-6 outputs)
Env:  STAB_NOISE=0.05       noise sd as a fraction of each feature's training sd
      N_STAB=60             patches used for the stability test      STAB_SHAP_SAMPLES=50  SHAP samples there
      N_RANDOM=5            random-deletion draws                    CBAM_MAP=spatial|energy
Outputs (outputs/):
  step7_validation_patches.npz     fused maps (N,64,64), agreement count, per-patch IoU / rho / support / deletion drop (20% deleted)
  step7_scene_fusion_<month>.npz   fused scene map + agreement count + coverage
  step7_fusion_<month>.tif         GeoTIFF (2 bands: fused 0-1, #methods agreeing 0-3) for QGIS
  step7_agreement.png / step7_fidelity.png / step7_stability.png / step7_fusion_patches.png / step7_fusion_scene_<month>.png
  step7_metrics.json
"""
import os, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import OUT_DIR, ROOT
from .gradcam import load_model, GradCAM
from .shap_explain import LogitGradFn, expected_gradients, evidence_map
from .cbam_attention import CBAMExtractor, to_map
from . import xai_utils as U
from . import validation_core as V

Q = 0.2                                                   # "top-20%" region, as in Steps 4-6
SIGMA = 1.5                                               # SHAP evidence smoothing, as in Step 5
NOISE = float(os.environ.get("STAB_NOISE", 0.05))
N_STAB = int(os.environ.get("N_STAB", 60))
STAB_SHAP_SAMPLES = int(os.environ.get("STAB_SHAP_SAMPLES", 50))
N_RANDOM = int(os.environ.get("N_RANDOM", 5))
CBAM_MAP = os.environ.get("CBAM_MAP", "spatial")          # which CBAM map represents "CBAM attention"
FRACS = (0.05, 0.10, 0.20, 0.30, 0.50)
F20 = FRACS.index(0.20)
N_BG = 100

ctx = U.load_context()
X, y, split, valid, hrm, meta = ctx.X, ctx.y, ctx.split, ctx.valid, ctx.hrm, ctx.meta
THR, MONTHS, PATCH, H, W = ctx.THR, ctx.MONTHS, ctx.PATCH, ctx.H, ctx.W
N = len(X); te = split == 2; idx_te = np.where(te)[0]
LOGIT_THR = V.logit_of(THR)
masks = U.env_masks(ctx)
model = load_model(ROOT / "models" / "mobilenetv2_cbam.keras")
fn = LogitGradFn(model)

# ------------------------------------------------------------------ 0. the three explanation maps (patch aligned)
cam = np.load(OUT_DIR / "step4_gradcam_patches.npz")["cam"].astype("float32")
d5 = np.load(OUT_DIR / "step5_shap_patches.npz"); shap_ev = d5["evidence"].astype("float32"); logit = d5["logit"].astype("float32")
d6 = np.load(OUT_DIR / "step6_cbam_patches.npz")
cbam = d6["spatial_map" if CBAM_MAP == "spatial" else "energy_map"].astype("float32")
prob = 1 / (1 + np.exp(-logit))
pred_hi = idx_te[logit[idx_te] >= LOGIT_THR]
print(f"patches {N} | test {len(idx_te)} | predicted high-risk (test) {len(pred_hi)} | CBAM map = {CBAM_MAP} | noise {NOISE}x sd")
chk = np.abs(fn.logit(X) - logit).max()
print(f"consistency check: max|logit(model) - logit(Step 5)| = {chk:.2e}")

BASE = {"gradcam": cam, "shap": shap_ev, "cbam": cbam}
fused_gs = V.fuse([cam, shap_ev], valid)                       # Grad-CAM + SHAP only
fused_all = V.fuse([cam, shap_ev, cbam], valid)                # all three (the document's design)
count = V.agreement_count([cam, shap_ev, cbam], valid, Q)      # (N,64,64) 0..3 methods agree per pixel
MAPS = {**BASE, "fused_gc_shap": fused_gs, "fused_all3": fused_all}

# ------------------------------------------------------------------ 1. spatial agreement
PAIRS = [("gradcam", "shap"), ("gradcam", "cbam"), ("shap", "cbam")]
prho, piou = {}, {}
for a, b in PAIRS:
    prho[(a, b)], piou[(a, b)] = V.pair_agreement(BASE[a], BASE[b], valid, Q)

def _agree(idx):
    out = {}
    for a, b in PAIRS:
        io = piou[(a, b)][idx]; ok = io[np.isfinite(io)]
        out[f"{a}~{b}"] = dict(spearman=U.describe(prho[(a, b)][idx]), top20_iou=U.describe(io),
                              share_iou_above_chance=float((ok > V.CHANCE).mean()) if len(ok) else None,
                              share_iou_above_2x_chance=float((ok >= 2 * V.CHANCE).mean()) if len(ok) else None)
    return out
AGREE = dict(all_test=_agree(idx_te), predicted_high=_agree(pred_hi), chance_top20_iou=V.CHANCE)
print(f"\n[1] SPATIAL AGREEMENT (mean top-20% IoU / Spearman; chance IoU = {V.CHANCE:.3f})")
for sub, idx in (("all test", idx_te), ("pred. high", pred_hi)):
    for a, b in PAIRS:
        print(f"    {sub:10s} {a:>7s} ~ {b:5s}  IoU {np.nanmean(piou[(a, b)][idx]):.3f} | rho {np.nanmean(prho[(a, b)][idx]):+.3f}")

# ------------------------------------------------------------------ 2. overlap with high-risk region + environment
OVER, FOCUS = {}, {}
for k, M in MAPS.items():
    iou_h, ch_h = V.region_overlap(M, valid, hrm, Q)
    OVER[k] = {s: dict(iou_with_proxy_highrisk=U.describe(iou_h[i]), chance_iou=U.describe(ch_h[i]),
                       share_above_chance=float(np.nanmean(iou_h[i] > ch_h[i])) if np.isfinite(iou_h[i]).any() else None)
               for s, i in (("all_test", idx_te), ("predicted_high", pred_hi))}
    FOCUS[k] = dict(all_test=U.summarise(U.focus_stats(M, ctx, masks, idx_te)),
                    predicted_high=U.summarise(U.focus_stats(M, ctx, masks, pred_hi)))
print("\n[2] OVERLAP: median enrichment of the top-20% region in each class (pred. high-risk test patches; 1 = chance)")
print(f"    {'method':14s}" + "".join(f"{c:>13s}" for c in ("high_risk", "vegetated", "moist", "near_water")))
for k in MAPS:
    print(f"    {k:14s}" + "".join(f"{(FOCUS[k]['predicted_high'][c] or {'median': np.nan})['median']:13.2f}"
                                   for c in ("high_risk", "vegetated", "moist", "near_water")))

# ------------------------------------------------------------------ 3. stability under small input perturbations
others = np.setdiff1d(idx_te, pred_hi)
n_o = max(N_STAB - len(pred_hi), 0)
sel_o = others[np.linspace(0, len(others) - 1, min(n_o, len(others))).astype(int)] if n_o and len(others) else np.array([], int)
idx_s = np.concatenate([pred_hi, sel_o]).astype(int)[:N_STAB]
Xs, vs = X[idx_s], valid[idx_s]
sd = np.sqrt(np.asarray(ctx.M3["norm_var"], np.float32))
Xn = V.add_noise(Xs, vs, sd, NOISE, seed=7)

gc = GradCAM(model, "last_conv"); ex = CBAMExtractor(model)
rng = np.random.RandomState(0)
bg = X[rng.choice(np.where(split == 0)[0], N_BG, replace=False)]            # same background as Step 5
cam_c, cam_n = cam[idx_s], gc(Xn)[0]
cb_c = cbam[idx_s]
An = ex(Xn, keep_features=False); cb_n = to_map(An["sp"] if CBAM_MAP == "spatial" else An["energy"], vs)[0]
phi_a, _ = expected_gradients(fn, Xs, bg, n_samples=STAB_SHAP_SAMPLES, batch=64, seed=10)
phi_b, _ = expected_gradients(fn, Xs, bg, n_samples=STAB_SHAP_SAMPLES, batch=64, seed=11)   # noise floor: new MC seed only
phi_n, _ = expected_gradients(fn, Xn, bg, n_samples=STAB_SHAP_SAMPLES, batch=64, seed=10)
sh_a, sh_b, sh_n = (evidence_map(p, vs, SIGMA) for p in (phi_a, phi_b, phi_n))
fu_c, fu_n = V.fuse([cam_c, sh_a, cb_c], vs), V.fuse([cam_n, sh_n, cb_n], vs)

def _stab(a, b):
    r, i = V.pair_agreement(a, b, vs, Q)
    return dict(spearman=U.describe(r), top20_iou=U.describe(i))
lg_n = fn.logit(Xn); dl = lg_n - logit[idx_s]; hi_mask = logit[idx_s] >= LOGIT_THR
STAB = dict(noise_level_x_feature_sd=NOISE, n_patches=int(len(idx_s)), n_predicted_high=int(hi_mask.sum()),
            gradcam=_stab(cam_c, cam_n), shap=_stab(sh_a, sh_n), cbam=_stab(cb_c, cb_n), fused_all3=_stab(fu_c, fu_n),
            shap_noise_floor_same_input_new_seed=_stab(sh_a, sh_b),
            prediction=dict(mean_abs_logit_change=float(np.abs(dl).mean()),
                            decision_flip_rate=float(((lg_n >= LOGIT_THR) != hi_mask).mean())))
print(f"\n[3] STABILITY under {NOISE}x-sd noise on {len(idx_s)} test patches ({int(hi_mask.sum())} predicted high-risk): clean-vs-noisy top-20% IoU / rho")
for k in ("gradcam", "shap", "cbam", "fused_all3", "shap_noise_floor_same_input_new_seed"):
    print(f"    {k:38s} IoU {STAB[k]['top20_iou']['mean']:.3f} | rho {STAB[k]['spearman']['mean']:+.3f}")
print(f"    prediction: mean |d logit| {STAB['prediction']['mean_abs_logit_change']:.3f} | decision flips {STAB['prediction']['decision_flip_rate']:.1%}")

# ------------------------------------------------------------------ 4. perturbation / fidelity (deletion test)
fill = np.asarray(ctx.M3["norm_mean"], np.float32)
D = V.deletion_test(fn.logit, X[idx_te], valid[idx_te], {k: M[idx_te] for k, M in MAPS.items()}, fill, FRACS,
                    n_random=N_RANDOM, seed=0)
pos_all = np.arange(len(idx_te)); pos_hi = np.where(np.isin(idx_te, pred_hi))[0]
DEL = {}
for k in MAPS:
    DEL[k] = dict(all_test=V.summarise_deletion(D[k]["top"], D["random"], D[k]["bottom"], FRACS, pos_all),
                  predicted_high=V.summarise_deletion(D[k]["top"], D["random"], D[k]["bottom"], FRACS, pos_hi))
    drop_t = D[k]["top"][F20, pos_hi]; drop_r = D["random"][F20, pos_hi]
    DEL[k]["predicted_high"]["decision_flip_rate_at_20pct_top"] = float(((D["base"][pos_hi] - drop_t) < LOGIT_THR).mean())
    DEL[k]["predicted_high"]["decision_flip_rate_at_20pct_random"] = float(((D["base"][pos_hi] - drop_r) < LOGIT_THR).mean())
print(f"\n[4] FIDELITY (deletion test, pred. high-risk test patches n={len(pos_hi)}): logit drop averaged over {FRACS}")
print(f"    {'method':14s}{'AOPC top':>10s}{'random':>9s}{'bottom':>9s}{'gain':>8s}{'top>rand':>10s}{'p':>10s}{'flip@20%':>10s}")
for k in MAPS:
    s = DEL[k]["predicted_high"]
    print(f"    {k:14s}{s['aopc_top']:10.2f}{s['aopc_random']:9.2f}{s['aopc_bottom']:9.2f}{s['mean_gain']:8.2f}"
          f"{s['share_top_gt_random']:10.0%}{s['wilcoxon_p']:10.1e}{s['decision_flip_rate_at_20pct_top']:10.0%}")

# ------------------------------------------------------------------ 5. per-patch support + fusion summary
support = np.array(["n/a"] * N, dtype="<U8"); n_pairs = np.zeros(N, np.uint8)
for i in range(N):
    support[i], n_pairs[i] = V.support_level([piou[p][i] for p in PAIRS])
del_top20 = {k: np.full(N, np.nan, np.float32) for k in ("fused_all3", "fused_gc_shap")}; del_rand20 = np.full(N, np.nan, np.float32)
for k in del_top20: del_top20[k][idx_te] = D[k]["top"][F20]
del_rand20[idx_te] = D["random"][F20]
sup_counts = {s: int((support[idx_te] == s).sum()) for s in ("strong", "partial", "weak")}
sup_counts_hi = {s: int((support[pred_hi] == s).sum()) for s in ("strong", "partial", "weak")}
print(f"\n[5] FUSION: explanation support per test patch {sup_counts} | predicted-high only {sup_counts_hi}")

cand = {k: DEL[k]["predicted_high"]["mean_gain"] for k in ("gradcam", "shap", "cbam", "fused_gc_shap", "fused_all3")}
best = max(cand, key=cand.get)
print("    deletion-gain ranking (pred. high):", ", ".join(f"{k} {v:+.2f}" for k, v in sorted(cand.items(), key=lambda t: -t[1])))
np.savez_compressed(OUT_DIR / "step7_validation_patches.npz", fused_all3=fused_all.astype(np.float16),
                    fused_gc_shap=fused_gs.astype(np.float16), count=count, support=support, n_pairs=n_pairs,
                    iou=np.stack([piou[p] for p in PAIRS]).astype(np.float32), rho=np.stack([prho[p] for p in PAIRS]).astype(np.float32),
                    pairs=np.array([f"{a}~{b}" for a, b in PAIRS]), del_top20_all3=del_top20["fused_all3"],
                    del_top20_gc_shap=del_top20["fused_gc_shap"], del_rand20=del_rand20)

# ------------------------------------------------------------------ 6. scene-level fusion (unseen test blocks)
region = U.region_map(ctx)
SCENE, scene_out = {}, {}
for m in MONTHS:
    s = ctx.scene[m]; sv = s["valid"]
    gcs = np.load(OUT_DIR / f"step4_scene_cam_{m}.npz")["cam"].astype("float32")
    shs = np.load(OUT_DIR / f"step5_scene_shap_{m}.npz")["evidence"].astype("float32")
    L6 = np.load(OUT_DIR / f"step6_scene_cbam_{m}.npz"); cbs = L6["spatial" if CBAM_MAP == "spatial" else "energy"].astype("float32")
    cover = sv & ~s["water"] & np.isfinite(gcs) & np.isfinite(shs) & np.isfinite(cbs)
    fsc = np.where(cover, V.fuse([gcs, shs, cbs], cover), np.nan).astype(np.float32)
    cnt = np.sum([V.top_scene(a, cover, Q) for a in (gcs, shs, cbs)], axis=0).astype(np.uint8)
    cnt_f = np.where(cover, cnt, np.nan).astype(np.float32)
    scene_out[m] = dict(fused=fsc, count=cnt_f, cover=cover)
    np.savez_compressed(OUT_DIR / f"step7_scene_fusion_{m}.npz", fused=fsc.astype(np.float16), count=cnt, cover=cover)
    U.save_geotiff(OUT_DIR / f"step7_fusion_{m}.tif", [fsc, cnt_f],
                   ["Fused explanation (rank-mean, 0-1)", "Methods agreeing, top-20% (0-3)"], ctx)
    prb = np.load(OUT_DIR / f"step3_scene_prob_{m}.npz")["prob"].astype("float32")
    pred = np.nan_to_num(prb, nan=0) >= THR
    ok = cover & (region[m] == 3)
    st = {}
    for name, a in (("gradcam", gcs), ("shap", shs), ("cbam", cbs), ("fused_all3", fsc)):
        r_ = U.scene_agreement(a, gcs if name != "gradcam" else None, prb, s["hr"], ok, THR) or {}
        t_ = V.top_scene(a, ok, Q)
        if t_.sum() and (pred & ok).sum():
            r_["top20_iou_with_predicted_highrisk"] = float((t_ & pred).sum() / max((t_ | (pred & ok)).sum(), 1))
            r_["chance_iou_with_predicted_highrisk"] = V.chance_iou(int(t_.sum()), int((pred & ok).sum()), int(ok.sum()))
        st[name] = r_
    cons = ok & (cnt >= 2)
    st["consensus_2of3"] = dict(share_of_test_land=float(cons[ok].mean()) if ok.any() else None,
                                inside_pred_highrisk=float(pred[cons].mean()) if cons.any() else None,
                                inside_proxy_highrisk=float(s["hr"][cons].mean()) if cons.any() else None,
                                pred_highrisk_share_of_test_land=float(pred[ok].mean()) if ok.any() else None,
                                predicted_high_covered_by_consensus=float(cons[pred & ok].mean()) if (pred & ok).any() else None)
    SCENE[m] = st
    print(f"[6] {m} fused: top20 inside predicted high-risk {st['fused_all3'].get('top20_inside_pred_highrisk', float('nan')):.3f} "
          f"(base rate {st['fused_all3'].get('pred_highrisk_share_of_test_land', float('nan')):.3f}) | "
          f"consensus(>=2 methods) {st['consensus_2of3']['share_of_test_land']:.3f} of test land, "
          f"{(st['consensus_2of3']['inside_pred_highrisk'] or 0):.3f} inside predicted high-risk")

# ------------------------------------------------------------------ metrics json
def _j(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))
summary = dict(
    agreement_gradcam_shap_above_chance=bool(np.nanmean(piou[("gradcam", "shap")][pred_hi]) > V.CHANCE),
    agreement_cbam_with_others_above_chance=bool(max(np.nanmean(piou[("gradcam", "cbam")][pred_hi]),
                                                     np.nanmean(piou[("shap", "cbam")][pred_hi])) > V.CHANCE),
    methods_with_positive_fidelity_gain={k: bool(DEL[k]["predicted_high"]["mean_gain"] > 0 and DEL[k]["predicted_high"]["wilcoxon_p"] < 0.05)
                                         for k in MAPS},
    best_by_deletion_gain_predicted_high=best,
    shap_stability_above_noise_floor=bool(STAB["shap"]["top20_iou"]["mean"] >= 0.9 * STAB["shap_noise_floor_same_input_new_seed"]["top20_iou"]["mean"]))
json.dump(dict(settings=dict(top_fraction=Q, stab_noise=NOISE, n_stab=int(len(idx_s)), stab_shap_samples=STAB_SHAP_SAMPLES,
                             n_random=N_RANDOM, cbam_map=CBAM_MAP, fractions=list(FRACS), threshold=THR),
               spatial_agreement=AGREE, overlap_with_proxy_highrisk=OVER, environment_focus=FOCUS, stability=STAB,
               fidelity_deletion=DEL, support_counts_test=sup_counts, support_counts_predicted_high=sup_counts_hi,
               scene=SCENE, summary=summary, recommended_fusion=("fused_gc_shap" if best == "fused_gc_shap" else "fused_all3")),
          open(OUT_DIR / "step7_metrics.json", "w"), indent=1, default=_j)

# ------------------------------------------------------------------ figures
COL = dict(gradcam="#e67e22", shap="#c0392b", cbam="#2980b9", fused_gc_shap="#8e44ad", fused_all3="#27ae60")

# (a) agreement
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
labs = [f"{a}\n~ {b}" for a, b in PAIRS]
for o, (nm, idx) in enumerate((("all test", idx_te), ("pred. high-risk", pred_hi))):
    pos = np.arange(3) * 3 + o
    for axx, arr, ttl in ((ax[0], piou, "top-20% IoU"), (ax[1], prho, "Spearman rho")):
        data = [arr[p][idx][np.isfinite(arr[p][idx])] for p in PAIRS]
        bp = axx.boxplot(data, positions=pos, widths=.8, patch_artist=True, showfliers=False)
        for b_ in bp["boxes"]: b_.set_facecolor(["#bbb", "#f5b041"][o])
        axx.set_title(f"Pairwise agreement: {ttl}")
ax[0].axhline(V.CHANCE, color="k", ls="--"); ax[0].text(-0.5, V.CHANCE + .01, "chance", fontsize=8)
ax[1].axhline(0, color="k", ls="--")
for a_ in ax: a_.set_xticks(np.arange(3) * 3 + .5, labs)
ax[0].legend([plt.Rectangle((0, 0), 1, 1, fc=c) for c in ("#bbb", "#f5b041")], ["all test", "pred. high-risk"], fontsize=8)
plt.tight_layout(); plt.savefig(OUT_DIR / "step7_agreement.png", dpi=80); plt.close()

# (b) fidelity curves
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for axx, (ttl, key) in zip(ax, (("all test patches", "all_test"), ("predicted high-risk test patches", "predicted_high"))):
    for k in MAPS:
        axx.plot(np.array(FRACS) * 100, DEL[k][key]["mean_drop_top_by_frac"], "-o", color=COL[k], label=k)
    axx.plot(np.array(FRACS) * 100, DEL["gradcam"][key]["mean_drop_random_by_frac"], "k--", label="random pixels")
    axx.plot(np.array(FRACS) * 100, np.mean([DEL[k][key]["mean_drop_bottom_by_frac"] for k in BASE], 0), ":", color="gray",
             label="least-important pixels (mean of methods)")
    axx.set_xlabel("% of valid pixels deleted"); axx.set_ylabel("mean logit drop"); axx.set_title(f"Deletion test - {ttl}"); axx.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT_DIR / "step7_fidelity.png", dpi=80); plt.close()

# (c) stability
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ks = ["gradcam", "shap", "cbam", "fused_all3"]
for axx, met, ttl in ((ax[0], "top20_iou", "top-20% IoU (clean vs noisy)"), (ax[1], "spearman", "Spearman rho (clean vs noisy)")):
    axx.bar(ks, [STAB[k][met]["mean"] for k in ks], color=[COL[k] for k in ks])
    axx.axhline(STAB["shap_noise_floor_same_input_new_seed"][met]["mean"], color="k", ls="--")
    axx.text(3.45, STAB["shap_noise_floor_same_input_new_seed"][met]["mean"], " SHAP MC noise floor", fontsize=7, va="bottom", ha="right")
    axx.set_title(ttl); axx.set_ylim(0, 1.05)
plt.tight_layout(); plt.savefig(OUT_DIR / "step7_stability.png", dpi=80); plt.close()

# (d) fused examples
def overlay(ax_, base_img, c, cmap, vmin, vmax, title):
    ax_.imshow(base_img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_.imshow(np.ma.masked_less(c, 0.15), cmap="jet", alpha=0.55, vmin=0, vmax=1)
    ax_.set_title(title, fontsize=8); ax_.axis("off")
sel = U.pick_examples(ctx, prob)
fig, ax = plt.subplots(6, len(sel), figsize=(2.3 * len(sel), 14))
for j, i in enumerate(sel):
    v = valid[i]; nd = np.ma.masked_where(~v, X[i][..., 0])
    ax[0, j].imshow(nd, cmap="YlGn", vmin=0, vmax=.7); ax[0, j].set_title(f"NDVI\np={prob[i]:.2f} true={'HIGH' if y[i] else 'low'}", fontsize=8)
    overlay(ax[1, j], nd, cam[i], "YlGn", 0, .7, "Grad-CAM")
    overlay(ax[2, j], nd, shap_ev[i], "YlGn", 0, .7, "SHAP evidence")
    overlay(ax[3, j], nd, cbam[i], "YlGn", 0, .7, "CBAM attention")
    overlay(ax[4, j], nd, fused_all[i], "YlGn", 0, .7, f"Fused | support: {support[i]}")
    ax[5, j].imshow(np.ma.masked_where(~v, count[i]), cmap="viridis", vmin=0, vmax=3)
    ax[5, j].contour(hrm[i], levels=[0.5], colors="red", linewidths=1.0)
    ax[5, j].set_title("# methods agreeing (0-3)\nred = proxy high-risk", fontsize=8)
    for a_ in ax[:, j]: a_.axis("off")
plt.tight_layout(); plt.savefig(OUT_DIR / "step7_fusion_patches.png", dpi=75); plt.close()

# (e) scene fusion
for m in MONTHS:
    s = ctx.scene[m]; sv = s["valid"]; so = scene_out[m]
    prb = np.load(OUT_DIR / f"step3_scene_prob_{m}.npz")["prob"].astype("float32"); pred = np.nan_to_num(prb, nan=0) >= THR
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.5))
    overlay(ax[0], np.ma.masked_where(~sv, s["ndvi"].astype("float32")), np.nan_to_num(so["fused"]), "YlGn", 0, .7, f"{m}  fused explanation over NDVI")
    ax[1].imshow(np.ma.masked_invalid(so["fused"]), cmap="magma", vmin=0, vmax=1)
    ax[1].contour(pred & so["cover"], levels=[0.5], colors="cyan", linewidths=.6)
    ax[1].set_title("Fused explanation | cyan = predicted high-risk", fontsize=9)
    im = ax[2].imshow(np.ma.masked_invalid(so["count"]), cmap="viridis", vmin=0, vmax=3); plt.colorbar(im, ax=ax[2], fraction=.03)
    ax[2].set_title("# methods whose top-20% contains the pixel (0-3)", fontsize=9)
    for a_ in ax: a_.axis("off")
    plt.tight_layout(); plt.savefig(OUT_DIR / f"step7_fusion_scene_{m}.png", dpi=55); plt.close()
print("done")