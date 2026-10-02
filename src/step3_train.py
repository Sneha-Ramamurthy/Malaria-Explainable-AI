"""STEP 3 - train the MobileNetV2+CBAM patch classifier, evaluate, and build the high-risk map.
Run:  python -m src.step3_train
"""
import os, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf, keras
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve)
from sklearn.linear_model import LogisticRegression
from .config import OUT_DIR, ROOT
from .models import build_model

keras.utils.set_random_seed(42)
MODEL_DIR = ROOT / "models"; MODEL_DIR.mkdir(exist_ok=True)
meta_j = json.load(open(OUT_DIR / "step2_meta.json"))
MONTHS, PATCH = meta_j["months"], meta_j["patch"]

d = np.load(OUT_DIR / "step2_patches.npz")
X, y, split, valid, mask, meta = (d[k] for k in ("X", "y", "split", "valid", "mask", "meta"))
X = X.astype("float32"); tr, va, te = (split == k for k in (0, 1, 2))
print(f"train {tr.sum()} | val {va.sum()} | test {te.sum()}")

# ---- normalisation stats from valid training pixels ----
v = valid[tr].astype(bool)
mean = [float(X[tr][..., c][v].mean()) for c in (0, 1)]
var = [float(X[tr][..., c][v].var()) for c in (0, 1)]
CFG = dict(alpha=0.35, cut="block_5_add", dropout=0.5, up=128)   # chosen on validation AUC (see README)
model = build_model(mean, var, **CFG)
print("params:", f"{model.count_params():,}")

# ---- data pipeline with label-preserving augmentation (flips / 90-degree rotations) ----
def aug(x, t):
    x = tf.image.random_flip_left_right(x); x = tf.image.random_flip_up_down(x)
    x = tf.image.rot90(x, tf.random.uniform([], 0, 4, tf.int32))
    x = x + tf.random.normal(tf.shape(x), 0, 0.02) + tf.random.normal([1, 1, 2], 0, 0.03)  # sensor noise + small offset
    return x, t
ds_tr = tf.data.Dataset.from_tensor_slices((X[tr], y[tr].astype("float32"))).shuffle(1000, seed=1).map(aug).batch(32)
ds_va = tf.data.Dataset.from_tensor_slices((X[va], y[va].astype("float32"))).batch(64)
pos = y[tr].mean(); cw = {0: 0.5 / (1 - pos), 1: 0.5 / pos}
model.compile(keras.optimizers.AdamW(5e-4, weight_decay=1e-2), "binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
t0 = time.time()
hist = model.fit(ds_tr, validation_data=ds_va, epochs=30, class_weight=cw, verbose=0, callbacks=[
    keras.callbacks.EarlyStopping("val_auc", mode="max", patience=10, restore_best_weights=True)])
ep = len(hist.history["loss"])
print(f"trained {ep} epochs in {time.time()-t0:.0f}s | best val AUC {max(hist.history['val_auc']):.3f}")
model.save(MODEL_DIR / "mobilenetv2_cbam.keras")

# ---- patch-level evaluation ----
p_va = model.predict(X[va], verbose=0).ravel(); p_te = model.predict(X[te], verbose=0).ravel()
# Threshold: labels were defined as the top 30% by high-risk fraction, so flag the top 30% of predictions.
# (Label-free "prior matching"; a val-F1 threshold was unreliable because val is 53% positive vs 26% train / 19% test.)
pall = model.predict(X, batch_size=128, verbose=0).ravel()
thr = float(np.quantile(pall, 0.70))

def patch_metrics(yt, p, t):
    yh = (p >= t).astype(int)
    return dict(precision=precision_score(yt, yh, zero_division=0), recall=recall_score(yt, yh),
                f1=f1_score(yt, yh), auc=roc_auc_score(yt, p), accuracy=float((yh == yt).mean()))
m_te = patch_metrics(y[te], p_te, thr); m_va = patch_metrics(y[va], p_va, thr)
# simple baseline: logistic regression on patch summary statistics (context for the CNN's score)
def feats(Xp, Vp):
    f = []
    for a, vv in zip(Xp, Vp.astype(bool)):
        n, w = a[..., 0][vv], a[..., 1][vv]
        f.append([n.mean(), n.std(), w.mean(), w.std(), np.percentile(w, 95)])
    return np.array(f)
lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(feats(X[tr], valid[tr]), y[tr])
pb = lr.predict_proba(feats(X[te], valid[te]))[:, 1]
m_lr = patch_metrics(y[te], pb, 0.5)
print("\nPATCH-LEVEL (test):  CNN+CBAM  ", {k: round(float(v), 3) for k, v in m_te.items()}, f"| thr={thr:.2f}")
print("PATCH-LEVEL (test):  baseline LR", {k: round(float(v), 3) for k, v in m_lr.items()})

# ---- scene-level probability maps (sliding window) + pixel-level IoU on TEST patches only ----
H, W = meta_j["shape"]
scene_prob = {}
for mi, m in enumerate(MONTHS):
    s = np.load(OUT_DIR / f"step2_scene_{m}.npz")
    img = np.stack([s["ndvi"], s["ndwi"]], -1).astype("float32") * s["valid"][..., None]
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    rs, cs, wins = [], [], []
    for r in range(0, H - PATCH + 1, 16):
        for c in range(0, W - PATCH + 1, 16):
            if s["valid"][r:r + PATCH, c:c + PATCH].mean() >= 0.6:
                rs.append(r); cs.append(c); wins.append(img[r:r + PATCH, c:c + PATCH])
    pr = model.predict(np.stack(wins), batch_size=128, verbose=0).ravel()
    for r, c, p in zip(rs, cs, pr):
        acc[r:r + PATCH, c:c + PATCH] += p; cnt[r:r + PATCH, c:c + PATCH] += 1
    prob = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
    scene_prob[m] = prob
    np.savez_compressed(OUT_DIR / f"step3_scene_prob_{m}.npz", prob=prob.astype(np.float16))

# paint test patches' own predictions -> honest pixel-level IoU on unseen blocks
tp_acc = {m: np.zeros((H, W), np.float32) for m in MONTHS}; tp_cnt = {m: np.zeros((H, W), np.float32) for m in MONTHS}
region = {m: np.zeros((H, W), np.uint8) for m in MONTHS}         # 0 none, 1 train, 2 val, 3 test
for i in range(len(X)):
    m = MONTHS[meta[i, 0]]; r, c = meta[i, 1], meta[i, 2]
    region[m][r:r + PATCH, c:c + PATCH] = np.maximum(region[m][r:r + PATCH, c:c + PATCH], split[i] + 1)
    if split[i] == 2:
        tp_acc[m][r:r + PATCH, c:c + PATCH] += pall[i]; tp_cnt[m][r:r + PATCH, c:c + PATCH] += 1
tp = fp = fn = 0
for m in MONTHS:
    s = np.load(OUT_DIR / f"step2_scene_{m}.npz")
    cov = (tp_cnt[m] > 0) & s["valid"] & ~s["water"]
    pred = (tp_acc[m] / np.maximum(tp_cnt[m], 1)) >= thr
    truth = s["hr"].astype(bool)
    tp += (pred & truth & cov).sum(); fp += (pred & ~truth & cov).sum(); fn += (~pred & truth & cov).sum()
iou = tp / max(tp + fp + fn, 1)
pix_p, pix_r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
print(f"PIXEL-LEVEL (test blocks): IoU {iou:.3f} | precision {pix_p:.3f} | recall {pix_r:.3f} | F1 {2*pix_p*pix_r/max(pix_p+pix_r,1e-9):.3f}")

json.dump(dict(model_config=CFG, threshold=thr, epochs=ep, params=int(model.count_params()),
               patch_val={k: float(v) for k, v in m_va.items()}, patch_test={k: float(v) for k, v in m_te.items()},
               baseline_logreg_test={k: float(v) for k, v in m_lr.items()},
               pixel_test=dict(iou=float(iou), precision=float(pix_p), recall=float(pix_r)),
               norm_mean=mean, norm_var=var), open(OUT_DIR / "step3_metrics.json", "w"), indent=1)

# ---- figures ----
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
ax[0].plot(hist.history["loss"], label="train"); ax[0].plot(hist.history["val_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
ax[1].plot(hist.history["auc"], label="train"); ax[1].plot(hist.history["val_auc"], label="val"); ax[1].set_title("AUC"); ax[1].legend()
cm = confusion_matrix(y[te], (p_te >= thr).astype(int))
ax[2].imshow(cm, cmap="Blues"); ax[2].set_title(f"Test confusion matrix (thr {thr:.2f})")
for (i, j), n in np.ndenumerate(cm): ax[2].text(j, i, n, ha="center", va="center", fontsize=14)
ax[2].set_xticks([0, 1], ["pred low", "pred high"]); ax[2].set_yticks([0, 1], ["true low", "true high"])
plt.tight_layout(); plt.savefig(OUT_DIR / "step3_training_and_test.png", dpi=80); plt.close()

for m in MONTHS:
    s = np.load(OUT_DIR / f"step2_scene_{m}.npz"); prob = scene_prob[m]
    pred = np.nan_to_num(prob, nan=0) >= thr; truth = s["hr"].astype(bool); ok = s["valid"] & ~s["water"]
    ov = np.zeros((H, W, 3)); ov[~s["valid"]] = .08; ov[ok] = .78; ov[s["sea"]] = (.05, .1, .5); ov[s["water"]] = (.2, .8, 1)
    ov[pred & truth & ok] = (.1, .7, .1); ov[pred & ~truth & ok] = (.95, .2, .2); ov[~pred & truth & ok] = (.2, .3, .95)
    dim = (region[m] != 3) & ok; ov[dim] = ov[dim] * .35 + .3
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    ax[0].imshow(np.ma.masked_invalid(prob), cmap="inferno", vmin=0, vmax=1); ax[0].set_title(f"{m}  CNN+CBAM P(high-risk)")
    ax[1].imshow(np.where(ok, pred, np.nan), cmap="Reds", vmin=0, vmax=1); ax[1].set_title("predicted high-risk mask")
    ax[2].imshow(ov); ax[2].set_title("vs proxy mask: green=hit, red=false alarm, blue=miss\n(bright = TEST blocks, faded = train/val)")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT_DIR / f"step3_riskmap_{m}.png", dpi=55); plt.close()
print("done")
