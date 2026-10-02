"""Does THIS machine reproduce the trained model?  Run:  python -m src.diag_model   (~1 min)"""
import os, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, tensorflow as tf, keras, sklearn
from sklearn.metrics import roc_auc_score
from .config import OUT_DIR, ROOT
from .gradcam import load_model
from .shap_explain import LogitGradFn
from .cbam_attention import CBAMExtractor
from . import xai_utils as U

print("python-side versions: tensorflow", tf.__version__, "| keras", keras.__version__, "| numpy", np.__version__, "| sklearn", sklearn.__version__)
ctx = U.load_context(); X, y, split = ctx.X, ctx.y, ctx.split
model = load_model(ROOT / "models" / "mobilenetv2_cbam.keras")
te = split == 2

print("\nreported in step3_metrics.json:", {k: v for k, v in ctx.M3.items() if any(s in k.lower() for s in ("auc", "f1", "iou", "thresh", "keras", "tensorflow", "version"))})
nl = model.get_layer("norm")
print("Normalization layer mean vs saved norm_mean:", np.round(np.ravel(np.asarray(nl.mean)), 4), np.round(np.ravel(ctx.M3["norm_mean"]), 4))
print("Normalization layer var  vs saved norm_var :", np.round(np.ravel(np.asarray(nl.variance)), 4), np.round(np.ravel(ctx.M3["norm_var"]), 4))

p_pred = model.predict(X, batch_size=128, verbose=0).ravel()
p4 = np.load(OUT_DIR / "step4_gradcam_patches.npz")["prob"]
print("\n[A] model.predict vs Step-4 cached prob : max|diff| %.4f  corr %.4f" % (np.abs(p_pred - p4).max(), np.corrcoef(p_pred, p4)[0, 1]))
print("    test AUC from model.predict (this machine): %.3f   | from Step-4 cached prob: %.3f" % (roc_auc_score(y[te], p_pred[te]), roc_auc_score(y[te], p4[te])))
print("    mean prob  this machine %.3f | cached %.3f | share >= threshold: %.3f vs %.3f" % (p_pred.mean(), p4.mean(), (p_pred >= ctx.THR).mean(), (p4 >= ctx.THR).mean()))

p_gm = CBAMExtractor(model)(X[:256], keep_features=False)["prob"]
print("[B] CBAM extractor prob vs model.predict : max|diff| %.5f" % np.abs(p_gm - p_pred[:256]).max())
fn = LogitGradFn(model); p_fn = 1 / (1 + np.exp(-fn.logit(X[:256])))
print("[C] SHAP logit path vs model.predict     : max|diff| %.5f" % np.abs(p_fn - p_pred[:256]).max())
p1 = np.concatenate([model.predict(X[i:i + 1], verbose=0).ravel() for i in range(32)])
print("[D] batch size 1 vs 128 (same 32 patches) : max|diff| %.5f" % np.abs(p1 - p_pred[:32]).max())
print("\nverdict hints: A big & AUC drops  -> model does not reproduce on this install (version issue)")
print("               A big but B,C,D ~0  -> cached Step-3/4 outputs are stale vs this model file; rerun steps 3-4")
print("               A,B,C all ~0        -> extraction fine; problem is elsewhere (SHAP method)")