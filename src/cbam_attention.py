"""Step 6 engine - extract CBAM attention from the MobileNetV2+CBAM model.

CBAM (see models.cbam_block) does two things in sequence on the trunk feature map F (B,h,w,C):
    1. channel attention  Mc = sigmoid(MLP(avgpool F) + MLP(maxpool F))   -> 'cbam_channel_att' (B,1,1,C)
       F' = F * Mc                     ("WHICH feature channels matter")
    2. spatial attention  Ms = sigmoid(conv7x7([mean_c F', max_c F']))   -> 'cbam_spatial_att' (B,h,w,1)
       out = F' * Ms                   ("WHERE it matters")             -> 'cbam_out'
Unlike Grad-CAM / SHAP, CBAM attention is a *gating* signal learned during training, not a class-specific
attribution of the output - Step 6 therefore compares it against Grad-CAM instead of assuming it is an explanation.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, cv2

from . import xai_utils as U


class CBAMExtractor:
    def __init__(self, model):
        import keras
        self.keras = keras
        # Pre-CBAM trunk features F: the trunk is a nested sub-model, so `trunk.output` lives in ITS OWN graph and is
        # not connected to the outer model's input. Take F from the OUTER graph instead: it is the first input of the
        # Multiply layer that applies channel attention (F * Mc); the other Multiply is 'cbam_out'.
        mul = next(l for l in model.layers if isinstance(l, keras.layers.Multiply) and l.name != "cbam_out")
        feat_t = mul.input[0]
        self.gm = keras.Model(model.inputs, [feat_t,                                         # F   (pre-CBAM features)
                                             model.get_layer("cbam_channel_att").output,     # Mc
                                             model.get_layer("cbam_spatial_att").output,     # Ms
                                             model.get_layer("cbam_out").output,             # F * Mc * Ms
                                             model.output])                                  # P(high-risk)

    def __call__(self, x, batch=128, size=64, keep_features=True):
        """x (N,64,64,2) -> dict:
             ch        (N,C)        channel attention weights
             sp_low    (N,h,w)      raw spatial attention at feature resolution
             sp        (N,size,size) raw spatial attention bilinearly resized to patch resolution
             energy    (N,size,size) mean |CBAM output| over channels, resized  ('what CBAM passes on')
             feat      (N,h,w,C)    pre-CBAM trunk features (for the channel <-> input analysis)
             prob      (N,)"""
        out = {k: [] for k in ("feat", "ch", "sp_low", "sp", "energy", "prob")}
        for i in range(0, len(x), batch):
            feat, ca, sa, co, p = (np.asarray(t) for t in self.gm(np.asarray(x[i:i + batch], np.float32), training=False))
            out["feat"].append(feat if keep_features else feat[:0])
            out["ch"].append(ca[:, 0, 0, :]); out["sp_low"].append(sa[..., 0]); out["prob"].append(p.ravel())
            out["sp"].append(np.stack([cv2.resize(a, (size, size), interpolation=cv2.INTER_LINEAR) for a in sa[..., 0]]))
            en = np.abs(co).mean(-1)
            out["energy"].append(np.stack([cv2.resize(a, (size, size), interpolation=cv2.INTER_LINEAR) for a in en]))
        return {k: np.concatenate(v) for k, v in out.items()}


def to_map(raw, valid):
    """Raw attention -> comparable [0,1] map (per-patch min-max over valid pixels).
    Also returns each patch's raw dynamic range, so near-flat attention (sigmoid ~0.5 everywhere) is visible."""
    rng = np.array([(r[v].max() - r[v].min()) if v.sum() else 0.0 for r, v in zip(raw, valid)])
    return U.minmax01(raw, valid), rng


def channel_input_affinity(feat, X, valid, idx):
    """How strongly does each trunk channel track NDVI / NDWI?  |Pearson r| between channel activation and the
    input (inputs area-averaged down to the feature resolution), pooled over valid cells of patches `idx`.
    Returns affinity (C,2) = (|r_NDVI|, |r_NDWI|) and signed r (C,2)."""
    h, w = feat.shape[1:3]
    Xi = np.stack([np.stack([cv2.resize(p[..., c], (w, h), interpolation=cv2.INTER_AREA) for c in (0, 1)], -1) for p in X[idx]])
    Vi = np.stack([cv2.resize(v.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) for v in valid[idx]]) > 0.5
    F = feat[idx][Vi]                                    # (M,C)
    Z = Xi[Vi]                                           # (M,2)
    Fz = (F - F.mean(0)) / np.maximum(F.std(0), 1e-8)
    Zz = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-8)
    r = (Fz.T @ Zz) / len(F)                             # (C,2)
    return np.abs(r), r


def attention_weighted_share(ch, affinity):
    """Per patch: share of channel attention mass that sits on NDVI-tracking vs NDWI-tracking channels.
    ch (N,C), affinity (C,2) -> (N,2) rows sum to 1."""
    s = ch @ affinity
    return s / np.maximum(s.sum(1, keepdims=True), 1e-12)