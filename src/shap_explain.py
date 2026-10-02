"""Step 5 engine - SHAP attributions for the MobileNetV2+CBAM patch classifier.

Method: *Expected Gradients* (Erion et al. 2021) = the algorithm behind `shap.GradientExplainer`.
It approximates SHAP values for a differentiable model by averaging, over random baselines b drawn from a
background set and a random point on the line b -> x,   (x - b) * d f / d input.
Properties we rely on:
  * additive ("Input -> Feature Contribution -> Risk Prediction"):  f(x) ~= E_b[f(b)] + sum(phi)   (completeness)
  * a value per input pixel and per feature (NDVI, NDWI), so spatial AND per-feature contributions come out of one run.

f is the pre-sigmoid LOGIT (same target as Grad-CAM in Step 4: avoids saturation; additive in log-odds).
The numerical core (`expected_gradients`) is backend-free so it can be unit-tested; `LogitGradFn` is the
TensorFlow/Keras 3 glue.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np

FEATURES = ("NDVI", "NDWI")


# ----------------------------------------------------------------------------- backend-free core
def expected_gradients(value_and_grad, X, baselines, n_samples=100, batch=64, seed=0):
    """
    value_and_grad(points) -> (values (B,), grads (B,H,W,C))   evaluated per sample (inference mode).
    X (N,H,W,C), baselines (M,H,W,C) background set.
    Returns phi (N,H,W,C) float32, base_value (float) = mean f(baselines).

    Variance reduction: the k-th of n_samples path points uses alpha ~ U[k/n, (k+1)/n) (stratified).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float32); baselines = np.asarray(baselines, np.float32)
    base_vals = np.concatenate([value_and_grad(baselines[i:i + batch])[0] for i in range(0, len(baselines), batch)])
    phi = np.zeros_like(X)
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]; B = len(xb)
        acc = np.zeros_like(xb)
        for k in range(n_samples):
            b = baselines[rng.integers(len(baselines), size=B)]
            alpha = ((k + rng.random(B)) / n_samples).astype(np.float32)[:, None, None, None]
            _, g = value_and_grad(b + alpha * (xb - b))
            acc += (xb - b) * g
        phi[i:i + batch] = acc / n_samples
    return phi, float(base_vals.mean())


def completeness(phi, fx, base):
    """How well  sum(phi) ~= f(x) - base  holds (Monte-Carlo error of the estimator)."""
    s = phi.reshape(len(phi), -1).sum(1); tgt = np.asarray(fx) - base
    err = np.abs(s - tgt)
    return dict(mean_abs_err=float(err.mean()), median_abs_err=float(np.median(err)),
                mean_abs_target=float(np.abs(tgt).mean()),
                pearson_sum_phi_vs_target=float(np.corrcoef(s, tgt)[0, 1]) if len(s) > 2 and s.std() > 0 else None,
                rel_err_of_variance=float(err.var() / max(tgt.var(), 1e-12)))


def channel_contributions(phi, valid=None):
    """(N,C) summed contribution of each input feature (logit units). Optionally only valid pixels."""
    if valid is not None:
        phi = phi * valid[..., None]
    return phi.sum((1, 2))


def evidence_map(phi, valid, sigma=1.5):
    """Spatial 'where does the model find evidence FOR high risk': positive part of the feature-summed SHAP,
    lightly smoothed (raw SHAP is pixel-noisy, Grad-CAM is a smooth 16x16 map), masked to valid, scaled to [0,1]."""
    from .xai_utils import smooth, max01
    pos = np.maximum(phi.sum(-1), 0) * valid
    return max01(smooth(pos, sigma), valid)


# ----------------------------------------------------------------------------- TensorFlow / Keras 3 glue
class LogitGradFn:
    """Callable  points -> (logit (B,), d logit / d input (B,H,W,C))  for a loaded `mobilenetv2_cbam` model."""

    def __init__(self, model):
        import tensorflow as tf, keras
        self.tf, self.keras = tf, keras
        self.dense = model.get_layer("risk_prob")
        self.feat = keras.Model(model.inputs, self.dense.input)             # -> pooled features before the head
        W = tf.convert_to_tensor(self.dense.kernel); b = tf.convert_to_tensor(self.dense.bias)
        self._W_np, self._b_np = np.asarray(W), np.asarray(b)

        @tf.function(reduce_retracing=True)
        def _vag(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                z = tf.matmul(self.feat(x, training=False), W) + b          # logit, inference mode
            return z[:, 0], tape.gradient(z, x)     # batch items are independent (BN frozen, dropout off)
        self._vag = _vag

    def __call__(self, x):
        z, g = self._vag(self.tf.convert_to_tensor(np.asarray(x, np.float32)))
        return z.numpy(), g.numpy()

    def logit(self, x, batch=256):
        return np.concatenate([self(x[i:i + batch])[0] for i in range(0, len(x), batch)])

    def as_logit_model(self):
        """Keras model input -> logit (for the optional cross-check with the `shap` library)."""
        keras = self.keras
        out = keras.layers.Dense(1, name="logit")(self.feat.output)
        m = keras.Model(self.feat.inputs, out)
        m.get_layer("logit").set_weights([self._W_np, self._b_np])
        return m


def shap_library_crosscheck(fn, background, X_eval, nsamples=100):
    """Optional: compare our expected-gradients to `shap.GradientExplainer` (same algorithm, library implementation).
    Returns dict(pearson, n) or dict(error=...) - never raises (shap/Keras-3 compatibility varies by version)."""
    try:
        import shap
        ex = shap.GradientExplainer(fn.as_logit_model(), background)
        sv = ex.shap_values(X_eval, nsamples=nsamples)
        sv = np.asarray(sv[0] if isinstance(sv, list) else sv).reshape(X_eval.shape)
        mine, _ = expected_gradients(fn, X_eval, background, n_samples=nsamples)
        return dict(n=int(len(X_eval)), pearson=float(np.corrcoef(sv.ravel(), mine.ravel())[0, 1]),
                    shap_version=getattr(shap, "__version__", "?"))
    except Exception as e:                                                   # noqa: BLE001
        return dict(error=f"{type(e).__name__}: {str(e)[:200]}")
