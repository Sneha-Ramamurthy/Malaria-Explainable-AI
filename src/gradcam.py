"""Step 4 - Grad-CAM for the MobileNetV2+CBAM patch classifier.

Gradients are taken w.r.t. the pre-sigmoid logit (same direction as d p, but no saturation).
Reusable in Steps 6-8 (fusion / final explanation).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, tensorflow as tf, keras
from .models import ChannelPool  # noqa: F401  (registers the custom layer for loading)


def load_model(path):
    return keras.models.load_model(path, compile=False)


class GradCAM:
    def __init__(self, model, layer="last_conv"):
        self.dense = model.get_layer("risk_prob")
        self.gm = keras.Model(model.inputs, [model.get_layer(layer).output, self.dense.input])

    def __call__(self, x, batch=128, size=64):
        """x: (N,64,64,2) float32 -> (cam in [0,1] (N,size,size), logits (N,), empty_flag (N,))"""
        cams, logits = [], []
        W = tf.convert_to_tensor(self.dense.kernel); b = tf.convert_to_tensor(self.dense.bias)
        for i in range(0, len(x), batch):
            xb = tf.convert_to_tensor(x[i:i + batch], tf.float32)
            with tf.GradientTape() as tape:
                A, h = self.gm(xb, training=False)
                logit = tf.matmul(h, W) + b
            g = tape.gradient(logit, A)
            w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)              # channel importance
            cam = tf.nn.relu(tf.reduce_sum(w * A, axis=-1, keepdims=True))  # keep positive evidence only
            cams.append(tf.image.resize(cam, (size, size), method="bilinear")[..., 0].numpy())
            logits.append(logit.numpy().ravel())
        cam = np.concatenate(cams); mx = cam.reshape(len(cam), -1).max(1)
        return cam / np.maximum(mx, 1e-8)[:, None, None], np.concatenate(logits), mx <= 1e-8
