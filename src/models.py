"""Step 3 - Hybrid MobileNetV2 + CBAM patch classifier (stand-in for the paper's model).

Input : 64x64x2 patch (NDVI, NDWI)       Output: P(high-risk)
Layers of interest for the XAI steps (all top-level, addressable by name):
    'cbam_channel_att'  (B,1,1,C)  channel-attention weights
    'cbam_spatial_att'  (B,H,W,1)  spatial-attention map
    'last_conv'                    Grad-CAM target layer
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="xai")
class ChannelPool(layers.Layer):
    """[mean, max] over the channel axis -> (B,H,W,2), the input of CBAM spatial attention."""
    def call(self, x):
        return ops.concatenate([ops.mean(x, axis=-1, keepdims=True),
                                ops.max(x, axis=-1, keepdims=True)], axis=-1)

    def compute_output_shape(self, s):
        return tuple(s[:-1]) + (2,)


def cbam_block(x, ratio=8, name="cbam"):
    ch = x.shape[-1]
    mlp1 = layers.Dense(max(ch // ratio, 4), activation="relu", name=f"{name}_mlp1")
    mlp2 = layers.Dense(ch, name=f"{name}_mlp2")
    avg = layers.GlobalAveragePooling2D(keepdims=True)(x)
    mx = layers.GlobalMaxPooling2D(keepdims=True)(x)
    ca = layers.Activation("sigmoid", name=f"{name}_channel_att")(
        layers.Add()([mlp2(mlp1(avg)), mlp2(mlp1(mx))]))
    x = layers.Multiply()([x, ca])
    sa = layers.Conv2D(1, 7, padding="same", activation="sigmoid", name=f"{name}_spatial_att")(ChannelPool()(x))
    return layers.Multiply(name=f"{name}_out")([x, sa])


def build_model(mean, var, input_shape=(64, 64, 2), alpha=0.5, cut="block_12_add", dropout=0.3, up=128):
    inp = keras.Input(input_shape, name="ndvi_ndwi")
    x = layers.Normalization(mean=mean, variance=var, name="norm")(inp)
    x = layers.Resizing(up, up, interpolation="bilinear", name="upsample")(x)
    bb = keras.applications.MobileNetV2(input_shape=(up, up, input_shape[-1]), alpha=alpha,
                                        include_top=False, weights=None)
    for l in bb.layers:                       # default momentum (0.999) is too slow for a small dataset:
        if isinstance(l, layers.BatchNormalization):   # running stats would never converge -> broken inference
            l.momentum = 0.9
    bb = keras.Model(bb.input, bb.get_layer(cut).output, name="mobilenetv2_trunk")  # 8x8 feature map
    x = bb(x)
    x = cbam_block(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="last_conv")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="risk_prob")(x)
    return keras.Model(inp, out, name="mobilenetv2_cbam")
