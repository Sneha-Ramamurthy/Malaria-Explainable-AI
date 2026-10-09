"""STEP 3 - Train MobileNetV2+CBAM, evaluate, and build risk maps.

Run: python -m src.step3_train
"""

import os
import json
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
)
from sklearn.linear_model import LogisticRegression

from .config import OUT_DIR, ROOT
from .models import build_model


# ---------------- Configuration ----------------

keras.utils.set_random_seed(42)

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 64
STRIDE = 16
BATCH_SIZE = 32
EPOCHS = 30


# ---------------- 1. Load Step 2 outputs ----------------

meta_path = OUT_DIR / "step2_meta.json"
patch_path = OUT_DIR / "step2_patches.npz"

if not meta_path.exists():
    raise FileNotFoundError(
        f"Missing {meta_path}. Run Step 2 first."
    )

if not patch_path.exists():
    raise FileNotFoundError(
        f"Missing {patch_path}. Run Step 2 first."
    )

with open(meta_path, "r", encoding="utf-8") as file:
    meta_j = json.load(file)

MONTHS = meta_j["months"]
PATCH = int(meta_j["patch"])
H, W = meta_j["shape"]

with np.load(patch_path, allow_pickle=False) as data:
    required = ("X", "y", "split", "valid", "mask", "meta")
    missing = [key for key in required if key not in data.files]

    if missing:
        raise ValueError(
            f"Step 2 dataset is missing required arrays: {missing}"
        )

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.uint8)
    split = data["split"].astype(np.uint8)
    valid = data["valid"].astype(bool)
    mask = data["mask"].astype(bool)
    patch_meta = data["meta"].astype(np.int32)

if X.ndim != 4 or X.shape[1:] != (PATCH, PATCH, 2):
    raise ValueError(
        f"Unexpected X shape {X.shape}; expected "
        f"(number_of_patches, {PATCH}, {PATCH}, 2)."
    )

if not (
    len(X) == len(y) == len(split) == len(valid)
    == len(mask) == len(patch_meta)
):
    raise ValueError("Step 2 arrays have inconsistent lengths.")

if valid.shape != X.shape[:3]:
    raise ValueError(
        "The saved valid mask does not match the patch dimensions."
    )

if not np.isfinite(X).all():
    raise ValueError("Training inputs contain NaN or infinite values.")

if not np.isin(y, [0, 1]).all():
    raise ValueError("Patch labels must be binary: 0 or 1.")

if not np.isin(split, [0, 1, 2]).all():
    raise ValueError("Split values must be 0=train, 1=validation, 2=test.")

tr = split == 0
va = split == 1
te = split == 2

print(
    f"Training patches: {tr.sum()} | "
    f"Validation patches: {va.sum()} | "
    f"Test patches: {te.sum()}"
)

if not tr.any() or not va.any() or not te.any():
    raise ValueError(
        "One or more data splits are empty. Check the spatial-block split "
        "in Step 2."
    )

if len(np.unique(y[tr])) < 2:
    raise ValueError(
        "Training data contains only one class. "
        "Check Step 2 patch labels and spatial splits."
    )

if len(np.unique(y[va])) < 2:
    raise ValueError(
        "Validation data contains only one class, so validation AUC "
        "cannot be calculated. Review Step 2 labels and split sizes."
    )

if len(np.unique(y[te])) < 2:
    raise ValueError(
        "Test data contains only one class, so test AUC cannot be "
        "calculated reliably. Review Step 2 labels and split sizes."
    )


# ---------------- 2. Calculate normalization statistics ----------------

train_X = X[tr]
train_valid = valid[tr]

means = []
variances = []

for channel in range(2):
    pixels = train_X[..., channel][train_valid]

    if pixels.size == 0:
        raise ValueError(
            f"No valid training pixels for channel {channel}."
        )

    means.append(float(pixels.mean()))
    variances.append(float(pixels.var()))

mean = means
var = variances

CFG = dict(
    alpha=0.35,
    cut="block_5_add",
    dropout=0.5,
    up=128,
)

model = build_model(mean, var, **CFG)

print("Model parameters:", f"{model.count_params():,}")


# ---------------- 3. Prepare training datasets ----------------

def augment_patch(x, target):
    """Apply label-preserving spatial augmentation and small noise."""
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    k = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=4,
        dtype=tf.int32,
    )
    x = tf.image.rot90(x, k)

    sensor_noise = tf.random.normal(
        tf.shape(x),
        mean=0.0,
        stddev=0.02,
    )

    channel_offset = tf.random.normal(
        [1, 1, 2],
        mean=0.0,
        stddev=0.03,
    )

    x = x + sensor_noise + channel_offset

    return x, target


ds_tr = (
    tf.data.Dataset.from_tensor_slices(
        (X[tr], y[tr].astype(np.float32))
    )
    .shuffle(
        buffer_size=min(1000, int(tr.sum())),
        seed=42,
        reshuffle_each_iteration=True,
    )
    .map(
        augment_patch,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

ds_va = (
    tf.data.Dataset.from_tensor_slices(
        (X[va], y[va].astype(np.float32))
    )
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)

positive_fraction = float(y[tr].mean())

if not 0 < positive_fraction < 1:
    raise ValueError(
        "Cannot calculate class weights: training labels need both classes."
    )

class_weights = {
    0: 0.5 / (1.0 - positive_fraction),
    1: 0.5 / positive_fraction,
}

model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=5e-4,
        weight_decay=1e-2,
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.AUC(name="auc"),
    ],
)

# ---------------- 4. Train the model ----------------

start_time = time.time()

history = model.fit(
    ds_tr,
    validation_data=ds_va,
    epochs=EPOCHS,
    class_weight=class_weights,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=10,
            restore_best_weights=True,
        )
    ],
)

epochs_trained = len(history.history["loss"])

print(
    f"Training finished in {time.time() - start_time:.0f} seconds."
)
print(
    f"Epochs trained: {epochs_trained} | "
    f"Best validation AUC: {max(history.history['val_auc']):.3f}"
)

model_path = MODEL_DIR / "mobilenetv2_cbam.keras"
model.save(model_path)

print("Saved model:", model_path)


# ---------------- 5. Patch-level evaluation ----------------

p_va = model.predict(X[va], batch_size=128, verbose=0).ravel()
p_te = model.predict(X[te], batch_size=128, verbose=0).ravel()

# Select the threshold using validation predictions only.
# This avoids using test predictions to choose the threshold.
threshold = float(np.quantile(p_va, 0.70))


def patch_metrics(true_labels, probabilities, cutoff):
    """Calculate binary classification metrics."""
    predictions = (probabilities >= cutoff).astype(np.uint8)

    return {
        "precision": float(
            precision_score(
                true_labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                true_labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                true_labels,
                predictions,
                zero_division=0,
            )
        ),
        "auc": float(roc_auc_score(true_labels, probabilities)),
        "accuracy": float(accuracy_score(true_labels, predictions)),
    }


metrics_val = patch_metrics(y[va], p_va, threshold)
metrics_test = patch_metrics(y[te], p_te, threshold)


# ---------------- 6. Logistic regression baseline ----------------

def patch_features(patches, valid_masks):
    """Summarise NDVI and NDWI within valid pixels."""
    features = []

    for patch, valid_mask in zip(patches, valid_masks):
        is_valid = valid_mask.astype(bool)

        ndvi_values = patch[..., 0][is_valid]
        ndwi_values = patch[..., 1][is_valid]

        if ndvi_values.size == 0 or ndwi_values.size == 0:
            raise ValueError(
                "A patch contains no valid pixels for baseline features."
            )

        features.append([
            float(ndvi_values.mean()),
            float(ndvi_values.std()),
            float(ndwi_values.mean()),
            float(ndwi_values.std()),
            float(np.percentile(ndwi_values, 95)),
        ])

    return np.asarray(features, dtype=np.float32)


train_features = patch_features(X[tr], valid[tr])
test_features = patch_features(X[te], valid[te])

baseline = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)

baseline.fit(train_features, y[tr])

baseline_prob = baseline.predict_proba(test_features)[:, 1]
metrics_baseline = patch_metrics(y[te], baseline_prob, 0.5)

print(
    "\nPATCH TEST METRICS — CNN + CBAM:",
    {key: round(value, 3) for key, value in metrics_test.items()},
    f"| threshold={threshold:.3f}",
)

print(
    "PATCH TEST METRICS — Logistic regression baseline:",
    {key: round(value, 3) for key, value in metrics_baseline.items()},
)


# ---------------- 7. Scene-level probability maps ----------------

scene_probabilities = {}
scene_data = {}

for month in MONTHS:
    scene_path = OUT_DIR / f"step2_scene_{month}.npz"

    if not scene_path.exists():
        raise FileNotFoundError(
            f"Missing {scene_path}. Run Step 2 first."
        )

    with np.load(scene_path, allow_pickle=False) as scene_file:
        required_scene_fields = (
            "ndvi",
            "ndwi",
            "valid",
            "valid_risk",
            "water",
            "risk",
            "hr",
            "sea",
        )

        missing_fields = [
            key for key in required_scene_fields
            if key not in scene_file.files
        ]

        if missing_fields:
            raise ValueError(
                f"{scene_path.name} is missing fields: {missing_fields}. "
                "Regenerate the Step 2 outputs with the updated code."
            )

        scene_data[month] = {
            key: scene_file[key].copy()
            for key in required_scene_fields
        }

    scene = scene_data[month]

    if scene["ndvi"].shape != (H, W):
        raise ValueError(
            f"Scene dimensions for {month} do not match step2_meta.json."
        )

    if scene["ndwi"].shape != (H, W):
        raise ValueError(
            f"NDWI dimensions for {month} do not match metadata."
        )

    if scene["valid_risk"].shape != (H, W):
        raise ValueError(
            f"Risk-validity mask for {month} has incorrect dimensions."
        )

    scene_valid = scene["valid_risk"].astype(bool)

    image = np.stack(
        [scene["ndvi"], scene["ndwi"]],
        axis=-1,
    ).astype(np.float32)

    image *= scene_valid[..., None]

    accumulator = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    rows = []
    cols = []
    windows = []

    for row in range(0, H - PATCH + 1, STRIDE):
        for col in range(0, W - PATCH + 1, STRIDE):
            patch_valid = scene_valid[
                row:row + PATCH,
                col:col + PATCH,
            ]

            if patch_valid.mean() < 0.6:
                continue

            rows.append(row)
            cols.append(col)
            windows.append(
                image[
                    row:row + PATCH,
                    col:col + PATCH,
                ]
            )

    if not windows:
        raise ValueError(
            f"No valid scene windows generated for {month}."
        )

    window_array = np.asarray(windows, dtype=np.float32)
    probabilities = model.predict(
        window_array,
        batch_size=128,
        verbose=0,
    ).ravel()

    for row, col, probability in zip(rows, cols, probabilities):
        accumulator[
            row:row + PATCH,
            col:col + PATCH,
        ] += probability

        counts[
            row:row + PATCH,
            col:col + PATCH,
        ] += 1

    probability_map = np.full((H, W), np.nan, dtype=np.float32)

    covered = counts > 0

    probability_map[covered] = (
        accumulator[covered] / counts[covered]
    )

    scene_probabilities[month] = probability_map

    np.savez_compressed(
        OUT_DIR / f"step3_scene_prob_{month}.npz",
        prob=probability_map.astype(np.float16),
    )


# ---------------- 8. Evaluate test-block pixels ----------------

test_probabilities = model.predict(
    X[te],
    batch_size=128,
    verbose=0,
).ravel()

test_indices = np.flatnonzero(te)

test_accumulator = {
    month: np.zeros((H, W), dtype=np.float32)
    for month in MONTHS
}

test_counts = {
    month: np.zeros((H, W), dtype=np.float32)
    for month in MONTHS
}

region = {
    month: np.zeros((H, W), dtype=np.uint8)
    for month in MONTHS
}

for local_index, index in enumerate(test_indices):
    month_index, row, col, _block_id = patch_meta[index]
    month = MONTHS[int(month_index)]

    row = int(row)
    col = int(col)
    probability = test_probabilities[local_index]

    region[month][row:row + PATCH, col:col + PATCH] = 3

    test_accumulator[month][row:row + PATCH, col:col + PATCH] += probability
    test_counts[month][row:row + PATCH, col:col + PATCH] += 1
# ---------------- 9. Save metrics ----------------

metrics = {
    "model_config": CFG,
    "threshold": threshold,
    "threshold_source": "70th percentile of validation predictions",
    "epochs": epochs_trained,
    "params": int(model.count_params()),
    "patch_val": metrics_val,
    "patch_test": metrics_test,
    "baseline_logreg_test": metrics_baseline,
    "pixel_test": {
        "iou": float(iou),
        "precision": float(pixel_precision),
        "recall": float(pixel_recall),
        "f1": float(pixel_f1),
    },
    "norm_mean": mean,
    "norm_var": var,
}

with open(
    OUT_DIR / "step3_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=2)


# ---------------- 10. Training and confusion-matrix figures ----------------

fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))

axes[0].plot(history.history["loss"], label="Train")
axes[0].plot(history.history["val_loss"], label="Validation")
axes[0].set_title("Training and validation loss")
axes[0].legend()

axes[1].plot(history.history["auc"], label="Train")
axes[1].plot(history.history["val_auc"], label="Validation")
axes[1].set_title("Training and validation AUC")
axes[1].legend()

confusion = confusion_matrix(
    y[te],
    (p_te >= threshold).astype(np.uint8),
    labels=[0, 1],
)

axes[2].imshow(confusion, cmap="Blues")
axes[2].set_title(f"Test confusion matrix (threshold {threshold:.2f})")

for (row, col), count in np.ndenumerate(confusion):
    axes[2].text(
        col,
        row,
        int(count),
        ha="center",
        va="center",
        fontsize=14,
    )

axes[2].set_xticks([0, 1], ["Predicted low", "Predicted high"])
axes[2].set_yticks([0, 1], ["Actual low", "Actual high"])

plt.tight_layout()
plt.savefig(
    OUT_DIR / "step3_training_and_test.png",
    dpi=80,
)
plt.close(fig)


# ---------------- 11. Save scene risk maps ----------------

for month in MONTHS:
    scene = scene_data[month]
    probability_map = scene_probabilities[month]

    predicted = (
        np.nan_to_num(probability_map, nan=0.0) >= threshold
    )

    truth = scene["hr"].astype(bool)
    valid_land = scene["valid_risk"].astype(bool)

    overlay = np.zeros((H, W, 3), dtype=np.float32)

    overlay[~scene["valid"]] = 0.08
    overlay[valid_land] = 0.78
    overlay[scene["sea"].astype(bool)] = (0.05, 0.1, 0.5)
    overlay[scene["water"].astype(bool)] = (0.2, 0.8, 1.0)

    overlay[predicted & truth & valid_land] = (0.1, 0.7, 0.1)
    overlay[predicted & ~truth & valid_land] = (0.95, 0.2, 0.2)
    overlay[~predicted & truth & valid_land] = (0.2, 0.3, 0.95)

    # Fade areas that were not part of the test patches.
    dim = (region[month] != 3) & valid_land
    overlay[dim] = overlay[dim] * 0.35 + 0.3

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].imshow(
        np.ma.masked_invalid(probability_map),
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    axes[0].set_title(f"{month} CNN+CBAM high-risk probability")

    axes[1].imshow(
        np.where(valid_land, predicted, np.nan),
        cmap="Reds",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Predicted high-risk mask")

    axes[2].imshow(overlay)
    axes[2].set_title(
        "Comparison with proxy labels:\n"
        "Green = hit, red = false alarm, blue = miss"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        OUT_DIR / f"step3_riskmap_{month}.png",
        dpi=55,
    )
    plt.close(fig)

print("Step 3 completed successfully.")
