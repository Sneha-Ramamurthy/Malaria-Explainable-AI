"""Tests for Step 2 dataset outputs.

Run with: python -m pytest tests -q
These tests need the Step 2 output file to exist.
"""

import numpy as np
import pytest

from src.config import OUT_DIR


PATCH_FILE = OUT_DIR / "step2_patches.npz"


@pytest.fixture
def dataset():
    if not PATCH_FILE.exists():
        pytest.skip(
            "Step 2 outputs not found. Run the dataset pipeline first."
        )

    with np.load(PATCH_FILE, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def test_patch_shapes_and_dimensions(dataset):
    X = dataset["X"]
    mask = dataset["mask"]
    risk = dataset["risk"]
    valid = dataset["valid"]

    assert X.ndim == 4
    assert X.shape[1:] == (64, 64, 2)
    assert mask.shape == risk.shape == valid.shape
    assert mask.shape == X.shape[:3]


def test_dataset_has_finite_values(dataset):
    assert np.isfinite(dataset["X"]).all()
    assert np.isfinite(dataset["risk"]).all()


def test_labels_and_splits_are_valid(dataset):
    y = dataset["y"]
    split = dataset["split"]

    assert len(y) == len(dataset["X"])
    assert len(split) == len(y)
    assert set(np.unique(y)).issubset({0, 1})
    assert set(np.unique(split)).issubset({0, 1, 2})


def test_spatial_blocks_do_not_leak_between_splits(dataset):
    meta = dataset["meta"]
    split = dataset["split"]

    assert meta.ndim == 2
    assert meta.shape[0] == len(split)
    assert meta.shape[1] >= 4

    block_ids = meta[:, 3]

    for first_split in (0, 1, 2):
        first_blocks = set(block_ids[split == first_split])

        for second_split in (0, 1, 2):
            if first_split >= second_split:
                continue

            second_blocks = set(block_ids[split == second_split])
            assert first_blocks.isdisjoint(second_blocks), (
                f"Spatial blocks overlap between splits "
                f"{first_split} and {second_split}"
            )
