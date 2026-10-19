import numpy as np
import torch

from src.data.augmentation import gray_world
from src.data.data_pipeline import get_train_transform, get_val_transform


def test_gray_world_shape_dtype():
    x = np.random.default_rng(42).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    y = gray_world(x)
    assert y.shape == x.shape and y.dtype == np.uint8


def test_train_transform_is_reproducible_with_seed():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    mask = rng.integers(0, 2, (64, 64), dtype=np.uint8)
    first = get_train_transform(64, seed=123)(image=image, mask=mask)
    second = get_train_transform(64, seed=123)(image=image, mask=mask)
    assert torch.equal(first["image"], second["image"])
    assert torch.equal(first["mask"], second["mask"])


def test_validation_transform_has_no_random_augmentation():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    mask = rng.integers(0, 2, (64, 64), dtype=np.uint8)

    first = get_val_transform(64, seed=123)(image=image, mask=mask)
    second = get_val_transform(64, seed=999)(image=image, mask=mask)

    assert torch.equal(first["image"], second["image"])
    assert torch.equal(first["mask"], second["mask"])

    transform_names = [type(step).__name__ for step in get_val_transform(64).transforms]
    assert transform_names == ["Resize", "Lambda", "Normalize", "ToTensorV2"]
