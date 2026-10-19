
import albumentations as A

from src.data.augmentation import (
    resize_tf,
    normalize_image_tf,
    gray_world,
    dihedral_tf,
    affine_tf,
    light_tf,
)


DEFAULT_SEED = 42


def get_train_transform(size, use_gray_world=True, seed=DEFAULT_SEED):
    """Resize -> optional Gray World -> augmentation -> normalize -> tensor."""
    steps = [resize_tf(size)]
    if use_gray_world:
        steps.append(A.Lambda(image=gray_world, p=1.0))
    steps.extend([
        dihedral_tf(),
        affine_tf(),
        light_tf(),
    ])
    steps.extend(normalize_image_tf())
    return A.Compose(steps, seed=int(seed))


def get_val_transform(size, use_gray_world=True, seed=DEFAULT_SEED):
    """Deterministic validation preprocessing."""
    steps = [resize_tf(size)]
    if use_gray_world:
        steps.append(A.Lambda(image=gray_world, p=1.0))
    steps.extend(normalize_image_tf())
    return A.Compose(steps, seed=int(seed))


def get_test_transform(size, use_gray_world=True, seed=DEFAULT_SEED):
    """Test preprocessing is identical to validation preprocessing."""
    return get_val_transform(
        size=size,
        use_gray_world=use_gray_world,
        seed=seed,
    )


__all__ = [
    "DEFAULT_SEED",
    "get_train_transform",
    "get_val_transform",
    "get_test_transform",
]

