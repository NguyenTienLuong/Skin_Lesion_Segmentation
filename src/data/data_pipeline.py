
import albumentations as A

from Skin_Lesion_Segmentation.src.data.augmentation import (
    resize_tf,
    normalize_image_tf,
    gray_world,
    dihedral_tf,
    affine_tf,
    light_tf,
)


def get_train_transform(size, use_gray_world):
    """Gray World (tùy chọn) -> Resize -> Augmentation -> Normalize -> Tensor"""
    steps = []
    if use_gray_world:
        steps.append(A.Lambda(image=gray_world, p=1.0))
    steps.append(resize_tf(size))
    steps.extend([
        dihedral_tf(),
        affine_tf(),
        light_tf(),
    ])
    steps.extend(normalize_image_tf())
    return A.Compose(steps)


def get_val_transform(size, use_gray_world):
    """Gray World (tùy chọn) -> Resize-> Normalize -> Tensor"""
    steps = []
    if use_gray_world:
        steps.append(A.Lambda(image=gray_world, p=1.0))
    steps.append(resize_tf(size))
    steps.extend(normalize_image_tf())
    return A.Compose(steps)


def get_test_transform(size, use_gray_world):
    """Gray World (tùy chọn) -> Resize-> Normalize -> Tensor"""
    steps = []
    if use_gray_world:
        steps.append(A.Lambda(image=gray_world, p=1.0))
    steps.append(resize_tf(size))
    steps.extend(normalize_image_tf())
    return A.Compose(steps)