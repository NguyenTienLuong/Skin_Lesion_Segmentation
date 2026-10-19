import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# 1. PREPROCESSING: chỉ áp dụng cho ảnh RGB, không áp dụng mask
def resize_tf(size):
    """Ảnh dùng bilinear, mask dùng nearest."""
    return A.Resize(
        size, 
        size,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST)

def normalize_image_tf():
    """Chuẩn hóa ảnh theo ImageNet và chuyển thành Tensor."""
    return [
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
        ToTensorV2(),
    ]

def gray_world(image, **kwargs):
    """Cân bằng màu Gray World."""
    img = image.astype(np.float32)
    channel_means = img.mean(axis=(0, 1))
    gray_mean = channel_means.mean()
    scale = gray_mean / np.maximum(channel_means, 1e-6)
    corrected = img * scale.reshape(1, 1, 3)
    return np.clip(corrected, 0, 255).astype(np.uint8)

# 2. AUGMENTATION ONLINE: chỉ cho train
def dihedral_tf():
    """8 phép đối xứng, áp dụng đồng bộ lên ảnh và mask."""
    return A.D4(p=1.0)

def affine_tf(max_rot=44, max_zoom=1.05, p=0.75):
    """Xoay và zoom đồng bộ lên ảnh và mask."""
    return A.Affine(
        rotate=(-max_rot, max_rot),
        scale=(1.0, max_zoom),
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        p=p,
    )

def light_tf(max_bright=0.2, max_contrast=0.2, p=0.75):
    """Chỉ thay đổi độ sáng và tương phản của ảnh."""
    return A.RandomBrightnessContrast(
        brightness_limit=max_bright,
        contrast_limit=max_contrast,
        p=p,
    )
    
    