import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Tiền xử lý: áp dụng cho MỌI tập
def resize_tf(size):
    """Ảnh dùng bilinear, mask dùng nearest."""
    return A.Resize(size, size,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST)

def normalize_tf():
    """ImageNet normalization theo protocol nhóm."""
    return [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]

# Augmentation online: CHỈ cho train
def dihedral_tf():
    """8 đối xứng của hình vuông."""
    return A.D4(p=1.0)

def affine_tf(max_rot=44, max_zoom=1.05, p=0.75):
    """Xoay tối đa 44 độ, phóng to tối đa 1,05."""
    return A.Affine(rotate=(-max_rot, max_rot), scale=(1.0, max_zoom),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST, p=p)

def light_tf(max_bright=0.2, max_contrast=0.2, p=0.75):
    """Đổi ánh sáng ngẫu nhiên."""
    return A.RandomBrightnessContrast(brightness_limit=max_bright,
                                      contrast_limit=max_contrast, p=p)

# Ghép pipeline
def get_transforms(size, train):
    steps = [resize_tf(size)]                                  # 1. tiền xử lý
    if train:
        steps += [dihedral_tf(), affine_tf(), light_tf()]      # 2. augmentation
    steps += normalize_tf()                                    # 3. chuẩn hóa
    return A.Compose(steps)