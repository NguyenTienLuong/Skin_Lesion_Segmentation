from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data_pipeline import (
    get_train_transform,
    get_val_transform,
    get_test_transform,
)


# 1. CONFIG

DATA_DIR = Path("data")

TRAIN_IMAGE_DIR = DATA_DIR / "train"
TRAIN_MASK_DIR = DATA_DIR / "mask_train"

VAL_IMAGE_DIR = DATA_DIR / "val"
VAL_MASK_DIR = DATA_DIR / "mask_val"   

TEST_IMAGE_DIR = DATA_DIR / "test"
TEST_MASK_DIR = DATA_DIR / "mask_test" 

IMAGE_SIZE = 256
BATCH_SIZE = 8
USE_GRAY_WORLD = True
NUM_WORKERS = 0 

# 2. Dataset: doc anh, mask va ap dung transform

class SkinLesionDataset(Dataset):
    def __init__(self, image_dir, transform, mask_dir=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.transform = transform

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Khong tim thay thu muc anh: {self.image_dir}")

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.image_paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )

        if len(self.image_paths) == 0:
            raise ValueError(f"Khong tim thay file anh trong: {self.image_dir}")

        self.mask_paths = {}
        if self.mask_dir is not None:
            if not self.mask_dir.is_dir():
                raise FileNotFoundError(
                    f"Khong tim thay thu muc mask: {self.mask_dir}"
                )

            for path in self.mask_dir.iterdir():
                if path.is_file() and path.suffix.lower() in extensions:
                    if path.stem in self.mask_paths:
                        raise ValueError(f"Ten mask bi trung: {path.stem}")
                    self.mask_paths[path.stem] = path

            missing_masks = [
                p.name for p in self.image_paths
                if p.stem not in self.mask_paths
            ]
            if missing_masks:
                raise ValueError(
                    f"Thieu mask cho {len(missing_masks)} anh trong "
                    f"{self.image_dir}. Vi du: {missing_masks[:5]}"
                )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        # OpenCV doc anh mau theo thu tu BGR; doi sang RGB.
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Khong doc duoc anh: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Neu khong co mask, transform chi nhan image.
        if self.mask_dir is None:
            result = self.transform(image=image)
            return result["image"], image_path.name

        mask_path = self.mask_paths[image_path.stem]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Khong doc duoc mask: {mask_path}")

        # Gia dinh mask nhi phan: nen=0, vung ton thuong=1.
        # Neu mask cua ban da la 0/1 thi buoc nay van giu dung nhan.
        mask = (mask > 127).astype(np.uint8)

        # Truyen image va mask cung luc de cac phep bien doi hinh hoc dong bo.
        result = self.transform(image=image, mask=mask)
        image = result["image"]
        mask = result["mask"]

        # Mask co shape [H, W], gia tri 0/1, dtype long.
        mask = torch.as_tensor(np.asarray(mask).copy(), dtype=torch.long)

        return image, mask


# 3. Tao transform cho tung tap
train_transform = get_train_transform(
    size=IMAGE_SIZE,
    use_gray_world=USE_GRAY_WORLD,
)

val_transform = get_val_transform(
    size=IMAGE_SIZE,
    use_gray_world=USE_GRAY_WORLD,
)

test_transform = get_test_transform(
    size=IMAGE_SIZE,
    use_gray_world=USE_GRAY_WORLD,
)

# 4. Tao Dataset

train_dataset = SkinLesionDataset(
    image_dir=TRAIN_IMAGE_DIR,
    mask_dir=TRAIN_MASK_DIR,
    transform=train_transform,
)

val_dataset = SkinLesionDataset(
    image_dir=VAL_IMAGE_DIR,
    mask_dir=VAL_MASK_DIR,
    transform=val_transform,
)

test_dataset = SkinLesionDataset(
    image_dir=TEST_IMAGE_DIR,
    mask_dir=TEST_MASK_DIR,
    transform=test_transform,
)

# 5. Tao DataLoader: gom du lieu thanh batch

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,       # Chi shuffle tap train
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

# 6. Kiem tra mot batch xem transform da hoat dong chua

if __name__ == "__main__":
    print("So anh train:", len(train_dataset))
    print("So anh val:  ", len(val_dataset))
    print("So anh test: ", len(test_dataset))

    # Train co mask
    images, masks = next(iter(train_loader))
    print("\nTRAIN")
    print("Image shape:", images.shape)  
    print("Mask shape: ", masks.shape)   
    print("Image dtype:", images.dtype)  
    print("Mask dtype: ", masks.dtype)   
    print("Mask labels:", torch.unique(masks))

    val_batch = next(iter(val_loader))
    print("\nVAL")
    print("Image batch shape:", val_batch[0].shape)
    print("Val co mask:", VAL_MASK_DIR is not None)

    test_batch = next(iter(test_loader))
    print("\nTEST")
    print("Image batch shape:", test_batch[0].shape)
    print("Test co mask:", TEST_MASK_DIR is not None)

    print("\nDa doc du lieu va ap dung transform thanh cong.")
