import hashlib
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

try:
    from .data_pipeline import (
        get_train_transform,
        get_val_transform,
        get_test_transform,
    )
    from .dataset import SkinLesionDataset
except ImportError:
    from data_pipeline import (
        get_train_transform,
        get_val_transform,
        get_test_transform,
    )
    from dataset import SkinLesionDataset


# 1. TODO: FIX PATH
TRAIN_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "manifest" / "data_manifest.csv"
VAL_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "manifest" / "val_manifest.csv"
TEST_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "manifest" / "test_manifest.csv"

TRAIN_IMAGE_DIR = Path(__file__).resolve().parents[2] / "data" / "train" / "images"
TRAIN_MASK_DIR = Path(__file__).resolve().parents[2] / "data" / "train" / "masks"

VAL_IMAGE_DIR = Path(__file__).resolve().parents[2] / "data" / "validation" / "images"
VAL_MASK_DIR = Path(__file__).resolve().parents[2] / "data" / "validation" / "masks"

TEST_IMAGE_DIR = Path(__file__).resolve().parents[2] / "data" / "test" / "images"
TEST_MASK_DIR = Path(__file__).resolve().parents[2] / "data" / "test" / "masks"

IMAGE_SIZE = 256
BATCH_SIZE = 8
NUM_WORKERS = 0
USE_GRAY_WORLD = True


# 2. KIỂM TRA MANIFEST VÀ DATA LEAKAGE
def read_manifest_ids(manifest_path):
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy manifest: {manifest_path}")

    df = pd.read_csv(manifest_path, dtype={"image_id": str})

    required = {"image_id", "image_path", "mask_path", "pair_valid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{manifest_path.name} thiếu cột: {sorted(missing)}")

    if df["image_id"].isna().any():
        raise ValueError(f"{manifest_path.name} có image_id bị thiếu.")
    df["image_id"] = df["image_id"].str.strip()

    if df["image_id"].duplicated().any():
        dup = df.loc[df["image_id"].duplicated(keep=False), "image_id"].tolist()
        raise ValueError(f"ID trùng trong {manifest_path.name}: {dup[:10]}")

    valid = (
        df["pair_valid"].astype(str).str.strip().str.lower()
        .isin({"true", "1", "yes"})
    )
    return df.loc[valid].copy()


def _normalize_manifest_path(value):
    return str(value).replace("\\", "/").strip().casefold()


def check_manifest_leakage():
    manifests = {
        "train": read_manifest_ids(TRAIN_MANIFEST),
        "val": read_manifest_ids(VAL_MANIFEST),
        "test": read_manifest_ids(TEST_MANIFEST),
    }

    names = list(manifests)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common_ids = (
                set(manifests[a]["image_id"].astype(str))
                & set(manifests[b]["image_id"].astype(str))
            )
            if common_ids:
                raise ValueError(
                    f"DATA LEAKAGE: image_id có ở {a} và {b}: "
                    f"{sorted(common_ids)[:10]}"
                )

            for column in ("image_path", "mask_path"):
                pa = set(manifests[a][column].dropna().map(_normalize_manifest_path))
                pb = set(manifests[b][column].dropna().map(_normalize_manifest_path))
                common = pa & pb
                if common:
                    raise ValueError(
                        f"DATA LEAKAGE: {column} có ở {a} và {b}: "
                        f"{sorted(common)[:10]}"
                    )

    print("Manifest leakage check passed.")
    for split, df in manifests.items():
        print(f"  {split}: {len(df)} valid pairs")


def _file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def check_dataset_leakage(datasets, check_content_hash=True):
    names = list(datasets)

    resolved = {
        split: {str(s["image_path"]).casefold() for s in ds.samples}
        for split, ds in datasets.items()
    }
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = resolved[a] & resolved[b]
            if common:
                raise ValueError(
                    f"DATA LEAKAGE: cùng file ảnh ở {a} và {b}: "
                    f"{sorted(common)[:5]}"
                )

    if check_content_hash:
        hashes = {
            split: {_file_md5(s["image_path"]) for s in ds.samples}
            for split, ds in datasets.items()
        }
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                common = hashes[a] & hashes[b]
                if common:
                    raise ValueError(
                        f"DATA LEAKAGE: {len(common)} ảnh trùng nội dung "
                        f"giữa {a} và {b}."
                    )

    print("Dataset leakage check passed.")


# 3. TẠO DATASET VÀ DATALOADER
def build_dataloaders(check_content_hash=True):
    image_size = IMAGE_SIZE
    batch_size = BATCH_SIZE
    num_workers = NUM_WORKERS
    use_gray_world = USE_GRAY_WORLD

    check_manifest_leakage()

    datasets = {
        "train": SkinLesionDataset(
            manifest_path=TRAIN_MANIFEST,
            image_dir=TRAIN_IMAGE_DIR,
            mask_dir=TRAIN_MASK_DIR,
            transform=get_train_transform(image_size, use_gray_world),
            split="train",
        ),
        "val": SkinLesionDataset(
            manifest_path=VAL_MANIFEST,
            image_dir=VAL_IMAGE_DIR,
            mask_dir=VAL_MASK_DIR,
            transform=get_val_transform(image_size, use_gray_world),
            split="val",
        ),
        "test": SkinLesionDataset(
            manifest_path=TEST_MANIFEST,
            image_dir=TEST_IMAGE_DIR,
            mask_dir=TEST_MASK_DIR,
            transform=get_test_transform(image_size, use_gray_world),
            split="test",
        ),
    }

    check_dataset_leakage(datasets, check_content_hash)

    def make(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )

    return (
        make(datasets["train"], True),
        make(datasets["val"], False),
        make(datasets["test"], False),
    )


# 4. LOAD DATA
if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_dataloaders()

    print(f"\nTrain: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"Val  : {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    print(f"Test : {len(test_loader.dataset)} samples, {len(test_loader)} batches")

    images, masks, image_ids = next(iter(train_loader))
    print("\nTrain batch:")
    print("  images:", tuple(images.shape), images.dtype)
    print("  masks :", tuple(masks.shape), masks.dtype, masks.unique().tolist())
    print("  ids   :", list(image_ids)[:3])