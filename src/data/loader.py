"""DataLoader utilities for the ISIC 2018 Task 1 segmentation dataset.

The loader supports both the Kaggle mount
``/kaggle/input/isic2018-challenge-task1-data-segmentation`` and a local
``data/`` directory. Image and mask directories may live under different
roots; this is useful when Kaggle provides the images while ground-truth masks
have been copied into the local project.
"""

from __future__ import annotations

import os
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


KAGGLE_DATA_ROOT = Path("/kaggle/input/isic2018-challenge-task1-data-segmentation")
LOCAL_DATA_ROOT = Path("data")

_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
    "heldout": "test",
}

_SPLIT_DIRECTORIES = {
    "train": (
        "ISIC2018_Task1-2_Training_Input",
        "ISIC2018_Task1_Training_GroundTruth",
    ),
    "validation": (
        "ISIC2018_Task1-2_Validation_Input",
        "ISIC2018_Task1_Validation_GroundTruth",
    ),
    "test": (
        "ISIC2018_Task1-2_Test_Input",
        "ISIC2018_Task1_Test_GroundTruth",
    ),
}

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_TRUE_VALUES = {"true", "1", "yes", "y"}


def _canonical_split(split: str) -> str:
    key = str(split).strip().lower()
    try:
        return _SPLIT_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown split {split!r}; expected train, validation, or test"
        ) from exc


def _search_roots(data_root=None) -> list[Path]:
    """Return existing roots in priority order without duplicating paths."""
    values: list[str | os.PathLike] = []
    if data_root is not None:
        if isinstance(data_root, (str, os.PathLike)):
            values.append(data_root)
        else:
            values.extend(data_root)
    else:
        configured = os.getenv("ISIC2018_DATA_ROOT")
        if configured:
            values.extend(part for part in configured.split(os.pathsep) if part)
        values.extend((KAGGLE_DATA_ROOT, LOCAL_DATA_ROOT))

    roots: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        root = Path(value).expanduser().resolve()
        if root in seen:
            continue
        seen.add(root)
        if root.is_dir():
            roots.append(root)

    if not roots:
        expected = ", ".join(str(Path(value)) for value in values)
        raise FileNotFoundError(f"No ISIC data root exists. Checked: {expected}")
    return roots


def _find_named_directory(roots: Sequence[Path], name: str) -> Path | None:
    """Find an exact ISIC directory directly or inside a Kaggle nesting layer."""
    direct = [root / name for root in roots if (root / name).is_dir()]
    if direct:
        return direct[0].resolve()

    matches: list[Path] = []
    for root in roots:
        matches.extend(path for path in root.rglob(name) if path.is_dir())
    if not matches:
        return None
    return min(matches, key=lambda path: (len(path.parts), str(path))).resolve()


def discover_isic_records(
    data_root=None,
    split: str = "train",
    *,
    image_root=None,
    mask_root=None,
) -> pd.DataFrame:
    """Discover and strictly pair Kaggle/local ISIC images with their masks.

    ``data_root`` may be one path or a sequence of paths. If omitted, the
    function searches ``ISIC2018_DATA_ROOT``, the standard Kaggle mount, and
    local ``data/``. Pass ``image_root`` and ``mask_root`` to force images and
    masks to come from different sources, for example Kaggle images plus local
    masks. Explicit roots take precedence over ``data_root`` for that side.
    """
    split = _canonical_split(split)
    shared_roots = None
    if image_root is None or mask_root is None:
        shared_roots = _search_roots(data_root)
    image_roots = _search_roots(image_root) if image_root is not None else shared_roots
    mask_roots = _search_roots(mask_root) if mask_root is not None else shared_roots
    image_dir_name, mask_dir_name = _SPLIT_DIRECTORIES[split]
    image_dir = _find_named_directory(image_roots, image_dir_name)
    mask_dir = _find_named_directory(mask_roots, mask_dir_name)

    missing = []
    if image_dir is None:
        missing.append(image_dir_name)
    if mask_dir is None:
        missing.append(mask_dir_name)
    if missing:
        image_roots_text = ", ".join(str(root) for root in image_roots)
        mask_roots_text = ", ".join(str(root) for root in mask_roots)
        raise FileNotFoundError(
            f"Missing ISIC {split} director{'y' if len(missing) == 1 else 'ies'}: "
            f"{', '.join(missing)}. Image roots: {image_roots_text}. "
            f"Mask roots: {mask_roots_text}"
        )

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not image_paths:
        raise RuntimeError(f"No input images found in {image_dir}")

    rows = []
    missing_masks = []
    for image_path in image_paths:
        mask_path = mask_dir / f"{image_path.stem}_segmentation.png"
        if not mask_path.is_file():
            missing_masks.append(str(mask_path))
            continue
        rows.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
            }
        )

    if missing_masks:
        preview = "\n  ".join(missing_masks[:5])
        raise FileNotFoundError(
            f"{len(missing_masks)} image(s) have no matching segmentation mask. "
            f"First missing paths:\n  {preview}"
        )

    expected_masks = {f"{path.stem}_segmentation.png" for path in image_paths}
    orphan_masks = sorted(
        path.name for path in mask_dir.glob("*_segmentation.png")
        if path.name not in expected_masks
    )
    if orphan_masks:
        preview = ", ".join(orphan_masks[:5])
        raise ValueError(
            f"{len(orphan_masks)} mask(s) have no matching input image in {image_dir}. "
            f"Examples: {preview}"
        )

    return pd.DataFrame(rows, columns=("image_id", "image_path", "mask_path"))


def _resolve_record_path(value, field: str) -> Path:
    if pd.isna(value):
        raise ValueError(f"Record has a missing {field}")
    path = Path(str(value).strip().replace("\\", os.sep)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _coerce_records(records, train: bool) -> list[dict]:
    if isinstance(records, (str, os.PathLike)):
        path = Path(records)
        if path.is_file() and path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, dtype={"image_id": str})
        elif path.is_dir():
            split = "train" if train else "validation"
            frame = discover_isic_records(path, split=split)
        else:
            raise FileNotFoundError(f"Records path does not exist: {path}")
    elif isinstance(records, pd.DataFrame):
        frame = records.copy()
    elif isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        frame = pd.DataFrame(list(records))
    else:
        raise TypeError(
            "records must be a DataFrame, a sequence of mappings, a CSV path, "
            "or an ISIC data directory"
        )

    required = {"image_path", "mask_path"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Records are missing columns: {sorted(missing_columns)}")

    if "pair_valid" in frame.columns:
        valid = (
            frame["pair_valid"].astype(str).str.strip().str.lower().isin(_TRUE_VALUES)
        )
        frame = frame.loc[valid].copy()

    if "split" in frame.columns:
        split_values = frame["split"].astype(str).str.strip().str.lower()
        wanted = {"train"} if train else {"val", "valid", "validation"}
        if split_values.isin(wanted).any():
            frame = frame.loc[split_values.isin(wanted)].copy()

    if frame.empty:
        raise ValueError("No image-mask records remain after validation/filtering")

    output: list[dict] = []
    seen_ids: set[str] = set()
    for index, row in frame.reset_index(drop=True).iterrows():
        image_path = _resolve_record_path(row["image_path"], "image_path")
        mask_path = _resolve_record_path(row["mask_path"], "mask_path")
        image_id = str(row.get("image_id", image_path.stem)).strip()
        if not image_id:
            raise ValueError(f"Record {index} has an empty image_id")
        if image_id in seen_ids:
            raise ValueError(f"Duplicate image_id: {image_id}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask does not exist: {mask_path}")
        seen_ids.add(image_id)
        output.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )
    return output


def _build_transform(size: int, train: bool):
    try:
        from src.data.data_pipeline import get_train_transform, get_val_transform
    except ImportError as exc:
        raise RuntimeError(
            "The data pipeline requires albumentations and OpenCV. "
            "Install the project dependencies with `pip install -r requirements.txt`."
        ) from exc

    if train:
        return get_train_transform(size=size, use_gray_world=True)
    return get_val_transform(size=size, use_gray_world=True)


class ISICSegmentationDataset(Dataset):
    """Dataset contract required by trainer/evaluator: ``(image, mask)``."""

    def __init__(self, records, transform, mask_threshold: int = 127) -> None:
        self.records = list(records)
        self.transform = transform
        self.mask_threshold = int(mask_threshold)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(record["image_path"]) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(record["mask_path"]) as source:
            mask = np.asarray(source.convert("L"), dtype=np.uint8).copy()

        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image/mask size mismatch for {record['image_id']}: "
                f"image={image.shape[:2]}, mask={mask.shape}"
            )
        mask = (mask > self.mask_threshold).astype(np.uint8)
        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"]

        if not isinstance(image_tensor, torch.Tensor):
            image_tensor = torch.from_numpy(np.asarray(image_tensor).copy()).permute(2, 0, 1)
        if not isinstance(mask_tensor, torch.Tensor):
            mask_tensor = torch.from_numpy(np.asarray(mask_tensor).copy())
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        image_tensor = image_tensor.float().contiguous()
        mask_tensor = (mask_tensor > 0).float().contiguous()
        if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
            raise ValueError(f"Invalid transformed image shape: {tuple(image_tensor.shape)}")
        if mask_tensor.ndim != 3 or mask_tensor.shape[0] != 1:
            raise ValueError(f"Invalid transformed mask shape: {tuple(mask_tensor.shape)}")
        return image_tensor, mask_tensor


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(
    records,
    size: int,
    train: bool,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build the loader used by baseline, Optuna, final training, and evaluation."""
    if int(size) <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if int(num_workers) < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")

    seed = 42
    samples = _coerce_records(records, train=bool(train))
    transform = _build_transform(int(size), bool(train))
    dataset = ISICSegmentationDataset(samples, transform=transform)
    generator = torch.Generator().manual_seed(seed)
    drop_last = bool(train) and len(dataset) >= int(batch_size)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(train),
        drop_last=drop_last,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=int(num_workers) > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def make_kaggle_loader(
    split: str,
    size: int,
    batch_size: int,
    data_root=None,
    image_root=None,
    mask_root=None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a loader using the project's Kaggle/local source policy.

    Source policy when explicit component roots are omitted:

    - ``train``: images and masks both come from the Kaggle dataset root.
    - ``validation``/``test``: images come from Kaggle, while masks come from
      the local ``data/`` directory.

    ``data_root`` denotes the Kaggle root. On Kaggle it defaults to the
    standard mounted dataset path. Pass ``image_root`` or ``mask_root`` only
    when overriding one side explicitly.
    """
    canonical_split = _canonical_split(split)
    kaggle_root = (
        data_root
        if data_root is not None
        else os.getenv("ISIC2018_KAGGLE_ROOT", str(KAGGLE_DATA_ROOT))
    )
    selected_image_root = image_root if image_root is not None else kaggle_root
    if mask_root is not None:
        selected_mask_root = mask_root
    elif canonical_split == "train":
        selected_mask_root = kaggle_root
    else:
        selected_mask_root = LOCAL_DATA_ROOT

    records = discover_isic_records(
        split=canonical_split,
        image_root=selected_image_root,
        mask_root=selected_mask_root,
    )
    return make_loader(
        records,
        size=size,
        train=canonical_split == "train",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


__all__ = [
    "ISICSegmentationDataset",
    "discover_isic_records",
    "make_kaggle_loader",
    "make_loader",
]
