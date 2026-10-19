"""PyTorch DataLoader utilities for ISIC 2018 Task 1 segmentation.

The loader accepts a path returned by ``kagglehub.dataset_download``. When no
Kaggle root is supplied, it reuses the Kaggle Notebook mount if available and
otherwise downloads the dataset through KaggleHub. Image and mask directories
may live under different roots; this is useful when Kaggle provides the images
while ground-truth masks live in the project's local ``data/`` directory.
Train/validation records can be split deterministically in memory, so CSV
split files are optional.
"""

from __future__ import annotations

import os
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, get_worker_info

from src.data.dataset import ISICSegmentationDataset


KAGGLE_DATASET_HANDLE = "tschandl/isic2018-challenge-task1-data-segmentation"
KAGGLE_DATA_ROOT = Path("/kaggle/input/isic2018-challenge-task1-data-segmentation")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATA_ROOT = PROJECT_ROOT / "data"

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


def _root_values(value) -> list[str | os.PathLike]:
    """Flatten one/many roots, including ``os.pathsep``-separated strings."""
    if isinstance(value, str):
        return [part.strip() for part in value.split(os.pathsep) if part.strip()]
    if isinstance(value, os.PathLike):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        roots: list[str | os.PathLike] = []
        for item in value:
            roots.extend(_root_values(item))
        return roots
    raise TypeError(
        "A data root must be a path, a path-list string, or a sequence of paths"
    )


def _configured_root_values() -> list[str | os.PathLike]:
    """Return configured roots with Kaggle-specific roots taking priority."""
    values: list[str | os.PathLike] = []
    for variable in ("ISIC2018_KAGGLE_ROOT", "ISIC2018_DATA_ROOT"):
        configured = os.getenv(variable)
        if configured:
            values.extend(_root_values(configured))
    return values


def _one_or_many_roots(values: list[str | os.PathLike]):
    """Keep the common single-root return shape while preserving many roots."""
    if len(values) == 1:
        return values[0]
    return tuple(values)


def resolve_kaggle_data_root(data_root=None):
    """Resolve the Kaggle root and download it only when necessary.

    Resolution order:

    1. an explicit ``data_root`` (including the path returned by KaggleHub),
    2. ``ISIC2018_KAGGLE_ROOT``, followed by ``ISIC2018_DATA_ROOT``,
    3. the standard Kaggle Notebook mount,
    4. ``kagglehub.dataset_download`` for Colab/local environments.

    Explicit roots and either environment variable may contain multiple paths,
    either as a sequence or separated by :data:`os.pathsep`. A single root is
    returned unchanged; multiple roots are returned as a tuple in priority
    order.
    """
    if data_root is not None:
        explicit_roots = _root_values(data_root)
        if not explicit_roots:
            raise ValueError("data_root does not contain a usable path")
        return _one_or_many_roots(explicit_roots)

    configured_roots = _configured_root_values()
    if configured_roots:
        return _one_or_many_roots(configured_roots)
    if KAGGLE_DATA_ROOT.is_dir():
        return KAGGLE_DATA_ROOT.resolve()

    try:
        import kagglehub
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Cannot find the ISIC 2018 Kaggle dataset. Install KaggleHub with "
            "`pip install kagglehub`, or pass the path returned by "
            "`kagglehub.dataset_download(...)` as data_root."
        ) from exc

    downloaded_root = Path(
        kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)
    ).expanduser().resolve()
    if not downloaded_root.is_dir():
        raise FileNotFoundError(
            f"KaggleHub returned a missing dataset directory: {downloaded_root}"
        )
    return downloaded_root


def _search_roots(data_root=None) -> list[Path]:
    """Return existing roots in priority order without duplicating paths."""
    values: list[str | os.PathLike] = []
    if data_root is not None:
        values.extend(_root_values(data_root))
    else:
        values.extend(_configured_root_values())
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
    for root in roots:
        if root.is_dir() and root.name == name:
            return root.resolve()
        direct = root / name
        if direct.is_dir():
            return direct.resolve()
        matches = [path for path in root.rglob(name) if path.is_dir()]
        if matches:
            return min(
                matches,
                key=lambda path: (len(path.parts), str(path)),
            ).resolve()
    return None


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


def _resolve_record_path(value, field: str, base_dir: Path) -> Path:
    if pd.isna(value):
        raise ValueError(f"Record has a missing {field}")
    path = Path(str(value).strip().replace("\\", os.sep)).expanduser()
    if not path.is_absolute():
        relative_to_csv = base_dir / path
        relative_to_cwd = Path.cwd() / path
        path = relative_to_csv if relative_to_csv.exists() else relative_to_cwd
    return path.resolve()


def _coerce_records(records, source_split: str | None = None) -> list[dict]:
    base_dir = Path.cwd()
    if isinstance(records, (str, os.PathLike)):
        path = Path(records)
        if path.is_file() and path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, dtype={"image_id": str})
            base_dir = path.resolve().parent
        elif path.is_dir():
            if source_split is None:
                raise ValueError(
                    "source_split is required when records is a data directory"
                )
            frame = discover_isic_records(path, split=source_split)
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

    if frame.empty:
        raise ValueError("No valid image-mask records were provided")

    output: list[dict] = []
    seen_ids: set[str] = set()
    for index, row in frame.reset_index(drop=True).iterrows():
        image_path = _resolve_record_path(row["image_path"], "image_path", base_dir)
        mask_path = _resolve_record_path(row["mask_path"], "mask_path", base_dir)
        raw_image_id = row.get("image_id", image_path.stem)
        if pd.isna(raw_image_id):
            raise ValueError(f"Record {index} has a missing image_id")
        image_id = str(raw_image_id).strip()
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


def _build_transform(size: int, train: bool, seed: int):
    try:
        from src.data.data_pipeline import get_train_transform, get_val_transform
    except ImportError as exc:
        raise RuntimeError(
            "The data pipeline requires albumentations and OpenCV. "
            "Install the project dependencies with `pip install -r requirements.txt`."
        ) from exc

    if train:
        return get_train_transform(size=size, use_gray_world=True, seed=seed)
    return get_val_transform(size=size, use_gray_world=True, seed=seed)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    worker = get_worker_info()
    if worker is not None:
        transform = getattr(worker.dataset, "transform", None)
        if hasattr(transform, "set_random_seed"):
            transform.set_random_seed(seed)


def make_loader(
    records,
    size: int,
    train: bool,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    source_split: str | None = None,
    drop_last: bool = False,
) -> DataLoader:
    """Wrap one already-selected record set in a PyTorch ``DataLoader``.

    This function never filters a ``split`` column. Use ``split_records`` or
    ``make_train_val_loaders`` when the train/validation split should be made
    at runtime. ``drop_last`` defaults to ``False`` so every selected sample is
    seen, including the final incomplete training batch.
    """
    if int(size) <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if int(num_workers) < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")

    if source_split is None and isinstance(records, (str, os.PathLike)):
        if Path(records).is_dir():
            source_split = "train" if train else "validation"
    samples = _coerce_records(records, source_split=source_split)
    transform = _build_transform(int(size), bool(train), int(seed))
    dataset = ISICSegmentationDataset(samples, transform=transform)
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(train),
        drop_last=bool(drop_last),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=int(num_workers) > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def split_records(
    records,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Deterministically split records in memory without writing CSV files."""
    fraction = float(val_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    samples = _coerce_records(records, source_split="train")
    if len(samples) < 2:
        raise ValueError("At least two samples are required for a train/validation split")

    val_count = round(len(samples) * fraction)
    val_count = min(max(1, val_count), len(samples) - 1)
    permutation = torch.randperm(
        len(samples),
        generator=torch.Generator().manual_seed(int(seed)),
    ).tolist()
    val_indices = set(permutation[:val_count])
    train_records = [sample for index, sample in enumerate(samples) if index not in val_indices]
    val_records = [sample for index, sample in enumerate(samples) if index in val_indices]
    return train_records, val_records


def kfold_records(
    records,
    n_splits: int = 3,
    seed: int = 42,
) -> list[tuple[list[dict], list[dict]]]:
    """Create deterministic K-fold train/validation record pairs in memory."""
    samples = _coerce_records(records, source_split="train")
    folds = int(n_splits)
    if folds < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    if folds > len(samples):
        raise ValueError(
            f"n_splits ({folds}) cannot exceed sample count ({len(samples)})"
        )

    permutation = torch.randperm(
        len(samples),
        generator=torch.Generator().manual_seed(int(seed)),
    ).tolist()
    fold_sizes = [len(samples) // folds] * folds
    for index in range(len(samples) % folds):
        fold_sizes[index] += 1

    output: list[tuple[list[dict], list[dict]]] = []
    offset = 0
    for fold_size in fold_sizes:
        val_indices = set(permutation[offset:offset + fold_size])
        train_records = [
            sample for index, sample in enumerate(samples) if index not in val_indices
        ]
        val_records = [
            sample for index, sample in enumerate(samples) if index in val_indices
        ]
        output.append((train_records, val_records))
        offset += fold_size
    return output


def make_train_val_loaders(
    records,
    size: int,
    batch_size: int,
    val_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Split records and build loaders; ``drop_last`` affects training only."""
    train_records, val_records = split_records(
        records,
        val_fraction=val_fraction,
        seed=seed,
    )
    train_loader = make_loader(
        train_records,
        size=size,
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        drop_last=drop_last,
    )
    val_loader = make_loader(
        val_records,
        size=size,
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )
    return train_loader, val_loader


def discover_kaggle_records(
    split: str,
    data_root=None,
    image_root=None,
    mask_root=None,
) -> pd.DataFrame:
    """Discover records using the project's Kaggle/local source policy.

    Source policy when explicit component roots are omitted:

    - ``train``: images and masks both come from the Kaggle dataset root.
    - ``validation``/``test``: images come from Kaggle, while masks come from
      the local ``data/`` directory.

    ``data_root`` may be the path returned by ``kagglehub.dataset_download``.
    If omitted, the root is resolved by :func:`resolve_kaggle_data_root`.
    Pass ``image_root`` or ``mask_root`` only when overriding one side.
    """
    canonical_split = _canonical_split(split)
    needs_kaggle_root = (
        image_root is None
        or (canonical_split == "train" and mask_root is None)
    )
    kaggle_root = (
        resolve_kaggle_data_root(data_root) if needs_kaggle_root else None
    )
    selected_image_root = image_root if image_root is not None else kaggle_root
    if mask_root is not None:
        selected_mask_root = mask_root
    elif canonical_split == "train":
        selected_mask_root = kaggle_root
    else:
        selected_mask_root = LOCAL_DATA_ROOT

    return discover_isic_records(
        split=canonical_split,
        image_root=selected_image_root,
        mask_root=selected_mask_root,
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
    seed: int = 42,
    drop_last: bool = False,
) -> DataLoader:
    """Create one official ISIC split loader from Kaggle/local sources."""
    canonical_split = _canonical_split(split)
    records = discover_kaggle_records(
        split=canonical_split,
        data_root=data_root,
        image_root=image_root,
        mask_root=mask_root,
    )
    return make_loader(
        records,
        size=size,
        train=canonical_split == "train",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        drop_last=drop_last,
    )


def make_kaggle_train_val_loaders(
    size: int,
    batch_size: int,
    val_fraction: float = 0.2,
    seed: int = 42,
    data_root=None,
    image_root=None,
    mask_root=None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Split Kaggle training records at runtime and return two loaders."""
    records = discover_kaggle_records(
        split="train",
        data_root=data_root,
        image_root=image_root,
        mask_root=mask_root,
    )
    return make_train_val_loaders(
        records,
        size=size,
        batch_size=batch_size,
        val_fraction=val_fraction,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


__all__ = [
    "KAGGLE_DATASET_HANDLE",
    "ISICSegmentationDataset",
    "discover_isic_records",
    "discover_kaggle_records",
    "kfold_records",
    "make_kaggle_loader",
    "make_kaggle_train_val_loaders",
    "make_loader",
    "make_train_val_loaders",
    "resolve_kaggle_data_root",
    "split_records",
]
