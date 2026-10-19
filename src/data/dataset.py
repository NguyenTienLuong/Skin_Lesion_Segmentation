"""PyTorch Dataset for ISIC 2018 lesion segmentation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ISICSegmentationDataset(Dataset):
    """Read paired RGB images and binary masks.

    ``records`` must contain ``image_id``, ``image_path`` and ``mask_path``.
    Splitting and shuffling intentionally live in ``loader.py`` so this class
    behaves like a regular PyTorch ``Dataset``.
    """

    def __init__(
        self,
        records: Sequence[Mapping],
        transform=None,
        mask_threshold: int = 127,
    ) -> None:
        if not records:
            raise ValueError("ISICSegmentationDataset requires at least one record")
        self.records = [dict(record) for record in records]
        self.transform = transform
        self.mask_threshold = int(mask_threshold)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image_path = Path(record["image_path"])
        mask_path = Path(record["mask_path"])

        try:
            with Image.open(image_path) as source:
                image = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        except Exception as exc:
            raise ValueError(f"Cannot read image {image_path}") from exc

        try:
            with Image.open(mask_path) as source:
                mask = np.asarray(source.convert("L"), dtype=np.uint8).copy()
        except Exception as exc:
            raise ValueError(f"Cannot read mask {mask_path}") from exc

        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image/mask size mismatch for {record['image_id']}: "
                f"image={image.shape[:2]}, mask={mask.shape}"
            )

        mask = (mask > self.mask_threshold).astype(np.uint8)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.asarray(mask).copy())
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        image = image.float().contiguous()
        mask = (mask > 0).float().contiguous()
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Invalid transformed image shape: {tuple(image.shape)}")
        if mask.ndim != 3 or mask.shape[0] != 1:
            raise ValueError(f"Invalid transformed mask shape: {tuple(mask.shape)}")
        return image, mask


# Keep the old import name; its constructor now follows the records-based API.
SkinLesionDataset = ISICSegmentationDataset


__all__ = ["ISICSegmentationDataset", "SkinLesionDataset"]
