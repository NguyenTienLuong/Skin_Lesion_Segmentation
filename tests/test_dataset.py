from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import ISICSegmentationDataset


def _save_rgb(path: Path, height: int, width: int) -> None:
    rows, columns = np.indices((height, width))
    image = np.stack(
        (
            (rows * 31 + columns * 7) % 256,
            (rows * 13 + columns * 19) % 256,
            (rows * 5 + columns * 29) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(image).save(path)


def test_dataset_returns_chw_float_image_and_binary_mask(tmp_path):
    image_path = tmp_path / "ISIC_0000001.jpg"
    mask_path = tmp_path / "ISIC_0000001_segmentation.png"
    _save_rgb(image_path, height=3, width=4)
    mask_values = np.array(
        [
            [0, 127, 128, 255],
            [255, 128, 127, 0],
            [0, 255, 0, 255],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(mask_values).save(mask_path)

    dataset = ISICSegmentationDataset(
        [
            {
                "image_id": "ISIC_0000001",
                "image_path": image_path,
                "mask_path": mask_path,
            }
        ]
    )

    image, mask = dataset[0]

    assert image.shape == (3, 3, 4)
    assert mask.shape == (1, 3, 4)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert set(torch.unique(mask).tolist()) == {0.0, 1.0}
    assert mask[0, 0].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_dataset_rejects_image_mask_size_mismatch(tmp_path):
    image_path = tmp_path / "ISIC_0000002.jpg"
    mask_path = tmp_path / "ISIC_0000002_segmentation.png"
    _save_rgb(image_path, height=5, width=6)
    Image.fromarray(np.zeros((4, 6), dtype=np.uint8)).save(mask_path)
    dataset = ISICSegmentationDataset(
        [
            {
                "image_id": "ISIC_0000002",
                "image_path": image_path,
                "mask_path": mask_path,
            }
        ]
    )

    with pytest.raises(ValueError, match="Image/mask size mismatch"):
        dataset[0]
