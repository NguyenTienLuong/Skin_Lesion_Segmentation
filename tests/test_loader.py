import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import src.data.loader as loader_module
from src.data.loader import (
    discover_isic_records,
    discover_kaggle_records,
    make_loader,
    split_records,
)


SPLIT_DIRECTORIES = {
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


def _write_image(path: Path, value: int, height: int = 11, width: int = 13) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, columns = np.indices((height, width))
    image = np.stack(
        (
            (rows * 17 + columns * 3 + value) % 256,
            (rows * 5 + columns * 23 + value * 2) % 256,
            (rows * 29 + columns * 7 + value * 3) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(image).save(path)


def _write_mask(path: Path, value: int, height: int = 11, width: int = 13) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((height, width), dtype=np.uint8)
    row_stop = 2 + value % max(1, height - 2)
    column_start = value % max(1, width - 2)
    mask[1:row_stop, column_start:column_start + 2] = 255
    Image.fromarray(mask).save(path)


def _write_official_pair(
    root: Path,
    split: str,
    image_id: str,
    value: int = 1,
) -> tuple[Path, Path]:
    image_dir_name, mask_dir_name = SPLIT_DIRECTORIES[split]
    image_path = root / image_dir_name / f"{image_id}.jpg"
    mask_path = root / mask_dir_name / f"{image_id}_segmentation.png"
    _write_image(image_path, value=value)
    _write_mask(mask_path, value=value)
    return image_path, mask_path


def _make_records(root: Path, count: int) -> list[dict]:
    records = []
    for index in range(count):
        image_id = f"ISIC_{index:07d}"
        image_path = root / "images" / f"{image_id}.png"
        mask_path = root / "masks" / f"{image_id}_segmentation.png"
        _write_image(image_path, value=index + 1)
        _write_mask(mask_path, value=index + 1)
        records.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )
    return records


def _ids(records: list[dict]) -> list[str]:
    return [record["image_id"] for record in records]


def test_discover_isic_records_pairs_images_and_masks_by_id(tmp_path):
    second_image, second_mask = _write_official_pair(
        tmp_path, "train", "ISIC_0000002", value=2
    )
    first_image, first_mask = _write_official_pair(
        tmp_path, "train", "ISIC_0000001", value=1
    )
    (tmp_path / SPLIT_DIRECTORIES["train"][0] / "README.txt").write_text(
        "ignored", encoding="utf-8"
    )

    records = discover_isic_records(data_root=tmp_path, split="train")

    assert records["image_id"].tolist() == ["ISIC_0000001", "ISIC_0000002"]
    assert records["image_path"].tolist() == [
        str(first_image.resolve()),
        str(second_image.resolve()),
    ]
    assert records["mask_path"].tolist() == [
        str(first_mask.resolve()),
        str(second_mask.resolve()),
    ]


def test_discover_isic_records_rejects_an_unpaired_image(tmp_path):
    image_dir_name, mask_dir_name = SPLIT_DIRECTORIES["train"]
    _write_image(tmp_path / image_dir_name / "ISIC_0000003.jpg", value=3)
    (tmp_path / mask_dir_name).mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="no matching segmentation mask"):
        discover_isic_records(data_root=tmp_path, split="train")


def test_discovery_accepts_direct_named_split_directories(tmp_path):
    image_root = tmp_path / SPLIT_DIRECTORIES["validation"][0]
    mask_root = tmp_path / SPLIT_DIRECTORIES["validation"][1]
    image_id = "ISIC_0000100"
    image_path = image_root / f"{image_id}.jpg"
    mask_path = mask_root / f"{image_id}_segmentation.png"
    _write_image(image_path, value=4)
    _write_mask(mask_path, value=4)

    records = discover_isic_records(
        split="validation",
        image_root=image_root,
        mask_root=mask_root,
    )

    assert records["image_id"].tolist() == [image_id]
    assert records.loc[0, "image_path"] == str(image_path.resolve())
    assert records.loc[0, "mask_path"] == str(mask_path.resolve())


def test_runtime_split_is_deterministic_disjoint_and_exhaustive(tmp_path):
    records = _make_records(tmp_path, count=12)

    train_a, val_a = split_records(records, val_fraction=0.25, seed=123)
    train_b, val_b = split_records(records, val_fraction=0.25, seed=123)
    train_c, val_c = split_records(records, val_fraction=0.25, seed=456)

    assert _ids(train_a) == _ids(train_b)
    assert _ids(val_a) == _ids(val_b)
    assert set(_ids(train_a)).isdisjoint(_ids(val_a))
    assert set(_ids(train_a) + _ids(val_a)) == set(_ids(records))
    assert len(train_a) == 9
    assert len(val_a) == 3
    assert _ids(val_a) != _ids(val_c)
    assert set(_ids(train_c) + _ids(val_c)) == set(_ids(records))


def test_make_loader_keeps_last_batch_by_default_and_can_drop_it(tmp_path):
    records = _make_records(tmp_path, count=5)

    complete_loader = make_loader(
        records,
        size=16,
        train=True,
        batch_size=2,
        seed=7,
    )
    dropped_loader = make_loader(
        records,
        size=16,
        train=True,
        batch_size=2,
        seed=7,
        drop_last=True,
    )

    assert complete_loader.drop_last is False
    assert [images.shape[0] for images, _ in complete_loader] == [2, 2, 1]
    assert dropped_loader.drop_last is True
    assert [images.shape[0] for images, _ in dropped_loader] == [2, 2]


@pytest.mark.parametrize("num_workers", [0, 2])
def test_make_loader_reproduces_shuffle_and_augmentation_with_same_seed(
    tmp_path, num_workers
):
    records = _make_records(tmp_path, count=8)
    kwargs = {
        "size": 16,
        "train": True,
        "batch_size": 4,
        "num_workers": num_workers,
        "seed": 918,
    }

    images_a, masks_a = next(iter(make_loader(records, **kwargs)))
    images_b, masks_b = next(iter(make_loader(records, **kwargs)))

    assert torch.equal(images_a, images_b)
    assert torch.equal(masks_a, masks_b)


def test_kaggle_source_policy_uses_local_masks_for_validation_and_test(
    tmp_path, monkeypatch
):
    kaggle_root = tmp_path / "kaggle"
    local_root = tmp_path / "local"
    train_image, train_mask = _write_official_pair(
        kaggle_root, "train", "ISIC_0000200", value=1
    )

    expected = {"train": (train_image, train_mask)}
    for offset, split in enumerate(("validation", "test"), start=2):
        image_id = f"ISIC_000020{offset}"
        image_dir_name, mask_dir_name = SPLIT_DIRECTORIES[split]
        image_path = kaggle_root / image_dir_name / f"{image_id}.jpg"
        mask_path = local_root / mask_dir_name / f"{image_id}_segmentation.png"
        _write_image(image_path, value=offset)
        _write_mask(mask_path, value=offset)
        expected[split] = (image_path, mask_path)

    resolve_calls = []

    def fake_resolve(data_root=None):
        resolve_calls.append(data_root)
        return kaggle_root

    monkeypatch.setattr(loader_module, "resolve_kaggle_data_root", fake_resolve)
    monkeypatch.setattr(loader_module, "LOCAL_DATA_ROOT", local_root)

    for split in ("train", "validation", "test"):
        records = discover_kaggle_records(split)
        expected_image, expected_mask = expected[split]
        assert records.loc[0, "image_path"] == str(expected_image.resolve())
        assert records.loc[0, "mask_path"] == str(expected_mask.resolve())

    assert resolve_calls == [None, None, None]


def test_environment_supports_multiple_data_roots(tmp_path, monkeypatch):
    image_root = tmp_path / "image-source"
    mask_root = tmp_path / "mask-source"
    image_dir_name, mask_dir_name = SPLIT_DIRECTORIES["train"]
    image_id = "ISIC_0000300"
    image_path = image_root / image_dir_name / f"{image_id}.jpg"
    mask_path = mask_root / mask_dir_name / f"{image_id}_segmentation.png"
    _write_image(image_path, value=3)
    _write_mask(mask_path, value=3)

    monkeypatch.delenv("ISIC2018_KAGGLE_ROOT", raising=False)
    monkeypatch.setenv(
        "ISIC2018_DATA_ROOT",
        os.pathsep.join((str(image_root), str(mask_root))),
    )
    monkeypatch.setattr(loader_module, "KAGGLE_DATA_ROOT", tmp_path / "not-mounted")

    records = discover_kaggle_records("train")

    assert records["image_id"].tolist() == [image_id]
    assert records.loc[0, "image_path"] == str(image_path.resolve())
    assert records.loc[0, "mask_path"] == str(mask_path.resolve())


def test_resolver_uses_kagglehub_only_when_no_existing_root(tmp_path, monkeypatch):
    downloaded_root = tmp_path / "kagglehub-cache"
    downloaded_root.mkdir()
    calls = []

    def fake_download(handle):
        calls.append(handle)
        return str(downloaded_root)

    monkeypatch.delenv("ISIC2018_KAGGLE_ROOT", raising=False)
    monkeypatch.delenv("ISIC2018_DATA_ROOT", raising=False)
    monkeypatch.setattr(loader_module, "KAGGLE_DATA_ROOT", tmp_path / "not-mounted")
    monkeypatch.setitem(
        sys.modules,
        "kagglehub",
        SimpleNamespace(dataset_download=fake_download),
    )

    resolved = loader_module.resolve_kaggle_data_root()

    assert resolved == downloaded_root.resolve()
    assert calls == [loader_module.KAGGLE_DATASET_HANDLE]
