"""End-to-end, in-memory smoke test for the real UNet34 training pipeline.

The raw synthetic RGB pixels are generated in [0, 1], then normalized with
ImageNet mean/std because the ResNet34 encoder is pretrained on ImageNet.
No image, mask, checkpoint, or metric file is written to disk.

Bash, full default run (1,000 samples):
    python -m pytest tests/test_smoke.py -q -s

Bash, short run without downloading pretrained weights:
    SMOKE_NUM_SAMPLES=2 SMOKE_PRETRAINED=0 python -m pytest tests/test_smoke.py -q -s
"""

from __future__ import annotations

import math
import os

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.metrics.segmentation_metrics import evaluate
from src.models.unet34 import UNet34
from src.training.losses import build_loss
from src.training.trainer import train_resolution
from src.utils.device import prepare_model, resolve_device
from src.utils.seed import set_seed


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name, "1" if default else "0").strip().lower()
    if value not in {"0", "1", "false", "true", "no", "yes"}:
        raise ValueError(f"{name} must be a boolean flag, got {value!r}")
    return value in {"1", "true", "yes"}


class SyntheticSegmentationDataset(Dataset):
    """Generate deterministic image/mask pairs lazily and only in memory."""

    def __init__(self, count: int, size: int = 64, seed: int = 42) -> None:
        if count < 2:
            raise ValueError("Smoke test requires at least two samples")
        if size < 64 or size % 32:
            raise ValueError("Image size must be >= 64 and divisible by 32")
        self.count = count
        self.size = size
        self.seed = seed
        axis = torch.linspace(-1.0, 1.0, size)
        self.yy, self.xx = torch.meshgrid(axis, axis, indexing="ij")

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        center_x = float(torch.empty(1).uniform_(-0.25, 0.25, generator=generator))
        center_y = float(torch.empty(1).uniform_(-0.25, 0.25, generator=generator))
        radius_x = float(torch.empty(1).uniform_(0.25, 0.50, generator=generator))
        radius_y = float(torch.empty(1).uniform_(0.20, 0.45, generator=generator))
        mask = (
            ((self.xx - center_x) / radius_x).square()
            + ((self.yy - center_y) / radius_y).square()
            <= 1.0
        ).float()

        skin = torch.tensor((0.76, 0.57, 0.48)).view(3, 1, 1)
        lesion = torch.tensor((0.24, 0.13, 0.10)).view(3, 1, 1)
        noise = torch.randn((3, self.size, self.size), generator=generator) * 0.025
        raw_image = (skin * (1.0 - mask) + lesion * mask + noise).clamp(0.0, 1.0)
        normalized_image = (raw_image - IMAGENET_MEAN) / IMAGENET_STD
        return normalized_image, mask.unsqueeze(0)


class SmokeUNet34(UNet34):
    """Expose the encoder blocks expected by the staged training pipeline."""

    def encoder_modules(self):
        return (self.stem, self.layer1, self.layer2, self.layer3, self.layer4)


def _device() -> torch.device:
    requested = os.getenv("SMOKE_DEVICE", "auto").lower()
    if requested == "gpu":
        requested = "cuda"
    return resolve_device(requested)


def test_training_pipeline_end_to_end() -> None:
    sample_count = _env_int("SMOKE_NUM_SAMPLES", 1000, minimum=2)
    image_size = _env_int("SMOKE_IMAGE_SIZE", 64, minimum=64)
    frozen_epochs = _env_int("SMOKE_FROZEN_EPOCHS", 1)
    unfrozen_epochs = _env_int("SMOKE_UNFROZEN_EPOCHS", 1)
    lr_iterations = _env_int("SMOKE_LR_ITERATIONS", 5, minimum=3)
    requested_batch_size = _env_int("SMOKE_BATCH_SIZE", 8)
    pretrained = _env_flag("SMOKE_PRETRAINED", True)
    data_parallel = _env_flag("SMOKE_DATA_PARALLEL", True)
    device = _device()
    set_seed(42)

    dataset = SyntheticSegmentationDataset(count=sample_count, size=image_size)
    validation_count = max(1, round(sample_count * 0.2))
    training_count = sample_count - validation_count
    train_dataset, validation_dataset = random_split(
        dataset,
        (training_count, validation_count),
        generator=torch.Generator().manual_seed(42),
    )
    batch_size = min(requested_batch_size, training_count)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    model = prepare_model(
        SmokeUNet34(pretrained=pretrained),
        device,
        data_parallel=data_parallel,
    )
    loss_name = os.getenv("SMOKE_LOSS", "Combo")
    loss_fn = build_loss({"name": loss_name, "alpha": 0.5})
    history = train_resolution(
        model,
        train_loader,
        validation_loader,
        loss_fn,
        {"name": "Adam", "weight_decay": 0.0},
        device,
        frozen_epochs=frozen_epochs,
        unfrozen_epochs=unfrozen_epochs,
        lr_finder_cfg={
            "lr_min": 1e-6,
            "lr_max": 1e-2,
            "n_iter": lr_iterations,
            "beta": 0.9,
        },
        scheduler_cfg={"cut_frac": 0.1, "ratio": 32.0},
        selection_metric="dice",
        tag="pytest-smoke",
    )
    metrics = evaluate([model], validation_loader, device=device, include_hd95=False)

    expected_frozen_steps = frozen_epochs * len(train_loader)
    expected_unfrozen_steps = unfrozen_epochs * len(train_loader)
    assert len(history["frozen"].lr_finder_lrs) >= 3
    assert len(history["unfrozen"].lr_finder_lrs) >= 3
    assert len(history["frozen"].lr_trace) == expected_frozen_steps
    assert len(history["unfrozen"].lr_trace) == expected_unfrozen_steps
    assert set(metrics) == {"jaccard", "threshold_jaccard", "dice", "sensitivity"}
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in metrics.values())

    print(
        "smoke passed: "
        f"device={device}, gpu_count={torch.cuda.device_count()}, "
        f"data_parallel={isinstance(model, torch.nn.DataParallel)}, "
        f"pretrained={pretrained}, samples={sample_count}, "
        f"train={training_count}, val={validation_count}, size={image_size}, "
        f"frozen_epochs={frozen_epochs}, unfrozen_epochs={unfrozen_epochs}, "
        f"frozen_lr={history['frozen'].lr:.3e}, "
        f"unfrozen_lr={history['unfrozen'].lr:.3e}, metrics={metrics}"
    )
