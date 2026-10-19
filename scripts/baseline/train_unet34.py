#!/usr/bin/env python
"""Train the baseline UNet34 with a deterministic runtime train/val split."""

from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.data.loader import discover_kaggle_records, make_loader, split_records
from src.metrics.segmentation_metrics import evaluate
from src.models.unet34 import UNet34
from src.training.losses import build_loss
from src.training.trainer import train_resolution
from src.utils.checkpoint import save_checkpoint
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline/unet34.yaml")
    parser.add_argument("--tag", default="baseline")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg.get("data", {})
    all_records = discover_kaggle_records(
        "train",
        data_root=data_cfg.get("data_root"),
    )
    train_records, val_records = split_records(
        all_records,
        val_fraction=float(data_cfg.get("val_fraction", 0.2)),
        seed=seed,
    )
    print(
        f"runtime split: train={len(train_records)}, val={len(val_records)}, "
        f"seed={seed}"
    )

    model = UNet34(pretrained=cfg["model"].get("pretrained", True)).to(device)
    loss_fn = build_loss(cfg["loss"])
    num_workers = int(data_cfg.get("num_workers", 2))
    pin_memory = device.type == "cuda"

    for size in cfg["training"]["sizes"]:
        batch_size = int(cfg["training"]["batch_size"])
        train_loader = make_loader(
            train_records,
            size=int(size),
            train=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )
        val_loader = make_loader(
            val_records,
            size=int(size),
            train=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )
        train_resolution(
            model,
            train_loader,
            val_loader,
            loss_fn,
            cfg["optimizer"],
            device,
            frozen_epochs=int(cfg["training"].get("frozen_epochs", 30)),
            unfrozen_epochs=int(cfg["training"].get("unfrozen_epochs", 30)),
            lr_finder_cfg=cfg["training"].get("lr_finder", {}),
            scheduler_cfg=cfg["training"].get("scheduler", {}),
            selection_metric=cfg["training"].get("selection_metric", "jaccard"),
            tag=f"{args.tag} {int(size)}px",
        )

    final_size = max(int(size) for size in cfg["training"]["sizes"])
    final_val_loader = make_loader(
        val_records,
        size=final_size,
        train=False,
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )
    metrics = evaluate([model], final_val_loader, device, include_hd95=True)

    checkpoint_dir = Path(
        cfg.get("artifacts", {}).get("checkpoint_dir", "checkpoints/baseline/unet34")
    )
    experiment_dir = Path(
        cfg.get("artifacts", {}).get("experiment_dir", "experiments/baseline/unet34")
    )
    checkpoint_path = checkpoint_dir / f"{args.tag}_best.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        seed=seed,
        train_samples=len(train_records),
        val_samples=len(val_records),
        metrics=metrics,
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = experiment_dir / f"{args.tag}_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"checkpoint: {checkpoint_path}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
