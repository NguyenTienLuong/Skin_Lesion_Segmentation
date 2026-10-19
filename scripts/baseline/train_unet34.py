#!/usr/bin/env python
"""Train a 3-fold UNet34 baseline and evaluate on official validation data.

Each fold holdout is used only for Best-Dice checkpoint selection. The
official ISIC validation split is never used by LR Finder or training; it is
used after each fold has finished and for the final equal-weight ensemble.
"""

from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import yaml

from src.data.loader import (
    discover_kaggle_records,
    kfold_records,
    make_loader,
)
from src.metrics.segmentation_metrics import evaluate
from src.models.unet34 import UNet34
from src.training.losses import build_loss
from src.training.trainer import train_resolution
from src.utils.checkpoint import save_checkpoint
from src.utils.device import (
    describe_device,
    device_from_config,
    prepare_model,
    unwrap_model,
)
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline/unet34.yaml")
    parser.add_argument("--tag", default="baseline")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device, data_parallel = device_from_config(cfg)
    print(describe_device(device, data_parallel))

    data_cfg = cfg.get("data", {})
    training_records = discover_kaggle_records(
        "train",
        data_root=data_cfg.get("data_root"),
    )
    official_validation_records = discover_kaggle_records(
        "validation",
        data_root=data_cfg.get("data_root"),
        mask_root=data_cfg.get("mask_root"),
    )
    folds = kfold_records(
        training_records,
        n_splits=int(data_cfg.get("n_splits", 3)),
        seed=seed,
    )
    print(
        f"baseline CV: folds={len(folds)}, train_total={len(training_records)}, "
        f"official_val={len(official_validation_records)}, seed={seed}"
    )

    training_cfg = cfg["training"]
    selection_metric = training_cfg.get("selection_metric", "dice")
    num_workers = int(data_cfg.get("num_workers", 2))
    pin_memory = device.type == "cuda"
    final_size = max(int(size) for size in training_cfg["sizes"])
    batch_size = int(training_cfg["batch_size"])

    artifacts_cfg = cfg.get("artifacts", {})
    checkpoint_root = Path(
        artifacts_cfg.get(
            "final_checkpoint_root",
            artifacts_cfg.get("checkpoint_dir", "checkpoints/baseline/unet34"),
        )
    )
    experiment_dir = Path(
        artifacts_cfg.get(
            "final_experiment_dir",
            artifacts_cfg.get("experiment_dir", "experiments/baseline/unet34"),
        )
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    fold_models = []
    fold_results = []
    for fold_index, (fold_train_records, fold_holdout_records) in enumerate(
        folds, start=1
    ):
        fold_seed = seed + fold_index - 1
        set_seed(fold_seed)
        print(
            f"fold {fold_index}/{len(folds)}: train={len(fold_train_records)}, "
            f"selection_holdout={len(fold_holdout_records)}, seed={fold_seed}"
        )

        model = prepare_model(
            UNet34(pretrained=cfg["model"].get("pretrained", True)),
            device,
            data_parallel=data_parallel,
        )
        loss_fn = build_loss(cfg["loss"])
        final_history = None

        for size in training_cfg["sizes"]:
            size = int(size)
            train_loader = make_loader(
                fold_train_records,
                size=size,
                train=True,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                seed=fold_seed,
            )
            fold_holdout_loader = make_loader(
                fold_holdout_records,
                size=size,
                train=False,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                seed=fold_seed,
            )
            final_history = train_resolution(
                model,
                train_loader,
                fold_holdout_loader,
                loss_fn,
                cfg["optimizer"],
                device,
                frozen_epochs=int(training_cfg.get("frozen_epochs", 30)),
                unfrozen_epochs=int(training_cfg.get("unfrozen_epochs", 30)),
                lr_finder_cfg=training_cfg.get("lr_finder", {}),
                scheduler_cfg=training_cfg.get("scheduler", {}),
                selection_metric=selection_metric,
                tag=f"{args.tag} fold{fold_index} {size}px",
            )

        fold_holdout_loader = make_loader(
            fold_holdout_records,
            size=final_size,
            train=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=fold_seed,
        )
        official_validation_loader = make_loader(
            official_validation_records,
            size=final_size,
            train=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=fold_seed,
        )
        holdout_metrics = evaluate(
            [model], fold_holdout_loader, device, include_hd95=False
        )
        validation_metrics = evaluate(
            [model], official_validation_loader, device, include_hd95=True
        )

        checkpoint_path = (
            checkpoint_root / f"fold_{fold_index}" / f"{args.tag}_best.pt"
        )
        save_checkpoint(
            checkpoint_path,
            model,
            fold=fold_index,
            seed=fold_seed,
            selection_metric=selection_metric,
            best_epoch=int(final_history["unfrozen"].epoch) + 1,
            fold_holdout_metrics=holdout_metrics,
            official_validation_metrics=validation_metrics,
        )
        fold_results.append({
            "fold": fold_index,
            "best_epoch": int(final_history["unfrozen"].epoch) + 1,
            **{f"holdout_{key}": value for key, value in holdout_metrics.items()},
            **{f"official_val_{key}": value for key, value in validation_metrics.items()},
            "checkpoint": str(checkpoint_path),
        })
        fold_models.append(unwrap_model(model).cpu())
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ensemble_models = [
        prepare_model(model, device, data_parallel=data_parallel)
        for model in fold_models
    ]
    official_validation_loader = make_loader(
        official_validation_records,
        size=final_size,
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )
    ensemble_metrics = evaluate(
        ensemble_models,
        official_validation_loader,
        device,
        include_hd95=True,
    )

    fold_metrics_path = experiment_dir / f"{args.tag}_fold_metrics.csv"
    pd.DataFrame(fold_results).to_csv(fold_metrics_path, index=False)
    summary = {
        "selection_metric": selection_metric,
        "train_samples": len(training_records),
        "official_validation_samples": len(official_validation_records),
        "folds": fold_results,
        "official_validation_ensemble": ensemble_metrics,
    }
    summary_path = experiment_dir / f"{args.tag}_validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"fold metrics: {fold_metrics_path}")
    print(f"validation summary: {summary_path}")


if __name__ == "__main__":
    main()
