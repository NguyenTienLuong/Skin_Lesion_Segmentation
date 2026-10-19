#!/usr/bin/env python
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import argparse
import json
import pandas as pd
import torch
import yaml

from src.data.loader import discover_kaggle_records, kfold_records, make_loader
from src.models.unet34 import UNet34
from src.training.losses import build_loss
from src.training.trainer import train_resolution
from src.metrics.segmentation_metrics import evaluate
from src.utils.seed import set_seed
from src.utils.checkpoint import save_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    data_cfg = cfg.get("data", {})
    records = discover_kaggle_records(
        "train",
        data_root=data_cfg.get("data_root"),
    )
    folds = kfold_records(
        records,
        n_splits=int(data_cfg.get("n_splits", 3)),
        seed=seed,
    )
    results = []

    for fold, (tr, va) in enumerate(folds):
        print(
            f"fold {fold + 1}/{len(folds)}: train={len(tr)}, val={len(va)}, "
            f"seed={seed}"
        )
        model = UNet34(pretrained=cfg["model"].get("pretrained", True)).to(device)
        loss_fn = build_loss(cfg["loss"])

        for size in cfg["training"]["sizes"]:
            bs = cfg["training"]["batch_size"]
            tr_dl = make_loader(tr, size, True, bs, cfg["data"].get("num_workers", 2),
                                device.type == "cuda", seed=seed)
            va_dl = make_loader(va, size, False, bs, cfg["data"].get("num_workers", 2),
                                device.type == "cuda", seed=seed)
            train_resolution(
                model, tr_dl, va_dl, loss_fn, cfg["optimizer"], device,
                frozen_epochs=cfg["training"]["frozen_epochs"],
                unfrozen_epochs=cfg["training"]["unfrozen_epochs"],
                lr_finder_cfg=cfg["training"]["lr_finder"],
                scheduler_cfg=cfg["training"]["scheduler"],
                selection_metric="jaccard", tag=f"{args.tag} fold{fold} {size}px",
            )

        final_dl = make_loader(va, max(cfg["training"]["sizes"]), False, cfg["training"]["batch_size"],
                               cfg["data"].get("num_workers", 2), device.type == "cuda",
                               seed=seed)
        metrics = evaluate([model], final_dl, device, include_hd95=True)
        ckpt_dir = Path(cfg["artifacts"].get("final_checkpoint_root", "checkpoints/final/unet34")) / f"fold_{int(fold)+1}"
        save_checkpoint(ckpt_dir / f"{args.tag}_best.pt", model, fold=int(fold), metrics=metrics)
        results.append({"fold": int(fold), **metrics})

    out = Path(cfg["artifacts"].get("final_experiment_dir", "experiments/final/unet34"))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / f"{args.tag}_fold_metrics.csv", index=False)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
