#!/usr/bin/env python
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import argparse
import csv
import json
import pandas as pd
import torch
import yaml

from src.data.loader import discover_kaggle_records, make_loader, split_records
from src.hpo.search_space import configuration_from_params
from src.models.unet34 import UNet34
from src.training.losses import build_loss
from src.training.trainer import train_resolution
from src.utils.checkpoint import load_model_state, save_checkpoint
from src.utils.device import describe_device, device_from_config, prepare_model
from src.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/optuna/unet34.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device, data_parallel = device_from_config(cfg)
    print(describe_device(device, data_parallel))

    data_cfg = cfg.get("data", {})
    records = discover_kaggle_records(
        "train",
        data_root=data_cfg.get("data_root"),
    )
    tr, va = split_records(
        records,
        val_fraction=float(data_cfg.get("val_fraction", 0.2)),
        seed=seed,
    )
    manifest_path = Path(cfg["artifacts"]["top5_manifest"])
    with manifest_path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 5:
        raise RuntimeError(
            f"Chưa đủ 5 COMPLETE candidates trong {manifest_path}"
        )

    results = []
    selection_metric = cfg["training"].get("selection_metric", "dice")
    out_dir = Path(cfg["artifacts"]["confirmation_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows[:5]:
        set_seed(seed)
        payload = torch.load(row["checkpoint"], map_location="cpu")
        params = json.loads(row["params_json"])
        sampled = configuration_from_params(params)
        model = prepare_model(
            UNet34(pretrained=False),
            device,
            data_parallel=data_parallel,
        )
        load_model_state(model, payload["model_state"])
        loss_fn = build_loss(sampled["loss"])
        bs = sampled["batch_size"]
        tr_dl = make_loader(tr, 256, True, bs, cfg["data"].get("num_workers", 2),
                            device.type == "cuda", seed=seed)
        va_dl = make_loader(va, 256, False, bs, cfg["data"].get("num_workers", 2),
                            device.type == "cuda", seed=seed)
        hist = train_resolution(
            model, tr_dl, va_dl, loss_fn, sampled["optimizer"], device,
            frozen_epochs=cfg["training"].get("frozen_epochs", 30),
            unfrozen_epochs=cfg["training"].get("unfrozen_epochs", 30),
            lr_finder_cfg=cfg["training"].get("lr_finder", {}),
            scheduler_cfg=cfg["training"].get("scheduler", {}),
            selection_metric=selection_metric,
            tag=f"trial {row['trial']} confirm256",
        )
        metrics = hist["unfrozen"].metrics
        ckpt = out_dir / f"trial_{int(row['trial']):04d}_256.pt"
        save_checkpoint(
            ckpt,
            model,
            trial=int(row["trial"]),
            params=params,
            selection_metric=selection_metric,
            metrics=metrics,
        )
        results.append({
            "trial": int(row["trial"]),
            "score": metrics[selection_metric],
            "dice": metrics["dice"],
            "jaccard": metrics["jaccard"],
            "params_json": json.dumps(params),
            "checkpoint": str(ckpt),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    out_csv = out_dir / "confirmation_results.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(pd.DataFrame(results))
    print(
        f"winner trial by {selection_metric}:",
        results[0]["trial"],
        f"score={results[0]['score']:.6f}",
    )


if __name__ == "__main__":
    main()
