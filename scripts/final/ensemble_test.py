#!/usr/bin/env python
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import argparse
import glob
import json
import torch
import yaml

from src.data.loader import discover_kaggle_records, make_loader
from src.models.unet34 import UNet34
from src.metrics.segmentation_metrics import evaluate
from src.utils.checkpoint import load_checkpoint
from src.utils.device import describe_device, device_from_config, prepare_model
from src.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-root", default=None)
    ap.add_argument("--tag", default="optimized")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device, data_parallel = device_from_config(cfg)
    print(describe_device(device, data_parallel))
    checkpoint_root = args.checkpoint_root or cfg.get("artifacts", {}).get(
        "final_checkpoint_root", "checkpoints/final/unet34"
    )
    files = sorted(glob.glob(str(Path(checkpoint_root) / "fold_*" / f"{args.tag}_best.pt")))
    expected_folds = int(cfg.get("data", {}).get("n_splits", 3))
    if len(files) != expected_folds:
        raise RuntimeError(
            f"Cần {expected_folds} checkpoint fold, tìm thấy {len(files)}: {files}"
        )

    models = []
    for f in files:
        m = prepare_model(
            UNet34(pretrained=False),
            device,
            data_parallel=data_parallel,
        )
        load_checkpoint(f, m, map_location=device); models.append(m)

    data_cfg = cfg.get("data", {})
    test = discover_kaggle_records(
        "test",
        data_root=data_cfg.get("data_root"),
        mask_root=data_cfg.get("mask_root"),
    )
    size = max(cfg["training"]["sizes"])
    dl = make_loader(test, size, False, cfg["training"]["batch_size"],
                     data_cfg.get("num_workers", 2), device.type == "cuda",
                     seed=seed)
    metrics = evaluate(models, dl, device, include_hd95=True)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
