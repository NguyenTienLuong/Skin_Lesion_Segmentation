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

from src.data.loader import make_loader
from src.models.unet34 import UNet34
from src.metrics.segmentation_metrics import evaluate
from src.utils.checkpoint import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline/unet34.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["val", "heldout"], default="val")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet34(pretrained=False).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    if args.split == "val":
        df = pd.read_csv(cfg["data"]["tuning_split"])
        df = df[df.split == "val"].reset_index(drop=True)
    else:
        df = pd.read_csv(cfg["data"]["heldout_test"])

    size = max(cfg["training"]["sizes"])
    dl = make_loader(df, size, False, cfg["training"]["batch_size"],
                     cfg["data"].get("num_workers", 2), device.type == "cuda")
    metrics = evaluate([model], dl, device, include_hd95=True)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

