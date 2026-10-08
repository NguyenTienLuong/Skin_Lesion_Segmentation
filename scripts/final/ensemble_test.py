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
import pandas as pd
import torch
import yaml

from src.data.loader import make_loader
from src.models.unet34 import UNet34
from src.metrics.segmentation_metrics import evaluate
from src.utils.checkpoint import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-root", default="checkpoints/final/unet34")
    ap.add_argument("--tag", default="optimized")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(glob.glob(str(Path(args.checkpoint_root) / "fold_*" / f"{args.tag}_best.pt")))
    if len(files) != 3:
        raise RuntimeError(f"Cần 3 checkpoint fold, tìm thấy {len(files)}: {files}")

    models = []
    for f in files:
        m = UNet34(pretrained=False).to(device)
        load_checkpoint(f, m, map_location=device); models.append(m)

    test = pd.read_csv(cfg["data"]["heldout_test"])
    size = max(cfg["training"]["sizes"])
    dl = make_loader(test, size, False, cfg["training"]["batch_size"],
                     cfg["data"].get("num_workers", 2), device.type == "cuda")
    metrics = evaluate(models, dl, device, include_hd95=True)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
