#!/usr/bin/env python
"""Kiểm tra trực quan: đưa 1 ảnh qua mô hình đã huấn luyện, so với mask đúng.

Ví dụ:
    # lấy ảnh số 0 trong tập val của split hiện tại
    python scripts/predict_one.py --checkpoint checkpoints/smoke/mobilenetv2_unet/size_128/best.pth --size 128 --index 0

    # hoặc chỉ định ảnh và mask cụ thể
    python scripts/predict_one.py --checkpoint <best.pth> --size 128 --image path/anh.jpg --mask path/mask.png

Kết quả: file PNG gồm 4 ô: ảnh gốc | mask đúng | mask dự đoán | chồng lên nhau,
kèm Jaccard và Dice của riêng ảnh đó.

"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.loader_temp import IMAGENET_MEAN, IMAGENET_STD, TempSegDataset
from src.models.mobilenetv2_unet import MobileNetV2UNet
from src.utils.checkpoint import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--size", type=int, required=True, choices=[128, 256])
    ap.add_argument("--config", default="configs/baseline/mobilenetv2_baseline.yaml")
    ap.add_argument("--index", type=int, default=0, help="Chỉ số ảnh trong tập val.")
    ap.add_argument("--image", default=None)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="Đường dẫn file PNG kết quả.")
    args = ap.parse_args()

    if args.image:
        if not args.mask:
            ap.error("--image cần đi kèm --mask")
        df = pd.DataFrame({"image_path": [args.image], "mask_path": [args.mask]})
        name = Path(args.image).stem
    else:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        full = pd.read_csv(cfg["data"]["tuning_split"])
        val = full[full.split == "val"].reset_index(drop=True)
        row = val.iloc[[args.index]].reset_index(drop=True)
        df = row[["image_path", "mask_path"]]
        name = row.loc[0, "image_id"]

    x, y = TempSegDataset(df, args.size, train=False)[0]

    model = MobileNetV2UNet(pretrained=False)
    meta = load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x[None]))[0, 0].numpy()

    pred = prob > args.threshold
    gt = y[0].numpy() > 0.5
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    jaccard = inter / union if union else 1.0
    denom = pred.sum() + gt.sum()
    dice = 2 * inter / denom if denom else 1.0

    img = x.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)

    overlay = img.copy()
    overlay[np.logical_and(pred, gt)] = overlay[np.logical_and(pred, gt)] * 0.4 + np.array([0, 1, 0]) * 0.6   # xanh: đúng
    overlay[np.logical_and(pred, ~gt)] = overlay[np.logical_and(pred, ~gt)] * 0.4 + np.array([1, 0, 0]) * 0.6  # đỏ: dự đoán thừa
    overlay[np.logical_and(~pred, gt)] = overlay[np.logical_and(~pred, gt)] * 0.4 + np.array([0, 0, 1]) * 0.6  # xanh dương: bỏ sót

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, im, title, cmap in [
        (axes[0], img, "Ảnh gốc", None),
        (axes[1], gt, "Mask đúng", "gray"),
        (axes[2], pred, "Mô hình dự đoán", "gray"),
        (axes[3], overlay, "Xanh lá=đúng, đỏ=thừa, xanh dương=sót", None),
    ]:
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{name} | Jaccard {jaccard:.3f} | Dice {dice:.3f}", fontsize=11)
    fig.tight_layout()

    out = Path(args.out) if args.out else Path(args.checkpoint).parent / f"predict_{name}.png"
    fig.savefig(out, dpi=130)
    print(f"Ảnh: {name} | Jaccard {jaccard:.4f} | Dice {dice:.4f}")
    if meta:
        print("Thông tin checkpoint:", {k: v for k, v in meta.items() if k != "metrics"})
    print(f"Đã lưu: {out}")


if __name__ == "__main__":
    main()