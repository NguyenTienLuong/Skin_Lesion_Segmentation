#!/usr/bin/env python
"""Chạy baseline MobileNetV2-UNet (W2-B2 @128, W2-B3 @256).

Dùng đúng pipeline của A: train_resolution (LR Finder + STLR, frozen 30 -> unfrozen 30),
evaluate() của metrics chung, Adam + BCE theo configs/baseline/mobilenetv2_baseline.yaml.

Ví dụ:
    # smoke nhanh (1+1 epoch, tập nhỏ) - kiểm tra pipeline chạy thông
    python scripts/baseline/train_mobilenetv2_unet.py --size 128 --smoke

    # W2-B2
    python scripts/baseline/train_mobilenetv2_unet.py --size 128

    # W2-B3 (pyramid transfer: khởi tạo từ checkpoint 128)
    python scripts/baseline/train_mobilenetv2_unet.py --size 256 \
        --init-checkpoint checkpoints/baseline/mobilenetv2_unet/size_128/best.pth
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import shutil

import numpy as np
import pandas as pd
import torch
import yaml

from src.metrics.segmentation_metrics import evaluate        # của A/C
from src.models.mobilenetv2_unet import MobileNetV2UNet
from src.training.losses import build_loss
from src.training.trainer import train_resolution
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.seed import set_seed


class Tee:
    """Ghi stdout đồng thời ra màn hình và file log."""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def save_lr_finder(exp_dir: Path, stage: str, lrs, losses, chosen_lr: float):
    pd.DataFrame({"lr": lrs, "loss": losses}).to_csv(exp_dir / f"lr_finder_{stage}.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(lrs, losses)
        ax.axvline(chosen_lr, color="red", linestyle="--", label=f"chosen {chosen_lr:.2e}")
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel("smoothed val loss")
        ax.set_title(f"LR Finder - {stage}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(exp_dir / f"lr_finder_{stage}.png", dpi=120)
        plt.close(fig)
    except ImportError:
        print("matplotlib chưa cài -> bỏ qua ảnh LR Finder (vẫn có file csv).")


def stage_summary(res):
    return {
        "lr": float(res.lr),
        "best_epoch": int(res.epoch) + 1,  # StageResult.epoch đánh số từ 0
        "val_metrics": {k: float(v) for k, v in res.metrics.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline/mobilenetv2_baseline.yaml")
    ap.add_argument("--size", type=int, required=True, choices=[128, 256])
    ap.add_argument("--init-checkpoint", default=None,
                    help="Checkpoint khởi tạo (B3: best @128).")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="Chạy thử nhanh, ghi vào thư mục smoke, KHÔNG phải kết quả baseline.")
    ap.add_argument("--temp-loader", action="store_true",
                    help="Dùng loader tạm (chỉ cho --smoke, khi chưa có file của C).")
    args = ap.parse_args()

    if args.temp_loader and not args.smoke:
        ap.error("--temp-loader chỉ được dùng cùng --smoke. Baseline thật phải dùng src/data/loader.py của C.")
    if args.temp_loader:
        from src.data.loader_temp import make_loader
        print("CẢNH BÁO: đang dùng LOADER TẠM, kết quả chỉ để kiểm tra pipeline.")
    else:
        from src.data.loader import make_loader  # W2-C1 (của C)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    run_id = args.run_id or f"mobilenetv2_unet_{args.size}_seed{seed}"

    exp_root = Path(cfg["artifacts"]["experiment_dir"])
    ckpt_root = Path(cfg["artifacts"]["checkpoint_dir"])
    train_cfg = dict(cfg["training"])
    lrf_cfg = dict(train_cfg["lr_finder"])
    frozen_epochs, unfrozen_epochs = train_cfg["frozen_epochs"], train_cfg["unfrozen_epochs"]

    if args.smoke:
        exp_root, ckpt_root = Path("experiments/smoke/mobilenetv2_unet"), Path("checkpoints/smoke/mobilenetv2_unet")
        run_id += "_SMOKE"
        frozen_epochs = unfrozen_epochs = 1
        lrf_cfg["n_iter"] = 20

    exp_dir = exp_root / f"size_{args.size}"
    ckpt_dir = ckpt_root / f"size_{args.size}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(exp_dir / "train.log")
    shutil.copy(args.config, exp_dir / "config_used.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"run_id={run_id} | size={args.size} | device={device} | seed={seed}")
    set_seed(seed)

    # --- Data: đúng split của A/C (tuning_split), không dùng held-out test ---
    df = pd.read_csv(cfg["data"]["tuning_split"])
    train_df = df[df.split == "train"].reset_index(drop=True)
    val_df = df[df.split == "val"].reset_index(drop=True)
    if args.smoke:
        train_df, val_df = train_df.head(32), val_df.head(16)
    print(f"train={len(train_df)} | val={len(val_df)}")

    bs = train_cfg["batch_size"]
    nw = cfg["data"].get("num_workers", 2)
    pin = device.type == "cuda"
    train_dl = make_loader(train_df, args.size, True, bs, nw, pin, cfg.get("augmentation", {}))
    val_dl = make_loader(val_df, args.size, False, bs, nw, pin)

    # --- Model ---
    from_ckpt = args.init_checkpoint is not None
    model = MobileNetV2UNet(pretrained=not from_ckpt).to(device)
    if from_ckpt:
        load_checkpoint(args.init_checkpoint, model, map_location=device)
        print(f"Khởi tạo từ checkpoint: {args.init_checkpoint}")
    loss_fn = build_loss(cfg["loss"])

    # --- Train: frozen -> unfrozen, mỗi stage có LR Finder riêng ---
    history = train_resolution(
        model, train_dl, val_dl, loss_fn, cfg["optimizer"], device,
        frozen_epochs=frozen_epochs, unfrozen_epochs=unfrozen_epochs,
        lr_finder_cfg=lrf_cfg, scheduler_cfg=train_cfg["scheduler"],
        selection_metric=train_cfg.get("selection_metric", "jaccard"),
        tag=f"{run_id}",
    )
    frozen, unfrozen = history["frozen"], history["unfrozen"]

    # --- Metric cuối (có HD95) trên val, model đang giữ best state của unfrozen stage ---
    final_metrics = evaluate([model], val_dl, device, include_hd95=True)
    print("Final val metrics:", json.dumps(final_metrics, indent=2))

    # --- Lưu artifacts ---
    for name, res in (("frozen", frozen), ("unfrozen", unfrozen)):
        save_lr_finder(exp_dir, name, res.lr_finder_lrs, res.lr_finder_losses, res.lr)
        pd.DataFrame({"step": np.arange(len(res.lr_trace)), "lr": res.lr_trace}).to_csv(
            exp_dir / f"lr_trace_{name}.csv", index=False)

    ckpt_path = ckpt_dir / "best.pth"
    save_checkpoint(ckpt_path, model, run_id=run_id, size=args.size, seed=seed,
                    best_epoch_unfrozen=int(unfrozen.epoch) + 1, metrics=final_metrics)

    report = {
        "run_id": run_id,
        "model": "mobilenetv2_unet",
        "size": args.size,
        "seed": seed,
        "smoke": bool(args.smoke),
        "init_checkpoint": args.init_checkpoint,
        "checkpoint": str(ckpt_path),
        "frozen": stage_summary(frozen),
        "unfrozen": stage_summary(unfrozen),
        "final_val_metrics": {k: float(v) for k, v in final_metrics.items()},
    }
    (exp_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {ckpt_path} | {exp_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()