from __future__ import annotations
from dataclasses import dataclass
import copy
import numpy as np
import torch

from .optimizers import build_optimizer
from .scheduler import stlr_lambda
from .lr_finder import lr_find
from src.metrics.segmentation_metrics import evaluate
from src.models.unet34 import freeze_encoder, unfreeze, set_train_mode
from ..utils.checkpoint import cpu_state_dict


@dataclass
class StageResult:
    metrics: dict
    epoch: int
    lr: float
    state: dict
    lr_finder_lrs: np.ndarray
    lr_finder_losses: np.ndarray
    lr_trace: np.ndarray


def _make_scaler(device):
    return torch.amp.GradScaler(device.type, enabled=device.type == "cuda")


def train_step(model, optimizer, scaler, loss_fn, x, y, device):
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        loss = loss_fn(model(x).float(), y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.item())


def fit_stage(model, train_loader, val_loader, loss_fn, optimizer_cfg: dict,
              lr: float, epochs: int, device, bn_frozen: bool,
              selection_metric: str = "jaccard", cut_frac: float = 0.1,
              ratio: float = 32.0, tag: str = "") -> StageResult:
    optimizer = build_optimizer(model, lr=lr, cfg=optimizer_cfg)
    scaler = _make_scaler(device)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, stlr_lambda(epochs * len(train_loader), cut_frac=cut_frac, ratio=ratio)
    )

    best_score = -float("inf")
    best = None
    lr_trace = []
    for epoch in range(epochs):
        set_train_mode(model, bn_frozen)
        total_loss = 0.0
        for x, y in train_loader:
            total_loss += train_step(model, optimizer, scaler, loss_fn, x, y, device)
            scheduler.step()
            lr_trace.append(optimizer.param_groups[0]["lr"])

        metrics = evaluate([model], val_loader, device=device, include_hd95=False)
        score = float(metrics[selection_metric])
        if score > best_score:
            best_score = score
            best = {
                "metrics": metrics,
                "epoch": epoch,
                "state": cpu_state_dict(model),
            }
        print(
            f"[{tag}] ep {epoch + 1:02d}/{epochs} "
            f"loss {total_loss / max(1, len(train_loader)):.4f} | "
            f"val jac {metrics['jaccard']:.4f} dice {metrics['dice']:.4f} "
            f"thr-jac {metrics['threshold_jaccard']:.4f}"
        )

    model.load_state_dict(best["state"])
    return StageResult(
        metrics=best["metrics"], epoch=best["epoch"], lr=lr,
        state=best["state"], lr_finder_lrs=np.array([]), lr_finder_losses=np.array([]),
        lr_trace=np.asarray(lr_trace),
    )


def train_resolution(model, train_loader, val_loader, loss_fn, optimizer_cfg: dict,
                     device, frozen_epochs: int = 30, unfrozen_epochs: int = 30,
                     lr_finder_cfg: dict | None = None, scheduler_cfg: dict | None = None,
                     selection_metric: str = "jaccard", tag: str = ""):
    lr_finder_cfg = lr_finder_cfg or {}
    scheduler_cfg = scheduler_cfg or {}
    val_batch = next(iter(val_loader))
    history = {}

    freeze_encoder(model)
    lr, lrs, losses = lr_find(
        model, train_loader, val_batch, loss_fn, optimizer_cfg, device,
        bn_frozen=False, **lr_finder_cfg,
    )
    frozen = fit_stage(
        model, train_loader, val_loader, loss_fn, optimizer_cfg, lr,
        frozen_epochs, device, bn_frozen=False, selection_metric=selection_metric,
        cut_frac=scheduler_cfg.get("cut_frac", 0.1), ratio=scheduler_cfg.get("ratio", 32.0),
        tag=f"{tag} frozen lr={lr:.2e}",
    )
    frozen.lr_finder_lrs, frozen.lr_finder_losses = lrs, losses
    history["frozen"] = frozen

    unfreeze(model, freeze_bn=True)
    lr, lrs, losses = lr_find(
        model, train_loader, val_batch, loss_fn, optimizer_cfg, device,
        bn_frozen=True, **lr_finder_cfg,
    )
    unfrozen = fit_stage(
        model, train_loader, val_loader, loss_fn, optimizer_cfg, lr,
        unfrozen_epochs, device, bn_frozen=True, selection_metric=selection_metric,
        cut_frac=scheduler_cfg.get("cut_frac", 0.1), ratio=scheduler_cfg.get("ratio", 32.0),
        tag=f"{tag} unfrozen lr={lr:.2e}",
    )
    unfrozen.lr_finder_lrs, unfrozen.lr_finder_losses = lrs, losses
    history["unfrozen"] = unfrozen
    return history
