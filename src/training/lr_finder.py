import copy
import numpy as np
import torch

from .optimizers import build_optimizer
from src.models.unet34 import set_train_mode


def _make_scaler(device, enabled):
    return torch.amp.GradScaler(device.type, enabled=enabled and device.type == "cuda")


def lr_find(model, train_loader, val_batch, loss_fn, optimizer_cfg: dict, device,
            bn_frozen: bool, lr_min: float = 1e-6, lr_max: float = 1.0,
            n_iter: int = 100, beta: float = 0.9):
    """LR Finder từ notebook, tổng quát optimizer/loss để dùng baseline và Optuna."""
    state = copy.deepcopy(model.state_dict())
    optimizer = build_optimizer(model, lr=lr_min, cfg=optimizer_cfg)
    scaler = _make_scaler(device, enabled=True)
    mult = (lr_max / lr_min) ** (1 / max(1, n_iter - 1))

    xv, yv = val_batch[0].to(device), val_batch[1].to(device)
    iterator = iter(train_loader)
    lrs, losses = [], []
    avg, best = 0.0, float("inf")

    for i in range(n_iter):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)

        lr = lr_min * mult ** i
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        set_train_mode(model, bn_frozen)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = loss_fn(model(x).float(), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        model.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            val_loss = loss_fn(model(xv).float(), yv).item()

        avg = beta * avg + (1 - beta) * val_loss
        smoothed = avg / (1 - beta ** (i + 1))
        if not np.isfinite(smoothed) or (i > 5 and smoothed > 4 * best):
            break
        best = min(best, smoothed)
        lrs.append(lr)
        losses.append(smoothed)

    model.load_state_dict(state)
    lrs = np.asarray(lrs, dtype=float)
    losses = np.asarray(losses, dtype=float)
    if len(lrs) < 3:
        raise RuntimeError("LR Finder dừng quá sớm; không đủ điểm để chọn learning rate.")
    gradient = np.gradient(losses, np.log10(lrs))
    selected = float(lrs[int(np.argmin(gradient))])
    return selected, lrs, losses

