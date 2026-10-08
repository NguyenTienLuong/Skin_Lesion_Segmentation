from __future__ import annotations
import math
import numpy as np
import torch
from scipy import ndimage


def per_image_scores(prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    pred = (prob > threshold).float()
    dims = (1, 2, 3)
    inter = (pred * target).sum(dims)
    sp = pred.sum(dims)
    st = target.sum(dims)
    union = sp + st - inter

    jaccard = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))
    dice = torch.where(sp + st > 0, 2 * inter / (sp + st).clamp(min=1), torch.ones_like(union))
    sensitivity = torch.where(st > 0, inter / st.clamp(min=1), torch.ones_like(st))
    return jaccard, dice, sensitivity


def threshold_jaccard(jaccard: torch.Tensor, cut: float = 0.65) -> torch.Tensor:
    return jaccard * (jaccard >= cut)


def _surface(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, border_value=0)
    return np.logical_xor(mask, eroded)


def hd95_single(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    h, w = target.shape
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float(math.hypot(h, w))

    ps = _surface(pred)
    ts = _surface(target)
    dt_to_t = ndimage.distance_transform_edt(~ts)
    dt_to_p = ndimage.distance_transform_edt(~ps)
    distances = np.concatenate([dt_to_t[ps], dt_to_p[ts]])
    return float(np.percentile(distances, 95)) if distances.size else 0.0


def summarize(prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5,
              cut: float = 0.65, include_hd95: bool = False) -> dict[str, float]:
    jac, dice, sens = per_image_scores(prob, target, threshold)
    out = {
        "jaccard": jac.mean().item(),
        "threshold_jaccard": threshold_jaccard(jac, cut).mean().item(),
        "dice": dice.mean().item(),
        "sensitivity": sens.mean().item(),
    }
    if include_hd95:
        pred = (prob > threshold).cpu().numpy()[:, 0]
        tgt = target.cpu().numpy()[:, 0] > 0.5
        out["hd95"] = float(np.mean([hd95_single(p, t) for p, t in zip(pred, tgt)]))
    return out


@torch.no_grad()
def predict(models, loader, device, use_amp: bool = True):
    probs, targets = [], []
    for x, y in loader:
        x = x.to(device)
        p = 0.0
        for model in models:
            model.eval()
            with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                p = p + torch.sigmoid(model(x).float())
        probs.append((p / len(models)).cpu())
        targets.append(y.cpu())
    return torch.cat(probs), torch.cat(targets)


@torch.no_grad()
def evaluate(models, loader, device, use_amp: bool = True, include_hd95: bool = False,
             threshold: float = 0.5, cut: float = 0.65):
    prob, target = predict(models, loader, device=device, use_amp=use_amp)
    return summarize(prob, target, threshold=threshold, cut=cut, include_hd95=include_hd95)

