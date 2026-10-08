import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (prob * target).sum(dims)
        denom = prob.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()


class ComboLoss(nn.Module):
    """alpha * BCE + beta * DiceLoss, beta = 1-alpha theo search space dự án."""
    def __init__(self, alpha: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = 1 - alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)

    def forward(self, logits, target):
        return self.alpha * self.bce(logits, target) + self.beta * self.dice(logits, target)


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = 1 - alpha
        self.smooth = smooth

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        tp = (prob * target).sum(dims)
        fp = (prob * (1 - target)).sum(dims)
        fn = ((1 - prob) * target).sum(dims)
        score = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1 - score.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = torch.exp(-bce)
        return (((1 - p_t) ** self.gamma) * bce).mean()


def build_loss(cfg: dict) -> nn.Module:
    name = cfg.get("name", "BCE").lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss(smooth=cfg.get("smooth", 1e-6))
    if name == "combo":
        return ComboLoss(alpha=cfg.get("alpha", 0.5), smooth=cfg.get("smooth", 1e-6))
    if name == "tversky":
        return TverskyLoss(alpha=cfg.get("alpha", 0.5), smooth=cfg.get("smooth", 1e-6))
    if name == "focal":
        return FocalLoss(gamma=cfg.get("gamma", 2.0))
    raise ValueError(f"Unknown loss: {cfg.get('name')}")
