import torch
from src.training.losses import build_loss


def test_all_losses_are_finite():
    logits = torch.randn(2, 1, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    cfgs = [
        {"name": "BCE"}, {"name": "Dice"}, {"name": "Combo", "alpha": 0.5},
        {"name": "Tversky", "alpha": 0.5}, {"name": "Focal", "gamma": 2.0},
    ]
    for cfg in cfgs:
        assert torch.isfinite(build_loss(cfg)(logits, target))
