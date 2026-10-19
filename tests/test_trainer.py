import torch
from types import SimpleNamespace

import src.training.trainer as trainer


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x * self.weight


def test_fit_stage_selects_best_dice_by_default(monkeypatch):
    model = _TinyModel()
    batch = (torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1))
    loader = [batch]
    epoch_metrics = iter([
        {
            "jaccard": 0.90,
            "threshold_jaccard": 0.90,
            "dice": 0.20,
            "sensitivity": 0.90,
        },
        {
            "jaccard": 0.30,
            "threshold_jaccard": 0.00,
            "dice": 0.80,
            "sensitivity": 0.50,
        },
    ])

    def fake_train_step(model, optimizer, scaler, loss_fn, x, y, device):
        optimizer.zero_grad(set_to_none=True)
        (model.weight * 0.0).backward()
        optimizer.step()
        with torch.no_grad():
            model.weight.add_(1.0)
        return 0.0

    monkeypatch.setattr(trainer, "train_step", fake_train_step)
    monkeypatch.setattr(trainer, "evaluate", lambda *args, **kwargs: next(epoch_metrics))

    result = trainer.fit_stage(
        model,
        loader,
        loader,
        torch.nn.BCEWithLogitsLoss(),
        {"name": "SGD"},
        lr=0.1,
        epochs=2,
        device=torch.device("cpu"),
        bn_frozen=False,
    )

    assert result.epoch == 1
    assert result.metrics["dice"] == 0.80
    assert model.weight.item() == 2.0


def test_train_resolution_never_passes_validation_loader_to_lr_finder(monkeypatch):
    model = _TinyModel()
    train_loader = object()
    validation_loader = object()
    lr_loader_calls = []
    fit_calls = []

    monkeypatch.setattr(trainer, "freeze_encoder", lambda model: None)
    monkeypatch.setattr(trainer, "unfreeze", lambda model, freeze_bn: None)

    def fake_lr_find(model, loader, *args, **kwargs):
        lr_loader_calls.append(loader)
        return 1e-3, torch.tensor([1e-4, 1e-3, 1e-2]).numpy(), torch.ones(3).numpy()

    def fake_fit_stage(model, train, validation, *args, **kwargs):
        fit_calls.append((train, validation, kwargs["selection_metric"]))
        return SimpleNamespace(lr_finder_lrs=None, lr_finder_losses=None)

    monkeypatch.setattr(trainer, "lr_find", fake_lr_find)
    monkeypatch.setattr(trainer, "fit_stage", fake_fit_stage)

    trainer.train_resolution(
        model,
        train_loader,
        validation_loader,
        torch.nn.BCEWithLogitsLoss(),
        {"name": "Adam"},
        torch.device("cpu"),
        frozen_epochs=1,
        unfrozen_epochs=1,
    )

    assert lr_loader_calls == [train_loader, train_loader]
    assert fit_calls == [
        (train_loader, validation_loader, "dice"),
        (train_loader, validation_loader, "dice"),
    ]
