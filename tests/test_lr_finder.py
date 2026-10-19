import torch

from src.training.lr_finder import lr_find


def test_lr_finder_uses_only_training_loader_and_restores_model():
    torch.manual_seed(42)
    model = torch.nn.Conv2d(1, 1, kernel_size=1)
    initial_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    train_loader = [
        (
            torch.rand(2, 1, 8, 8),
            torch.randint(0, 2, (2, 1, 8, 8)).float(),
        )
    ]

    selected_lr, lrs, losses = lr_find(
        model,
        train_loader,
        torch.nn.BCEWithLogitsLoss(),
        {"name": "Adam"},
        torch.device("cpu"),
        bn_frozen=False,
        lr_min=1e-5,
        lr_max=1e-2,
        n_iter=4,
    )

    assert len(lrs) == len(losses) == 4
    assert selected_lr in lrs
    assert all(torch.equal(model.state_dict()[key], value) for key, value in initial_state.items())
