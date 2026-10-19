import torch

from src.utils.checkpoint import cpu_state_dict, load_checkpoint, load_model_state, save_checkpoint
from src.utils.device import (
    device_from_config,
    prepare_model,
    resolve_device,
    should_use_data_parallel,
    unwrap_model,
)


def test_auto_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    device, data_parallel = device_from_config({})

    assert device == torch.device("cpu")
    assert data_parallel is True
    assert prepare_model(torch.nn.Linear(2, 1), device) is not None


def test_explicit_cuda_fails_clearly_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    try:
        resolve_device("cuda")
    except RuntimeError as exc:
        assert "CUDA is not available" in str(exc)
    else:
        raise AssertionError("Unavailable CUDA request must fail")


def test_multi_gpu_policy_uses_all_visible_non_indexed_cuda_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert should_use_data_parallel(torch.device("cuda"), enabled=True)
    assert not should_use_data_parallel(torch.device("cuda:1"), enabled=True)
    assert not should_use_data_parallel(torch.device("cuda"), enabled=False)
    assert not should_use_data_parallel(torch.device("cpu"), enabled=True)


def test_parallel_checkpoints_are_wrapper_independent(tmp_path):
    source = torch.nn.DataParallel(torch.nn.Linear(2, 1))
    with torch.no_grad():
        unwrap_model(source).weight.fill_(2.0)
        unwrap_model(source).bias.fill_(3.0)

    state = cpu_state_dict(source)
    assert state
    assert not any(key.startswith("module.") for key in state)

    target = torch.nn.DataParallel(torch.nn.Linear(2, 1))
    load_model_state(
        target,
        {f"module.{key}": value for key, value in state.items()},
    )
    assert torch.equal(unwrap_model(target).weight, unwrap_model(source).weight)
    assert torch.equal(unwrap_model(target).bias, unwrap_model(source).bias)

    checkpoint = tmp_path / "parallel.pt"
    save_checkpoint(checkpoint, source, selection_metric="dice")
    plain_target = torch.nn.Linear(2, 1)
    metadata = load_checkpoint(checkpoint, plain_target)
    assert metadata == {"selection_metric": "dice"}
    assert torch.equal(plain_target.weight, unwrap_model(source).weight)
    assert torch.equal(plain_target.bias, unwrap_model(source).bias)
