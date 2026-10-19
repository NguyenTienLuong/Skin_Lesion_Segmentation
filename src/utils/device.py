"""Runtime device selection and optional single-process multi-GPU support."""

from __future__ import annotations

import torch
from torch import nn


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto``, ``cpu``, ``cuda`` or an explicit CUDA device."""
    if isinstance(requested, torch.device):
        device = requested
    else:
        key = str(requested).strip().lower()
        if key == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(key)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is not available")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {device}, but only {torch.cuda.device_count()} CUDA "
                "device(s) are available"
            )
    return device


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model from DataParallel/DDP wrappers."""
    wrappers = (nn.DataParallel, nn.parallel.DistributedDataParallel)
    while isinstance(model, wrappers):
        model = model.module
    return model


def should_use_data_parallel(
    device: torch.device,
    enabled: bool = True,
) -> bool:
    """Use all visible GPUs only for a non-indexed CUDA device."""
    if not isinstance(enabled, bool):
        raise TypeError("runtime.data_parallel must be true or false")
    return (
        enabled
        and device.type == "cuda"
        and device.index is None
        and torch.cuda.device_count() > 1
    )


def prepare_model(
    model: nn.Module,
    device: torch.device,
    data_parallel: bool = True,
) -> nn.Module:
    """Move a model to its device and use every visible GPU when requested."""
    model = model.to(device)
    if should_use_data_parallel(device, enabled=data_parallel):
        model = nn.DataParallel(
            model,
            device_ids=list(range(torch.cuda.device_count())),
        )
    return model


def device_from_config(cfg: dict) -> tuple[torch.device, bool]:
    """Read the runtime device policy while keeping backward compatibility."""
    runtime_cfg = cfg.get("runtime", {})
    device = resolve_device(runtime_cfg.get("device", "auto"))
    data_parallel = runtime_cfg.get("data_parallel", True)
    if not isinstance(data_parallel, bool):
        raise TypeError("runtime.data_parallel must be true or false")
    return device, data_parallel


def describe_device(device: torch.device, data_parallel: bool = True) -> str:
    """Return a concise description for experiment logs."""
    if device.type != "cuda":
        return "device=cpu, replicas=1"
    if should_use_data_parallel(device, enabled=data_parallel):
        names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        return f"device=cuda, replicas={len(names)}, gpus={names}"
    index = 0 if device.index is None else device.index
    return (
        f"device=cuda:{index}, replicas=1, "
        f"gpu={torch.cuda.get_device_name(index)}"
    )


__all__ = [
    "describe_device",
    "device_from_config",
    "prepare_model",
    "resolve_device",
    "should_use_data_parallel",
    "unwrap_model",
]
