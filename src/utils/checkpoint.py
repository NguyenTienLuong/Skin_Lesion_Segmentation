from pathlib import Path
from typing import Any
import torch

from .device import unwrap_model


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    model = unwrap_model(model)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_model_state(model: torch.nn.Module, state: dict) -> None:
    """Load canonical or legacy ``module.``-prefixed weights into any wrapper."""
    while state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    unwrap_model(model).load_state_dict(state)


def save_checkpoint(path: str | Path, model: torch.nn.Module, **meta: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": cpu_state_dict(model), "meta": meta}, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    state = payload.get("model_state", payload)
    load_model_state(model, state)
    return payload.get("meta", {})


__all__ = [
    "cpu_state_dict",
    "load_checkpoint",
    "load_model_state",
    "save_checkpoint",
]

