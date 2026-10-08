from pathlib import Path
from typing import Any
import torch


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_checkpoint(path: str | Path, model: torch.nn.Module, **meta: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": cpu_state_dict(model), "meta": meta}, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    state = payload.get("model_state", payload)
    model.load_state_dict(state)
    return payload.get("meta", {})

