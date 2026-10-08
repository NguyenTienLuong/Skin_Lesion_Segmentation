import torch


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def build_optimizer(model, lr: float, cfg: dict):
    name = cfg.get("name", "Adam").lower()
    params = trainable_parameters(model)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=cfg.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=cfg.get("weight_decay", 1e-4))
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=cfg.get("momentum", 0.9),
            weight_decay=cfg.get("weight_decay", 0.0),
        )
    raise ValueError(f"Unknown optimizer: {cfg.get('name')}")
