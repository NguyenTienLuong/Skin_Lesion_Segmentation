def sample_configuration(trial, allow_batch32: bool = False) -> dict:
    batches = [8, 16] + ([32] if allow_batch32 else [])
    batch_size = trial.suggest_categorical("batch_size", batches)

    opt_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    optimizer = {"name": opt_name}
    if opt_name == "SGD":
        optimizer["momentum"] = trial.suggest_float("momentum", 0.80, 0.99)
        optimizer["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    elif opt_name == "Adam":
        optimizer["weight_decay"] = 0.0
    else:
        optimizer["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    loss_name = trial.suggest_categorical("loss_type", ["BCE", "Dice", "Combo", "Tversky", "Focal"])
    loss = {"name": loss_name}
    if loss_name in {"Combo", "Tversky"}:
        loss["alpha"] = trial.suggest_float("alpha", 0.2, 0.8)
    elif loss_name == "Focal":
        loss["gamma"] = trial.suggest_float("gamma", 1.0, 4.0)

    return {"batch_size": batch_size, "optimizer": optimizer, "loss": loss}


def configuration_from_params(params: dict) -> dict:
    opt_name = params["optimizer"]
    optimizer = {"name": opt_name}
    if opt_name == "SGD":
        optimizer["momentum"] = params["momentum"]
        optimizer["weight_decay"] = params["weight_decay"]
    elif opt_name == "Adam":
        optimizer["weight_decay"] = 0.0
    else:
        optimizer["weight_decay"] = params["weight_decay"]

    loss_name = params["loss_type"]
    loss = {"name": loss_name}
    if loss_name in {"Combo", "Tversky"}:
        loss["alpha"] = params["alpha"]
    elif loss_name == "Focal":
        loss["gamma"] = params["gamma"]

    return {"batch_size": int(params["batch_size"]), "optimizer": optimizer, "loss": loss}
