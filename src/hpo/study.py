from pathlib import Path
import optuna


def create_or_load_study(cfg: dict):
    storage_path = Path(cfg["study"]["storage"])
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path.as_posix()}"

    sampler = optuna.samplers.TPESampler(seed=cfg.get("seed", 42))
    p = cfg["study"].get("pruner", {})
    pruner = optuna.pruners.PercentilePruner(
        percentile=float(p.get("percentile", 25.0)),
        n_startup_trials=int(p.get("n_startup_trials", 5)),
        n_warmup_steps=int(p.get("n_warmup_steps", 0)),
        interval_steps=int(p.get("interval_steps", 1)),
        n_min_trials=int(p.get("n_min_trials", 5)),
    )
    return optuna.create_study(
        study_name=cfg["study"]["name"],
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
