#!/usr/bin/env python
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import argparse
import torch
import yaml

from src.data.loader import discover_kaggle_records, split_records
from src.hpo.objective import UNet34Objective
from src.hpo.study import create_or_load_study
from src.utils.seed import set_seed
from src.utils.device import describe_device, device_from_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/optuna/unet34.yaml")
    ap.add_argument("--timeout-seconds", type=int, default=None)
    ap.add_argument("--n-trials", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    data_cfg = cfg.get("data", {})
    records = discover_kaggle_records(
        "train",
        data_root=data_cfg.get("data_root"),
    )
    train_records, val_records = split_records(
        records,
        val_fraction=float(data_cfg.get("val_fraction", 0.2)),
        seed=seed,
    )
    print(
        f"runtime split: train={len(train_records)}, val={len(val_records)}, "
        f"seed={seed}"
    )
    device, data_parallel = device_from_config(cfg)
    print(describe_device(device, data_parallel))

    study = create_or_load_study(cfg)
    objective = UNet34Objective(train_records, val_records, cfg, device)
    study.optimize(objective, timeout=args.timeout_seconds, n_trials=args.n_trials, gc_after_trial=True)
    print("attempted:", len(study.trials))
    print("best:", study.best_trial.number, study.best_value, study.best_trial.params)


if __name__ == "__main__":
    main()
