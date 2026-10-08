#!/usr/bin/env python
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import argparse
import pandas as pd
import torch
import yaml

from src.hpo.objective import UNet34Objective
from src.hpo.study import create_or_load_study
from src.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/optuna/unet34.yaml")
    ap.add_argument("--timeout-seconds", type=int, default=None)
    ap.add_argument("--n-trials", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(cfg.get("seed", 42))
    split = pd.read_csv(cfg["data"]["tuning_split"])
    train_records = split[split.split == "train"].reset_index(drop=True)
    val_records = split[split.split == "val"].reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    study = create_or_load_study(cfg)
    objective = UNet34Objective(train_records, val_records, cfg, device)
    study.optimize(objective, timeout=args.timeout_seconds, n_trials=args.n_trials, gc_after_trial=True)
    print("attempted:", len(study.trials))
    print("best:", study.best_trial.number, study.best_value, study.best_trial.params)


if __name__ == "__main__":
    main()
