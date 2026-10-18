#!/usr/bin/env python
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import argparse
import yaml
from src.hpo.study import create_or_load_study


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/optuna/unet34.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    study = create_or_load_study(cfg)
    out = Path(cfg["artifacts"]["trials_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
