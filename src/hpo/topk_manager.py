from pathlib import Path
import csv
import json
import torch

from ..utils.checkpoint import cpu_state_dict


class TopKManager:
    def __init__(self, checkpoint_dir, manifest_path, k: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.manifest_path = Path(manifest_path)
        self.k = k
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self):
        if not self.manifest_path.exists():
            return []
        with self.manifest_path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _write(self, rows):
        fields = ["trial", "score", "checkpoint", "params_json", "metrics_json"]
        with self.manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)

    def update(self, trial_number: int, score: float, model, params: dict, metrics: dict):
        rows = [r for r in self._read() if int(r["trial"]) != int(trial_number)]
        ckpt = self.checkpoint_dir / f"trial_{trial_number:04d}.pt"
        candidate = {
            "trial": str(trial_number),
            "score": f"{float(score):.10f}",
            "checkpoint": str(ckpt),
            "params_json": json.dumps(params, ensure_ascii=False),
            "metrics_json": json.dumps(metrics, ensure_ascii=False),
        }
        rows.append(candidate)
        rows = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
        keep = rows[: self.k]
        keep_trials = {int(r["trial"]) for r in keep}

        if trial_number in keep_trials:
            torch.save({"model_state": cpu_state_dict(model), "trial": trial_number,
                        "score": float(score), "params": params, "metrics": metrics}, ckpt)

        for r in rows[self.k:]:
            p = Path(r["checkpoint"])
            if p.exists():
                p.unlink()
        self._write(keep)
        return trial_number in keep_trials
