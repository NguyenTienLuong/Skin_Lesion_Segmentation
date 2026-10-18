from __future__ import annotations
import optuna
import torch

from .search_space import sample_configuration
from .topk_manager import TopKManager
from ..data.loader import make_loader
from ..models.unet34 import UNet34, freeze_encoder, unfreeze
from ..training.losses import build_loss
from ..training.lr_finder import lr_find
from ..training.trainer import fit_stage
from ..utils.seed import set_seed


class UNet34Objective:
    """Stage-aware Optuna objective @128 theo PROJECT_CONTEXT."""

    def __init__(self, train_records, val_records, cfg: dict, device):
        self.train_records = train_records
        self.val_records = val_records
        self.cfg = cfg
        self.device = device
        self.manager = TopKManager(
            cfg["artifacts"]["top5_dir"], cfg["artifacts"]["top5_manifest"], k=5
        )

    def __call__(self, trial):
        seed = int(self.cfg.get("seed", 42))
        set_seed(seed)
        sampled = sample_configuration(trial, allow_batch32=self.cfg["search"].get("allow_batch32", False))
        bs = sampled["batch_size"]
        size = 128
        augment_cfg = self.cfg.get("augmentation", {})

        train_loader = make_loader(
            self.train_records, size=size, train=True, batch_size=bs,
            num_workers=self.cfg["data"].get("num_workers", 2),
            pin_memory=self.device.type == "cuda", augment_cfg=augment_cfg,
        )
        val_loader = make_loader(
            self.val_records, size=size, train=False, batch_size=bs,
            num_workers=self.cfg["data"].get("num_workers", 2),
            pin_memory=self.device.type == "cuda",
        )

        model = UNet34(pretrained=True).to(self.device)
        loss_fn = build_loss(sampled["loss"])
        val_batch = next(iter(val_loader))
        lrf_cfg = self.cfg["training"].get("lr_finder", {})
        sched_cfg = self.cfg["training"].get("scheduler", {})

        freeze_encoder(model)
        lr, _, _ = lr_find(
            model, train_loader, val_batch, loss_fn, sampled["optimizer"], self.device,
            bn_frozen=False, **lrf_cfg,
        )
        frozen = fit_stage(
            model, train_loader, val_loader, loss_fn, sampled["optimizer"], lr,
            self.cfg["training"].get("frozen_epochs", 30), self.device,
            bn_frozen=False, selection_metric="jaccard",
            cut_frac=sched_cfg.get("cut_frac", 0.1), ratio=sched_cfg.get("ratio", 32.0),
            tag=f"trial {trial.number} 128 frozen",
        )
        trial.report(float(frozen.metrics["jaccard"]), step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        unfreeze(model, freeze_bn=True)
        lr, _, _ = lr_find(
            model, train_loader, val_batch, loss_fn, sampled["optimizer"], self.device,
            bn_frozen=True, **lrf_cfg,
        )
        unfrozen = fit_stage(
            model, train_loader, val_loader, loss_fn, sampled["optimizer"], lr,
            self.cfg["training"].get("unfrozen_epochs", 30), self.device,
            bn_frozen=True, selection_metric="jaccard",
            cut_frac=sched_cfg.get("cut_frac", 0.1), ratio=sched_cfg.get("ratio", 32.0),
            tag=f"trial {trial.number} 128 unfrozen",
        )
        score = float(unfrozen.metrics["jaccard"])
        self.manager.update(trial.number, score, model, trial.params, unfrozen.metrics)
        return score
