import importlib.util
import sys
from types import SimpleNamespace

import torch
import yaml

import scripts.baseline.train_unet34 as baseline_script

if importlib.util.find_spec("optuna") is None:
    sys.modules["optuna"] = SimpleNamespace(
        TrialPruned=type("TrialPruned", (Exception,), {})
    )

import src.hpo.objective as objective_module
import scripts.optuna.run_unet34_optuna as optuna_script


METRICS = {
    "jaccard": 0.5,
    "threshold_jaccard": 0.0,
    "dice": 0.7,
    "sensitivity": 0.6,
    "hd95": 1.0,
}


class _FakeModel:
    def to(self, device):
        return self

    def cpu(self):
        return self


class _FakeLoader:
    def __init__(self, records, train):
        self.records = tuple(records)
        self.train = bool(train)


def test_baseline_uses_fold_holdout_for_selection_and_official_val_for_report(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "baseline.yaml"
    checkpoint_root = tmp_path / "checkpoints"
    experiment_root = tmp_path / "experiments"
    config_path.write_text(
        yaml.safe_dump({
            "seed": 42,
            "model": {"pretrained": False},
            "data": {
                "data_root": "kaggle-root",
                "mask_root": "local-mask-root",
                "n_splits": 3,
                "num_workers": 0,
            },
            "optimizer": {"name": "Adam"},
            "loss": {"name": "BCE"},
            "training": {
                "selection_metric": "dice",
                "sizes": [64],
                "batch_size": 2,
                "frozen_epochs": 1,
                "unfrozen_epochs": 1,
                "lr_finder": {"n_iter": 3},
                "scheduler": {},
            },
            "artifacts": {
                "final_checkpoint_root": str(checkpoint_root),
                "final_experiment_dir": str(experiment_root),
            },
        }),
        encoding="utf-8",
    )

    discovery_calls = []
    loader_calls = []
    training_calls = []
    evaluation_calls = []
    checkpoint_calls = []

    def fake_discover(split, data_root=None, mask_root=None):
        discovery_calls.append((split, data_root, mask_root))
        if split == "train":
            return ["all-training-records"]
        if split == "validation":
            return ["official-validation"]
        raise AssertionError(f"Baseline must not discover {split!r}")

    folds = [
        ([f"fold-{index}-train"], [f"fold-{index}-holdout"])
        for index in range(1, 4)
    ]

    def fake_kfold(records, n_splits, seed):
        assert records == ["all-training-records"]
        assert n_splits == 3
        assert seed == 42
        return folds

    def fake_make_loader(records, *, train, **kwargs):
        loader = _FakeLoader(records, train)
        loader_calls.append(loader)
        return loader

    def fake_train_resolution(model, train_loader, holdout_loader, *args, **kwargs):
        training_calls.append((train_loader, holdout_loader, kwargs["selection_metric"]))
        return {"unfrozen": SimpleNamespace(epoch=0)}

    def fake_evaluate(models, loader, *args, **kwargs):
        evaluation_calls.append((len(models), loader))
        return dict(METRICS)

    def fake_save_checkpoint(path, model, **metadata):
        checkpoint_calls.append((path, metadata))

    monkeypatch.setattr(baseline_script, "discover_kaggle_records", fake_discover)
    monkeypatch.setattr(baseline_script, "kfold_records", fake_kfold)
    monkeypatch.setattr(baseline_script, "make_loader", fake_make_loader)
    monkeypatch.setattr(baseline_script, "train_resolution", fake_train_resolution)
    monkeypatch.setattr(baseline_script, "evaluate", fake_evaluate)
    monkeypatch.setattr(baseline_script, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(baseline_script, "UNet34", lambda pretrained: _FakeModel())
    monkeypatch.setattr(baseline_script, "build_loss", lambda cfg: object())
    monkeypatch.setattr(baseline_script, "set_seed", lambda seed: None)
    monkeypatch.setattr(baseline_script.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        baseline_script.sys,
        "argv",
        ["train_unet34.py", "--config", str(config_path)],
    )

    baseline_script.main()

    assert discovery_calls == [
        ("train", "kaggle-root", None),
        ("validation", "kaggle-root", "local-mask-root"),
    ]
    assert len(training_calls) == 3
    assert all(train_loader.train for train_loader, _, _ in training_calls)
    assert all(not holdout_loader.train for _, holdout_loader, _ in training_calls)
    assert all(metric == "dice" for _, _, metric in training_calls)
    assert not any(
        loader.train and loader.records == ("official-validation",)
        for loader in loader_calls
    )
    assert not any(
        loader.train and "holdout" in loader.records[0]
        for loader in loader_calls
    )
    assert any(
        model_count == 3
        and loader.records == ("official-validation",)
        and not loader.train
        for model_count, loader in evaluation_calls
    )
    assert len(checkpoint_calls) == 3
    assert all(
        metadata["selection_metric"] == "dice"
        for _, metadata in checkpoint_calls
    )


def test_optuna_uses_augmented_train_and_clean_validation_only_for_selection(
    tmp_path, monkeypatch
):
    train_records = ["hpo-train"]
    validation_records = ["hpo-validation"]
    objective = objective_module.UNet34Objective.__new__(
        objective_module.UNet34Objective
    )
    objective.train_records = train_records
    objective.val_records = validation_records
    objective.device = torch.device("cpu")
    objective.cfg = {
        "seed": 42,
        "search": {"allow_batch32": False},
        "data": {"num_workers": 0},
        "training": {
            "selection_metric": "dice",
            "frozen_epochs": 1,
            "unfrozen_epochs": 1,
            "lr_finder": {},
            "scheduler": {},
        },
    }

    manager_calls = []
    objective.manager = SimpleNamespace(
        update=lambda *args: manager_calls.append(args)
    )
    loader_calls = []
    lr_loader_calls = []
    fit_calls = []

    def fake_make_loader(records, *, train, **kwargs):
        loader = _FakeLoader(records, train)
        loader_calls.append(loader)
        return loader

    def fake_lr_find(model, loader, *args, **kwargs):
        lr_loader_calls.append(loader)
        return 1e-3, [], []

    def fake_fit_stage(model, train_loader, val_loader, *args, **kwargs):
        fit_calls.append((train_loader, val_loader, kwargs["selection_metric"]))
        dice = 0.6 if len(fit_calls) == 1 else 0.8
        return SimpleNamespace(metrics={"dice": dice, "jaccard": dice - 0.1})

    monkeypatch.setattr(objective_module, "make_loader", fake_make_loader)
    monkeypatch.setattr(objective_module, "lr_find", fake_lr_find)
    monkeypatch.setattr(objective_module, "fit_stage", fake_fit_stage)
    monkeypatch.setattr(objective_module, "freeze_encoder", lambda model: None)
    monkeypatch.setattr(
        objective_module, "unfreeze", lambda model, freeze_bn: None
    )
    monkeypatch.setattr(objective_module, "UNet34", lambda pretrained: _FakeModel())
    monkeypatch.setattr(objective_module, "build_loss", lambda cfg: object())
    monkeypatch.setattr(objective_module, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        objective_module,
        "sample_configuration",
        lambda trial, allow_batch32: {
            "batch_size": 2,
            "loss": {"name": "BCE"},
            "optimizer": {"name": "Adam"},
        },
    )

    class FakeTrial:
        number = 7
        params = {}

        def __init__(self):
            self.reports = []

        def report(self, value, step):
            self.reports.append((value, step))

        def should_prune(self):
            return False

    trial = FakeTrial()
    score = objective(trial)

    train_loader = next(loader for loader in loader_calls if loader.train)
    validation_loader = next(loader for loader in loader_calls if not loader.train)
    assert train_loader.records == ("hpo-train",)
    assert validation_loader.records == ("hpo-validation",)
    assert lr_loader_calls == [train_loader, train_loader]
    assert fit_calls == [
        (train_loader, validation_loader, "dice"),
        (train_loader, validation_loader, "dice"),
    ]
    assert trial.reports == [(0.6, 0)]
    assert score == 0.8
    assert manager_calls[0][1] == 0.8


def test_optuna_entrypoint_splits_only_the_official_training_set(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "optuna.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "seed": 123,
            "data": {
                "data_root": "kaggle-root",
                "val_fraction": 0.25,
            },
        }),
        encoding="utf-8",
    )
    all_training_records = ["all-official-training-records"]
    hpo_train_records = ["hpo-train"]
    hpo_validation_records = ["hpo-validation"]
    discovery_calls = []
    split_calls = []
    objective_calls = []
    optimize_calls = []

    def fake_discover(split, data_root=None):
        discovery_calls.append((split, data_root))
        if split != "train":
            raise AssertionError("Optuna must split only the official training set")
        return all_training_records

    def fake_split(records, val_fraction, seed):
        split_calls.append((records, val_fraction, seed))
        return hpo_train_records, hpo_validation_records

    objective_marker = object()

    def fake_objective(train_records, val_records, cfg, device):
        objective_calls.append((train_records, val_records, cfg, device))
        return objective_marker

    class FakeStudy:
        trials = [object()]
        best_trial = SimpleNamespace(number=7, params={"batch_size": 8})
        best_value = 0.8

        def optimize(self, objective, **kwargs):
            optimize_calls.append((objective, kwargs))

    monkeypatch.setattr(optuna_script, "discover_kaggle_records", fake_discover)
    monkeypatch.setattr(optuna_script, "split_records", fake_split)
    monkeypatch.setattr(optuna_script, "UNet34Objective", fake_objective)
    monkeypatch.setattr(optuna_script, "create_or_load_study", lambda cfg: FakeStudy())
    monkeypatch.setattr(optuna_script, "set_seed", lambda seed: None)
    monkeypatch.setattr(optuna_script.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        optuna_script.sys,
        "argv",
        [
            "run_unet34_optuna.py",
            "--config",
            str(config_path),
            "--n-trials",
            "1",
        ],
    )

    optuna_script.main()

    assert discovery_calls == [("train", "kaggle-root")]
    assert split_calls == [(all_training_records, 0.25, 123)]
    assert objective_calls[0][0] is hpo_train_records
    assert objective_calls[0][1] is hpo_validation_records
    assert objective_calls[0][3] == torch.device("cpu")
    assert optimize_calls == [(
        objective_marker,
        {
            "timeout": None,
            "n_trials": 1,
            "gc_after_trial": True,
        },
    )]
