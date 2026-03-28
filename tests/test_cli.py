"""Smoke tests for the package CLI entrypoint."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest


def cli_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(Path("src").resolve())
    env["PYTHONPATH"] = (
        src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]
    )
    return env


def test_cli_help_lists_subcommands():
    result = subprocess.run(
        [sys.executable, "-m", "ndswin.cli", "--help"],
        capture_output=True,
        text=True,
        env=cli_env(),
    )
    assert result.returncode == 0
    assert "train" in result.stdout
    assert "sweep" in result.stdout
    assert "auto-sweep" in result.stdout
    assert "queue" in result.stdout
    assert "fetch-data" in result.stdout
    assert "show-best" in result.stdout
    assert "validate" in result.stdout


def test_cli_show_best_smoke(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            [
                {
                    "trial": 0,
                    "status": "success",
                    "val_accuracy": 0.6,
                    "elapsed_seconds": 11,
                    "config_path": "a.json",
                },
                {
                    "trial": 1,
                    "status": "success",
                    "val_accuracy": 0.8,
                    "elapsed_seconds": 9,
                    "config_path": "b.json",
                },
            ]
        )
    )
    result = subprocess.run(
        [sys.executable, "-m", "ndswin.cli", "show-best", "--summary", str(summary_path)],
        capture_output=True,
        text=True,
        env=cli_env(),
    )
    assert result.returncode == 0
    assert "Best Trial: 1" in result.stdout
    assert "Val Accuracy: 0.8000" in result.stdout


def test_cli_queue_dry_run_smoke(tmp_path: Path):
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(
        json.dumps(
            [
                {
                    "name": "dry_train",
                    "type": "train",
                    "config": "configs/cifar10.json",
                    "epochs": 1,
                }
            ]
        )
    )
    result = subprocess.run(
        [sys.executable, "-m", "ndswin.cli", "queue", "--queue", str(queue_path), "--dry-run"],
        capture_output=True,
        text=True,
        env=cli_env(),
    )
    assert result.returncode == 0
    assert "QUEUE RUNNER" in result.stdout
    assert "dry_train" in result.stdout


def test_run_train_command_sets_up_gpus_before_loading_training_runtime(monkeypatch):
    import ndswin.cli as cli

    calls: list[str] = []

    def fake_setup_optimal_gpus():
        calls.append("gpu_setup")

    def fake_load_training_runtime():
        calls.append("load_runtime")
        raise RuntimeError("stop_after_order_check")

    monkeypatch.setattr("ndswin.utils.gpu.setup_optimal_gpus", fake_setup_optimal_gpus)
    monkeypatch.setattr(cli, "load_training_runtime", fake_load_training_runtime)

    args = SimpleNamespace(
        config="configs/cifar10.json",
        epochs=None,
        batch_size=None,
        lr=None,
        seed=None,
        data_dir=None,
        max_steps_per_epoch=None,
        stamp=None,
        no_log_file=True,
        log_level="INFO",
        overrides=[],
    )

    with pytest.raises(RuntimeError, match="stop_after_order_check"):
        cli.run_train_command(args)

    assert calls == ["gpu_setup", "load_runtime"]


def _write_volume_split(root: Path, split: str, num_classes: int) -> None:
    for class_idx in range(num_classes):
        class_dir = root / split / f"class_{class_idx:03d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            class_dir / "00000.npz",
            image=np.zeros((1, 4, 4, 4), dtype=np.float32),
        )


def test_run_train_command_rejects_class_count_mismatch_before_runtime(tmp_path: Path, monkeypatch):
    import ndswin.cli as cli

    config = json.loads(Path("configs/modelnet10.json").read_text())
    config["data"]["data_dir"] = str(tmp_path / "modelnet40_like")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    _write_volume_split(tmp_path / "modelnet40_like", "train", 40)
    _write_volume_split(tmp_path / "modelnet40_like", "validation", 40)

    runtime_loaded = False

    def fake_load_training_runtime():
        nonlocal runtime_loaded
        runtime_loaded = True
        raise AssertionError("training runtime should not load on dataset mismatch")

    monkeypatch.setattr("ndswin.utils.gpu.setup_optimal_gpus", lambda: None)
    monkeypatch.setattr(cli, "load_training_runtime", fake_load_training_runtime)

    args = SimpleNamespace(
        config=str(config_path),
        epochs=None,
        batch_size=None,
        lr=None,
        seed=None,
        data_dir=None,
        max_steps_per_epoch=None,
        max_devices=None,
        stamp=None,
        no_log_file=True,
        log_level="INFO",
        overrides=[],
    )

    assert cli.run_train_command(args) == 1
    assert runtime_loaded is False


def test_validate_experiment_dataset_contract_accepts_true_modelnet10_layout(tmp_path: Path):
    import ndswin.cli as cli

    config = json.loads(Path("configs/modelnet10.json").read_text())
    config["data"]["data_dir"] = str(tmp_path / "modelnet10")
    config_path = tmp_path / "modelnet10.json"
    config_path.write_text(json.dumps(config))

    _write_volume_split(tmp_path / "modelnet10", "train", 10)
    _write_volume_split(tmp_path / "modelnet10", "validation", 10)

    exp_config = cli.load_experiment_config(str(config_path))
    contract = cli.validate_experiment_dataset_contract(exp_config, require_validation_split=True)
    assert contract is not None
    assert contract["num_classes"] == 10


def test_run_train_command_requires_validation_split_before_runtime(tmp_path: Path, monkeypatch):
    import ndswin.cli as cli

    config = json.loads(Path("configs/modelnet10.json").read_text())
    config["data"]["data_dir"] = str(tmp_path / "modelnet10")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    _write_volume_split(tmp_path / "modelnet10", "train", 10)
    _write_volume_split(tmp_path / "modelnet10", "test", 10)

    runtime_loaded = False

    def fake_load_training_runtime():
        nonlocal runtime_loaded
        runtime_loaded = True
        raise AssertionError("training runtime should not load without a validation split")

    monkeypatch.setattr("ndswin.utils.gpu.setup_optimal_gpus", lambda: None)
    monkeypatch.setattr(cli, "load_training_runtime", fake_load_training_runtime)

    args = SimpleNamespace(
        config=str(config_path),
        epochs=None,
        batch_size=None,
        lr=None,
        seed=None,
        data_dir=None,
        max_steps_per_epoch=None,
        max_devices=None,
        stamp=None,
        no_log_file=True,
        log_level="INFO",
        overrides=[],
    )

    assert cli.run_train_command(args) == 1
    assert runtime_loaded is False


def test_export_dataset_writes_manifest_for_string_labels(tmp_path: Path, monkeypatch):
    import ndswin.cli as cli

    fake_dataset = {
        "train": [
            {"image": np.zeros((8, 8, 3), dtype=np.uint8), "label": "chair"},
            {"image": np.zeros((8, 8, 3), dtype=np.uint8), "label": "table"},
        ],
        "validation": [
            {"image": np.zeros((8, 8, 3), dtype=np.uint8), "label": "chair"},
        ],
    }

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda hf_id: fake_dataset),
    )

    cli.export_dataset("demo/furniture", str(tmp_path))

    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text())
    assert manifest["task"] == "classification"
    assert manifest["num_classes"] == 2
    assert manifest["split_counts"]["train"] == 2
    assert set(manifest["class_labels"].values()) == {"chair", "table"}


def test_cli_show_best_lists_each_summary(tmp_path: Path):
    sweeps_dir = tmp_path / "sweeps"
    alpha_dir = sweeps_dir / "alpha_sweep"
    beta_dir = sweeps_dir / "beta_sweep"
    alpha_dir.mkdir(parents=True)
    beta_dir.mkdir(parents=True)
    (alpha_dir / "summary.json").write_text(
        json.dumps(
            [{"trial": 0, "status": "success", "val_accuracy": 0.6, "config_path": "a.json"}]
        )
    )
    (beta_dir / "summary.json").write_text(
        json.dumps(
            [{"trial": 1, "status": "success", "val_accuracy": 0.8, "config_path": "b.json"}]
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ndswin.cli",
            "show-best",
            "--outputs-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=cli_env(),
    )

    assert result.returncode == 0
    assert f"Source: {alpha_dir / 'summary.json'}" in result.stdout
    assert f"Source: {beta_dir / 'summary.json'}" in result.stdout


def test_run_auto_sweep_uses_sweep_file_trials_by_default(tmp_path: Path, monkeypatch):
    import ndswin.cli as cli

    sweep_out = tmp_path / "sweep_out"
    sweep_out.mkdir()
    best_config = tmp_path / "best_config.json"
    best_config.write_text(json.dumps({"name": "best"}))
    (sweep_out / "summary.json").write_text(
        json.dumps(
            {
                "metric": "val_accuracy",
                "results": [
                    {
                        "trial": 0,
                        "status": "success",
                        "val_accuracy": 0.7,
                        "config_path": str(best_config),
                        "dataset": "demo",
                        "stamp": "stamp",
                    }
                ],
            }
        )
    )

    commands: list[list[str]] = []

    def fake_run(cmd, check=True, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        cli,
        "load_sweep",
        lambda path: {"trials": 7, "output_dir": str(sweep_out), "metric": "val_accuracy"},
    )
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(cli.shutil, "copy", lambda src, dst: None)

    args = SimpleNamespace(
        sweep="dummy.yaml",
        base_config=None,
        trials=None,
        train_epochs=25,
        outdir=None,
        seed=None,
        log_level="INFO",
    )

    assert cli.run_auto_sweep_command(args) == 0
    assert commands
    assert "--trials" not in commands[0]


def test_run_train_command_persists_best_metric_metadata(tmp_path: Path, monkeypatch):
    import ndswin.cli as cli

    config_path = tmp_path / "cifar10.json"
    config_path.write_text(Path("configs/cifar10.json").read_text())

    class FakeModel:
        def __init__(self, config):
            self.config = config

        def init(self, rng, dummy_input, deterministic=True):
            return {"params": {"w": jnp.ones((2, 2))}}

    class FakeLoader:
        dataset_info = SimpleNamespace(name="fake", num_train=16, num_test=4)

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 1

        def reset(self):
            return None

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            self.best_epoch = 2
            self.best_metric_name = "val_accuracy"
            self.best_metric_value = 0.81
            self.best_metrics = {"loss": 0.2, "accuracy": 0.81, "top5_accuracy": 0.95}

        def fit(self, train_loader, val_loader, max_steps_per_epoch=None):
            return {"train_loss": [1.0], "val_accuracy": [0.81]}

        def evaluate(self, val_loader):
            return {"loss": 0.2, "accuracy": 0.81, "top5_accuracy": 0.95}

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    log_file = tmp_path / "train.log"

    monkeypatch.setattr("ndswin.utils.gpu.setup_optimal_gpus", lambda: None)
    monkeypatch.setattr(
        cli, "validate_experiment_dataset_contract", lambda *args, **kwargs: {"num_classes": 10}
    )
    monkeypatch.setattr(
        cli,
        "load_training_runtime",
        lambda: (FakeModel, FakeModel, FakeTrainer, lambda **kwargs: FakeLoader()),
    )
    monkeypatch.setattr(
        cli, "setup_output_dirs", lambda *args, **kwargs: (checkpoint_dir, log_file)
    )

    args = SimpleNamespace(
        config=str(config_path),
        epochs=None,
        batch_size=None,
        lr=None,
        seed=None,
        data_dir=None,
        max_steps_per_epoch=None,
        max_devices=None,
        stamp=None,
        no_log_file=True,
        log_level="INFO",
        overrides=[],
    )

    assert cli.run_train_command(args) == 0
    metrics = json.loads((checkpoint_dir / "metrics.json").read_text())
    assert metrics["best_epoch"] == 2
    assert metrics["best_metric_name"] == "val_accuracy"
    assert metrics["best_metric_value"] == 0.81
    assert metrics["best_metrics"]["accuracy"] == 0.81
    assert metrics["final_metrics"]["accuracy"] == 0.81


def test_run_validate_command_enables_validation_fallback_for_smoke_train(monkeypatch):
    import ndswin.cli as cli

    observed: dict[str, str | None] = {}

    monkeypatch.delenv("ALLOW_VALIDATION_SPLIT_FALLBACK", raising=False)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)

    def fake_train(args):
        observed["env"] = os.environ.get("ALLOW_VALIDATION_SPLIT_FALLBACK")
        return 0

    monkeypatch.setattr(cli, "run_train_command", fake_train)

    args = SimpleNamespace(
        config="configs/cifar10.json",
        skip_tests=True,
        skip_train=False,
        train_epochs=2,
        max_steps_per_epoch=None,
        max_devices=None,
        stamp=None,
        no_log_file=True,
        log_level="INFO",
        pytest_args=[],
        overrides=[],
    )

    assert cli.run_validate_command(args) == 0
    assert observed["env"] == "1"
    assert "ALLOW_VALIDATION_SPLIT_FALLBACK" not in os.environ
