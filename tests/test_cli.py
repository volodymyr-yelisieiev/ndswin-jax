"""Smoke tests for the package CLI entrypoint."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
