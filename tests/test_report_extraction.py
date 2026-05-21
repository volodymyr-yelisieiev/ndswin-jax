"""Tests for report artifact extraction."""

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _metrics_payload() -> dict[str, object]:
    return {
        "history": {
            "train_accuracy": [0.5],
            "val_accuracy": [0.6],
            "train_loss": [1.0],
            "val_loss": [0.9],
            "train_top5_accuracy": [0.8],
            "val_top5_accuracy": [0.9],
        }
    }


def test_report_extractor_accepts_configurable_artifact_paths(tmp_path: Path):
    sources = tmp_path / "sources"
    out = tmp_path / "derived"
    cifar_summary = "outputs/sweeps/cifar/summary.json"
    cifar100_summary = "outputs/sweeps/cifar100/summary.json"
    modelnet_summary = "outputs/sweeps/modelnet/summary.json"
    cifar_metrics = "outputs/cifar10/run/checkpoints/metrics.json"
    cifar100_metrics = "outputs/cifar100/run/checkpoints/metrics.json"
    modelnet_metrics = "outputs/volume_folder/run/checkpoints/metrics.json"

    _write_json(
        sources / cifar_summary,
        {
            "metric": "val_accuracy",
            "results": [
                {
                    "trial": 0,
                    "status": "success",
                    "val_accuracy": 0.7,
                    "val_top5_accuracy": 0.9,
                    "elapsed_seconds": 1.0,
                }
            ],
        },
    )
    _write_json(
        sources / cifar100_summary,
        {
            "metric": "val_accuracy",
            "results": [
                {
                    "trial": 0,
                    "status": "success",
                    "val_accuracy": 0.65,
                    "val_top5_accuracy": 0.88,
                    "elapsed_seconds": 3.0,
                }
            ],
        },
    )
    _write_json(
        sources / modelnet_summary,
        {
            "metric": "val_accuracy",
            "results": [
                {
                    "trial": 0,
                    "status": "success",
                    "val_accuracy": 0.8,
                    "val_top5_accuracy": 0.95,
                    "elapsed_seconds": 2.0,
                }
            ],
        },
    )
    _write_json(sources / cifar_metrics, _metrics_payload())
    _write_json(sources / cifar100_metrics, _metrics_payload())
    _write_json(sources / modelnet_metrics, _metrics_payload())

    result = subprocess.run(
        [
            sys.executable,
            "report/extract_report_data.py",
            "--sources-root",
            str(sources),
            "--data-dir",
            str(out),
            "--cifar-summary",
            cifar_summary,
            "--cifar100-summary",
            cifar100_summary,
            "--modelnet-summary",
            modelnet_summary,
            "--cifar-metrics",
            cifar_metrics,
            "--cifar100-metrics",
            cifar100_metrics,
            "--modelnet-metrics",
            modelnet_metrics,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (out / "cifar10_sweep.csv").exists()
    assert (out / "cifar100_sweep.csv").exists()
    assert (out / "modelnet40_sweep_success.csv").exists()
    assert (out / "cifar10_training.csv").exists()
    assert (out / "cifar100_training.csv").exists()
    assert "0.7" in (out / "cifar10_sweep.csv").read_text()


def test_report_extractor_syncs_latest_report_source_bundle(tmp_path: Path):
    live = tmp_path / "live"
    sources = tmp_path / "sources"
    out = tmp_path / "derived"

    _write_json(
        live / "outputs/sweeps/cifar10_tuned_hyperparam_sweep/summary.json",
        {
            "metric": "val_accuracy",
            "results": [
                {
                    "trial": 3,
                    "status": "success",
                    "val_accuracy": 0.7344,
                    "val_top5_accuracy": 0.9672,
                    "elapsed_seconds": 1.0,
                }
            ],
        },
    )
    _write_json(
        live / "outputs/sweeps/cifar100_tuned_hyperparam_sweep_fixed_20260519/summary.json",
        {
            "metric": "val_accuracy",
            "results": [
                {
                    "trial": 11,
                    "status": "success",
                    "val_accuracy": 0.5902,
                    "val_top5_accuracy": 0.8612,
                    "elapsed_seconds": 3.0,
                }
            ],
        },
    )
    _write_json(
        live / "outputs/sweeps/modelnet40_stable_hyperparam_sweep/summary.json",
        {
            "metric": "val_accuracy",
            "results": [
                {
                    "trial": 11,
                    "status": "success",
                    "val_accuracy": 0.8061,
                    "val_top5_accuracy": 0.9499,
                    "elapsed_seconds": 2.0,
                }
            ],
        },
    )
    _write_json(
        live / "outputs/cifar10/cifar10_rngfix_retrain600_20260518/checkpoints/metrics.json",
        _metrics_payload(),
    )
    _write_json(
        live / "outputs/cifar10/cifar10_rngfix_retrain600_20260518/checkpoints/test_metrics.json",
        {"split": "test", "metrics": {"accuracy": 0.9}},
    )
    _write_json(
        live / "outputs/cifar100/cifar100_tuned_0fd87157_20260520_112244/checkpoints/metrics.json",
        _metrics_payload(),
    )
    _write_json(
        live
        / "outputs/cifar100/cifar100_tuned_0fd87157_20260520_112244/checkpoints/test_metrics.json",
        {"split": "test", "metrics": {"accuracy": 0.69}},
    )
    _write_json(
        live / "outputs/volume_folder/modelnet40_f67c9398_20260429_165927/checkpoints/metrics.json",
        _metrics_payload(),
    )
    _write_json(
        live
        / "outputs/volume_folder/modelnet40_f67c9398_20260429_165927/checkpoints/test_metrics.json",
        {"split": "test", "metrics": {"accuracy": 0.69}},
    )
    _write_json(live / "configs/auto_best/best_cifar10_trial003.json", {"dataset": "cifar10"})
    _write_json(
        live / "configs/auto_best/best_cifar100_cifar100_tuned_a9ef3d19_trial011.json",
        {"dataset": "cifar100"},
    )
    _write_json(
        live / "configs/auto_best/best_volume_folder_modelnet40_trial011.json",
        {"dataset": "modelnet40"},
    )
    _write_json(
        live / "data/modelnet40/dataset_manifest.json",
        {"split_counts": {"train": 1, "validation": 1, "test": 1}},
    )
    for rel in (
        "logs/tmux/auto_sweep/autosweep_20260519_214743.log",
        "logs/tmux/auto_sweep/autosweep_20260429_155416.log",
        "logs/cifar10/cifar10_rngfix_retrain600_20260518.log",
        "logs/cifar100/cifar100_tuned_0fd87157_20260520_112244.log",
        "logs/volume_folder/modelnet40_f67c9398_20260429_165927.log",
    ):
        path = live / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok")

    result = subprocess.run(
        [
            sys.executable,
            "report/extract_report_data.py",
            "--live-root",
            str(live),
            "--sources-root",
            str(sources),
            "--data-dir",
            str(out),
            "--sync",
            "--sync-latest",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (sources / "configs/auto_best/best_cifar10_trial003.json").exists()
    assert (
        sources / "configs/auto_best/best_cifar100_cifar100_tuned_a9ef3d19_trial011.json"
    ).exists()
    assert (sources / "data/modelnet40/dataset_manifest.json").exists()
    assert (sources / "logs/tmux/auto_sweep/autosweep_20260429_155416.log").exists()
    assert (sources / "logs/tmux/auto_sweep/autosweep_20260519_214743.log").exists()
    assert not (sources / "logs/queue_20260427_214146.json").exists()
    assert not (sources / "logs/tmux/benchmark/benchmark_20260427_214145.log").exists()
    assert "0.7344" in (out / "cifar10_sweep.csv").read_text()
    assert "0.5902" in (out / "cifar100_sweep.csv").read_text()
