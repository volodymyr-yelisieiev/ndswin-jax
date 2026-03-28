import json
from pathlib import Path

import pytest

from ndswin.cli import get_best_trial_from_summary


def test_get_best_trial_accuracy(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    data = [
        {"trial": 0, "status": "done", "val_accuracy": 0.5, "val_top5_accuracy": 0.7, "loss": 1.0},
        {"trial": 1, "status": "done", "val_accuracy": 0.8, "val_top5_accuracy": 0.9, "loss": 0.5},
        {"trial": 2, "status": "done", "val_accuracy": 0.4, "val_top5_accuracy": 0.6, "loss": 1.2},
        {
            "trial": 3,
            "status": "error",
            "val_accuracy": 0.9,
            "val_top5_accuracy": 0.99,
            "loss": 0.1,
        },  # Error should be ignored
    ]
    summary_path.write_text(json.dumps(data))

    best = get_best_trial_from_summary(summary_path)
    assert best["trial"] == 1
    assert best["val_accuracy"] == 0.8


def test_get_best_trial_dice(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    data = [
        {"trial": 0, "status": "done", "val_dice": 0.5, "loss": 1.0},
        {"trial": 1, "status": "done", "val_dice": 0.4, "loss": 0.5},
        {"trial": 2, "status": "done", "val_dice": 0.82, "loss": 1.2},
    ]
    summary_path.write_text(json.dumps(data))

    best = get_best_trial_from_summary(summary_path)
    assert best["trial"] == 2
    assert best["val_dice"] == 0.82


def test_get_best_trial_loss(tmp_path: Path):
    # Tests the fallback
    summary_path = tmp_path / "summary.json"
    data = [
        {"trial": 0, "status": "done", "loss": 1.0},
        {"trial": 1, "status": "done", "loss": 0.4},
        {"trial": 2, "status": "done", "loss": 0.5},
    ]
    summary_path.write_text(json.dumps(data))

    best = get_best_trial_from_summary(summary_path)
    assert best["trial"] == 1
    assert best["loss"] == 0.4


def test_get_best_trial_respects_explicit_metric(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    data = [
        {"trial": 0, "status": "done", "val_accuracy": 0.9, "val_top5_accuracy": 0.92},
        {"trial": 1, "status": "done", "val_accuracy": 0.8, "val_top5_accuracy": 0.98},
    ]
    summary_path.write_text(json.dumps(data))

    best = get_best_trial_from_summary(summary_path, metric="val_top5_accuracy")
    assert best["trial"] == 1
    assert best["val_top5_accuracy"] == 0.98


def test_get_best_trial_reads_metric_from_summary_payload(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    payload = {
        "metric": "val_top5_accuracy",
        "results": [
            {"trial": 0, "status": "done", "val_accuracy": 0.9, "val_top5_accuracy": 0.92},
            {"trial": 1, "status": "done", "val_accuracy": 0.8, "val_top5_accuracy": 0.98},
        ],
    }
    summary_path.write_text(json.dumps(payload))

    best = get_best_trial_from_summary(summary_path)
    assert best["trial"] == 1
    assert best["val_top5_accuracy"] == 0.98


def test_get_best_trial_empty_or_missing(tmp_path: Path):
    summary_path = tmp_path / "summary.json"

    with pytest.raises(FileNotFoundError):
        get_best_trial_from_summary(summary_path)

    summary_path.write_text(json.dumps([]))
    with pytest.raises(ValueError, match="Sweep summary is empty"):
        get_best_trial_from_summary(summary_path)

    summary_path.write_text(json.dumps([{"trial": 0, "status": "error"}]))
    with pytest.raises(ValueError, match="No successful trials found"):
        get_best_trial_from_summary(summary_path)
