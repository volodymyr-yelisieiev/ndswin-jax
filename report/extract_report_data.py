from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPORT_DIR.parent
DATA_DIR = REPORT_DIR / "data"
SOURCES_DIR = DATA_DIR / "sources"

ARTIFACTS = [
    "outputs/sweeps/cifar10_tuned_hyperparam_sweep/summary.json",
    "outputs/sweeps/modelnet40_stable_hyperparam_sweep/summary.json",
    "outputs/cifar10/cifar10_tuned_4b2313b9_20260401_032614/checkpoints/metrics.json",
    "outputs/volume_folder/modelnet40_f67c9398_20260401_060749/checkpoints/metrics.json",
    "logs/queue_20260330_174711.json",
    "logs/tmux/queue/queue_20260330_174711.log",
    "logs/cifar10/cifar10_tuned_4b2313b9_20260401_032614.log",
    "logs/volume_folder/modelnet40_f67c9398_20260401_060749.log",
    "data/modelnet40/dataset_manifest.json",
    "configs/auto_best/best_cifar10_cifar10_tuned_07366165_20260331_090657_trial022.json",
    "configs/auto_best/best_volume_folder_modelnet40_e548c891_20260401_055016_trial011.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync committed report artifacts and regenerate derived CSV tables."
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Copy the preserved raw artifacts from the repository into report/data/sources before regenerating CSVs.",
    )
    return parser.parse_args()


def sync_artifacts() -> None:
    for rel_path in ARTIFACTS:
        src = REPO_ROOT / rel_path
        if not src.exists():
            raise FileNotFoundError(f"Missing source artifact: {src}")
        dest = SOURCES_DIR / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"synced {rel_path}")


def load_json(rel_path: str) -> object:
    return json.loads((SOURCES_DIR / rel_path).read_text())


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def build_cifar10_sweep() -> None:
    summary = load_json("outputs/sweeps/cifar10_tuned_hyperparam_sweep/summary.json")
    assert isinstance(summary, dict)
    results = [row for row in summary["results"] if row.get("status") == "success"]
    best_trial = max(results, key=lambda row: row["val_accuracy"])["trial"]
    rows = [
        {
            "trial": row["trial"],
            "val_accuracy": row["val_accuracy"],
            "val_top5_accuracy": row["val_top5_accuracy"],
            "elapsed_seconds": row["elapsed_seconds"],
            "is_best": int(row["trial"] == best_trial),
        }
        for row in results
    ]
    write_csv(
        DATA_DIR / "cifar10_sweep.csv",
        ["trial", "val_accuracy", "val_top5_accuracy", "elapsed_seconds", "is_best"],
        rows,
    )


def build_modelnet40_sweep() -> None:
    summary = load_json("outputs/sweeps/modelnet40_stable_hyperparam_sweep/summary.json")
    assert isinstance(summary, dict)
    results = summary["results"]
    successful = [row for row in results if row.get("status") == "success"]
    invalid = [row for row in results if row.get("status") != "success"]
    best_trial = max(successful, key=lambda row: row["val_accuracy"])["trial"]
    success_rows = [
        {
            "trial": row["trial"],
            "val_accuracy": row["val_accuracy"],
            "val_top5_accuracy": row["val_top5_accuracy"],
            "elapsed_seconds": row["elapsed_seconds"],
            "is_best": int(row["trial"] == best_trial),
        }
        for row in successful
    ]
    invalid_rows = [{"trial": row["trial"], "status_band": 0} for row in invalid]
    write_csv(
        DATA_DIR / "modelnet40_sweep_success.csv",
        ["trial", "val_accuracy", "val_top5_accuracy", "elapsed_seconds", "is_best"],
        success_rows,
    )
    write_csv(
        DATA_DIR / "modelnet40_sweep_invalid.csv",
        ["trial", "status_band"],
        invalid_rows,
    )


def build_training_curve(rel_path: str, output_name: str) -> None:
    metrics = load_json(rel_path)
    assert isinstance(metrics, dict)
    history = metrics["history"]
    assert isinstance(history, dict)
    epochs = len(history["train_accuracy"])
    rows = []
    for idx in range(epochs):
        rows.append(
            {
                "epoch": idx + 1,
                "train_accuracy": history["train_accuracy"][idx],
                "val_accuracy": history["val_accuracy"][idx],
                "train_loss": history["train_loss"][idx],
                "val_loss": history["val_loss"][idx],
                "train_top5_accuracy": history["train_top5_accuracy"][idx],
                "val_top5_accuracy": history["val_top5_accuracy"][idx],
            }
        )
    write_csv(
        DATA_DIR / output_name,
        [
            "epoch",
            "train_accuracy",
            "val_accuracy",
            "train_loss",
            "val_loss",
            "train_top5_accuracy",
            "val_top5_accuracy",
        ],
        rows,
    )


def main() -> None:
    args = parse_args()
    if args.sync:
        sync_artifacts()
    build_cifar10_sweep()
    build_modelnet40_sweep()
    build_training_curve(
        "outputs/cifar10/cifar10_tuned_4b2313b9_20260401_032614/checkpoints/metrics.json",
        "cifar10_training.csv",
    )
    build_training_curve(
        "outputs/volume_folder/modelnet40_f67c9398_20260401_060749/checkpoints/metrics.json",
        "modelnet40_training.csv",
    )


if __name__ == "__main__":
    main()
