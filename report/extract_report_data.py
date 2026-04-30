from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPORT_DIR.parent
DEFAULT_DATA_DIR = REPORT_DIR / "data"
DEFAULT_SOURCES_DIR = DEFAULT_DATA_DIR / "sources"

DEFAULT_ARTIFACTS = {
    "cifar_summary": "outputs/sweeps/cifar10_tuned_hyperparam_sweep/summary.json",
    "modelnet_summary": "outputs/sweeps/modelnet40_stable_hyperparam_sweep/summary.json",
    "cifar_metrics": "outputs/cifar10/cifar10_tuned_8021c956_20260429_064522/checkpoints/metrics.json",
    "modelnet_metrics": "outputs/volume_folder/modelnet40_f67c9398_20260429_165927/checkpoints/metrics.json",
    "cifar_best_config": "configs/auto_best/best_cifar10_cifar10_tuned_05f7e1b8_20260427_234823_trial003.json",
    "modelnet_best_config": (
        "configs/auto_best/best_volume_folder_modelnet40_e548c891_20260429_164145_trial011.json"
    ),
    "modelnet_manifest": "data/modelnet40/dataset_manifest.json",
    "queue_results": "logs/queue_20260427_214146.json",
    "benchmark_log": "logs/tmux/benchmark/benchmark_20260427_214145.log",
    "modelnet_resume_log": "logs/tmux/auto_sweep/autosweep_20260429_155416.log",
    "cifar_final_log": "logs/cifar10/cifar10_tuned_8021c956_20260429_064522.log",
    "modelnet_final_log": "logs/volume_folder/modelnet40_f67c9398_20260429_165927.log",
}

LATEST_PATTERNS = {
    "cifar_summary": "outputs/sweeps/cifar10_tuned_hyperparam_sweep/summary.json",
    "modelnet_summary": "outputs/sweeps/modelnet40_stable_hyperparam_sweep/summary.json",
    "cifar_metrics": "outputs/cifar10/*/checkpoints/metrics.json",
    "modelnet_metrics": "outputs/volume_folder/*/checkpoints/metrics.json",
    "cifar_best_config": "configs/auto_best/best_cifar10_*.json",
    "modelnet_best_config": "configs/auto_best/best_volume_folder_modelnet40_*.json",
    "modelnet_manifest": "data/modelnet40/dataset_manifest.json",
    "benchmark_log": "logs/tmux/benchmark/benchmark_*.log",
    "modelnet_resume_log": "logs/tmux/auto_sweep/autosweep_*.log",
    "cifar_final_log": "logs/cifar10/cifar10_tuned_*.log",
    "modelnet_final_log": "logs/volume_folder/modelnet40_*.log",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync report artifacts and regenerate derived CSV tables."
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Copy selected artifacts from --live-root into --sources-root before extraction.",
    )
    parser.add_argument(
        "--sync-latest",
        action="store_true",
        help="Resolve latest live benchmark artifacts before syncing/extracting.",
    )
    parser.add_argument("--live-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--sources-root", type=Path, default=DEFAULT_SOURCES_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cifar-summary", default=None)
    parser.add_argument("--modelnet-summary", default=None)
    parser.add_argument("--cifar-metrics", default=None)
    parser.add_argument("--modelnet-metrics", default=None)
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def latest_path(root: Path, pattern: str) -> str:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No artifact matches {pattern} under {root}")
    return str(matches[0].relative_to(root))


def build_artifact_map(args: argparse.Namespace) -> dict[str, str]:
    artifacts = dict(DEFAULT_ARTIFACTS)
    for key, attr in (
        ("cifar_summary", "cifar_summary"),
        ("modelnet_summary", "modelnet_summary"),
        ("cifar_metrics", "cifar_metrics"),
        ("modelnet_metrics", "modelnet_metrics"),
    ):
        explicit = getattr(args, attr)
        if explicit:
            artifacts[key] = explicit

    if args.sync_latest:
        for key, pattern in LATEST_PATTERNS.items():
            artifacts[key] = latest_path(args.live_root, pattern)

    return artifacts


def sync_artifacts(
    artifacts: dict[str, str], live_root: Path, sources_root: Path
) -> dict[str, str]:
    synced: dict[str, str] = {}
    for key, rel_path in artifacts.items():
        src = resolve_path(live_root, rel_path)
        if not src.exists():
            raise FileNotFoundError(f"Missing source artifact: {src}")
        try:
            dest_rel = src.relative_to(live_root)
        except ValueError:
            dest_rel = Path(src.name)
        dest = sources_root / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        synced[key] = str(dest_rel)
        print(f"synced {display_path(src)} -> {display_path(dest)}")
    return synced


def load_json(sources_root: Path, rel_path: str) -> object:
    return json.loads(resolve_path(sources_root, rel_path).read_text())


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {display_path(path)}")


def successful_trials(summary: object) -> list[dict[str, object]]:
    if isinstance(summary, dict):
        results = summary.get("results", [])
    else:
        results = summary
    if not isinstance(results, list):
        raise ValueError("Sweep summary results must be a list")
    return [
        row
        for row in results
        if isinstance(row, dict) and row.get("status") in {"success", "done", "completed"}
    ]


def build_sweep_csv(
    *,
    sources_root: Path,
    data_dir: Path,
    summary_rel: str,
    output_name: str,
    invalid_output_name: str | None = None,
) -> None:
    summary = load_json(sources_root, summary_rel)
    if isinstance(summary, dict):
        all_results = summary.get("results", [])
    else:
        all_results = summary
    if not isinstance(all_results, list):
        raise ValueError("Sweep summary results must be a list")
    results = successful_trials(summary)
    if not results:
        raise ValueError(f"No successful trials in {summary_rel}")
    best_trial = max(results, key=lambda row: row.get("val_accuracy", 0.0))["trial"]
    rows = [
        {
            "trial": row.get("trial"),
            "val_accuracy": row.get("val_accuracy", 0.0),
            "val_top5_accuracy": row.get("val_top5_accuracy", 0.0),
            "elapsed_seconds": row.get("elapsed_seconds", 0.0),
            "is_best": int(row.get("trial") == best_trial),
        }
        for row in results
    ]
    write_csv(
        data_dir / output_name,
        ["trial", "val_accuracy", "val_top5_accuracy", "elapsed_seconds", "is_best"],
        rows,
    )

    if invalid_output_name is not None:
        invalid_rows = [
            {"trial": row.get("trial"), "status_band": 0}
            for row in all_results
            if isinstance(row, dict) and row not in results
        ]
        write_csv(data_dir / invalid_output_name, ["trial", "status_band"], invalid_rows)


def build_training_curve(
    *,
    sources_root: Path,
    data_dir: Path,
    metrics_rel: str,
    output_name: str,
) -> None:
    metrics = load_json(sources_root, metrics_rel)
    if not isinstance(metrics, dict):
        raise ValueError("Metrics artifact must be a JSON object")
    history = metrics["history"]
    if not isinstance(history, dict):
        raise ValueError("Metrics history must be a JSON object")

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
                "train_top5_accuracy": history.get("train_top5_accuracy", [0.0] * epochs)[idx],
                "val_top5_accuracy": history.get("val_top5_accuracy", [0.0] * epochs)[idx],
            }
        )
    write_csv(
        data_dir / output_name,
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
    artifacts = build_artifact_map(args)
    if args.sync:
        artifacts = sync_artifacts(artifacts, args.live_root, args.sources_root)

    build_sweep_csv(
        sources_root=args.sources_root,
        data_dir=args.data_dir,
        summary_rel=artifacts["cifar_summary"],
        output_name="cifar10_sweep.csv",
    )
    build_sweep_csv(
        sources_root=args.sources_root,
        data_dir=args.data_dir,
        summary_rel=artifacts["modelnet_summary"],
        output_name="modelnet40_sweep_success.csv",
        invalid_output_name="modelnet40_sweep_invalid.csv",
    )
    build_training_curve(
        sources_root=args.sources_root,
        data_dir=args.data_dir,
        metrics_rel=artifacts["cifar_metrics"],
        output_name="cifar10_training.csv",
    )
    build_training_curve(
        sources_root=args.sources_root,
        data_dir=args.data_dir,
        metrics_rel=artifacts["modelnet_metrics"],
        output_name="modelnet40_training.csv",
    )


if __name__ == "__main__":
    main()
