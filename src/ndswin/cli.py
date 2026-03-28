from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from contextlib import suppress
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from .utils.logging import get_logger, setup_logging

yaml: Any | None
try:
    yaml = importlib.import_module("yaml")
except Exception:  # pragma: no cover - optional dependency at runtime
    yaml = None

logger = get_logger("ndswin.cli")
DEFAULT_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


def add_log_level_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=DEFAULT_LOG_LEVELS,
        help="Logging verbosity level",
    )


def add_config_argument(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument(
        "--config",
        type=str,
        required=required,
        help="Path to JSON configuration file",
    )


def add_sweep_argument(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument(
        "--sweep",
        type=str,
        required=required,
        help="Path to sweep configuration file (YAML or JSON)",
    )


def add_queue_argument(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument(
        "--queue",
        type=str,
        required=required,
        help="Path to queue definition file (YAML or JSON)",
    )


def add_train_override_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Cap the number of training steps per epoch",
    )
    parser.add_argument(
        "--max-devices",
        type=int,
        default=None,
        help="Limit training to at most this many visible JAX devices",
    )
    parser.add_argument(
        "--stamp",
        type=str,
        default=None,
        help="Override experiment stamp used for outputs and log files",
    )
    parser.add_argument("--no-log-file", action="store_true", help="Disable logging to file")


def parse_json_override(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value


class ParseOverrideAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        overrides = cast(list[tuple[str, Any]] | None, getattr(namespace, self.dest, None))
        if overrides is None:
            overrides = []
        if values is None or isinstance(values, str) or len(values) != 2:
            raise ValueError("Override action expects a key/value pair.")
        key = values[0]
        raw_value = values[1]
        overrides.append((str(key), parse_json_override(str(raw_value))))
        setattr(namespace, self.dest, overrides)


def add_overrides_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o",
        "--override",
        dest="overrides",
        action=ParseOverrideAction,
        nargs=2,
        metavar=("PATH", "VALUE"),
        default=[],
        help="Override any config field using dotted paths, for example: -o training.epochs 2",
    )


def set_by_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        if isinstance(cur, dict):
            if part not in cur or not isinstance(cur[part], dict):
                cur[part] = {}
            cur = cur[part]
            continue
        cur = getattr(cur, part)
    last = parts[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        setattr(cur, last, value)


def apply_path_overrides(exp_config: Any, overrides: Sequence[tuple[str, Any]]) -> None:
    for path, value in overrides:
        set_by_path(exp_config, path, value)


def configure_logging(
    log_level: str, *, log_file: str | None = None, log_to_file: bool = False
) -> None:
    setup_logging(
        level=log_level,
        log_file=log_file,
        log_to_file=log_to_file,
        log_to_console=True,
        use_rich=False,
    )


def load_experiment_config(config_path: str) -> Any:
    from .config import ExperimentConfig

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config_dict = cast(dict[str, Any], json.loads(path.read_text()))
    if "dataset" in config_dict and "data" not in config_dict:
        config_dict["data"] = config_dict.pop("dataset")
    if "augmentation" in config_dict:
        config_dict.setdefault("data", {}).update(config_dict.pop("augmentation"))
    return ExperimentConfig.from_dict(config_dict)


def apply_train_overrides(exp_config: Any, args: argparse.Namespace) -> None:
    if args.epochs is not None:
        exp_config.training.epochs = args.epochs
    if args.batch_size is not None:
        exp_config.training.batch_size = args.batch_size
    if args.lr is not None:
        exp_config.training.learning_rate = args.lr
    if args.seed is not None:
        exp_config.training.seed = args.seed
    if args.data_dir is not None:
        exp_config.data.data_dir = args.data_dir
    apply_path_overrides(exp_config, args.overrides)


def setup_output_dirs(
    exp_config: Any, config_path: str, stamp: str | None = None
) -> tuple[Path, Path]:
    if stamp is None:
        stamp = exp_config.get_stamp()

    dataset_name = (
        getattr(exp_config.data, "hf_id", None)
        or getattr(exp_config.data, "dataset", None)
        or "dataset"
    )
    dataset_name = str(dataset_name).replace("/", "_").replace(":", "_")

    checkpoint_dir = Path("outputs") / dataset_name / stamp / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_copy_path = Path("outputs") / dataset_name / stamp / "config.json"
    shutil.copy(config_path, config_copy_path)

    logs_dir = Path("logs") / dataset_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{stamp}.log"
    return checkpoint_dir, log_file


def validate_experiment_dataset_contract(
    exp_config: Any, *, require_validation_split: bool = False
) -> dict[str, Any] | None:
    """Validate dataset/model compatibility before launching any work."""
    from .training.data import validate_dataset_contract

    model_num_classes = getattr(exp_config.model, "num_classes", None)
    if model_num_classes is not None:
        exp_config.training.num_classes = int(model_num_classes)

    return validate_dataset_contract(
        exp_config.data,
        expected_num_classes=model_num_classes,
        require_validation_split=require_validation_split,
    )


def load_training_runtime() -> tuple[Any, Any, Any, Any]:
    """Import training runtime dependencies lazily.

    This keeps GPU selection logic ahead of any JAX-backed imports so
    ``setup_optimal_gpus`` can still influence backend discovery.
    """
    from .models.classifier import SwinForSegmentation
    from .models.swin import NDSwinTransformer
    from .training import Trainer, create_data_loader

    return SwinForSegmentation, NDSwinTransformer, Trainer, create_data_loader


def run_train_command(args: argparse.Namespace) -> int:
    from .utils.gpu import setup_optimal_gpus

    setup_optimal_gpus()

    exp_config = load_experiment_config(args.config)
    apply_train_overrides(exp_config, args)

    configure_logging(args.log_level, log_to_file=False)
    try:
        dataset_contract = validate_experiment_dataset_contract(
            exp_config,
            require_validation_split=True,
        )
    except Exception as exc:
        logger.error("Dataset/config validation failed: %s", exc)
        return 1

    SwinForSegmentation, NDSwinTransformer, Trainer, create_data_loader = load_training_runtime()

    import jax
    import jax.numpy as jnp
    import numpy as np

    checkpoint_dir, log_file = setup_output_dirs(exp_config, args.config, stamp=args.stamp)
    configure_logging(
        args.log_level,
        log_file=None if args.no_log_file else str(log_file),
        log_to_file=not args.no_log_file,
    )

    train_logger = get_logger("train")
    train_logger.info("=" * 70)
    train_logger.info("NDSwin-JAX Training")
    train_logger.info("=" * 70)
    train_logger.info("Config file: %s", args.config)
    train_logger.info("Experiment: %s", exp_config.name)
    train_logger.info("Description: %s", exp_config.description)
    train_logger.info("Config hash: %s", exp_config.get_config_hash())
    train_logger.info("Checkpoint dir: %s", checkpoint_dir)
    train_logger.info("Log file: %s", log_file)
    train_logger.info("Timestamp: %s", datetime.now().isoformat())
    if dataset_contract is not None and isinstance(dataset_contract.get("num_classes"), int):
        train_logger.info("Dataset classes: %d", dataset_contract["num_classes"])

    rng = jax.random.PRNGKey(exp_config.training.seed)
    np.random.seed(exp_config.training.seed)

    model_config = exp_config.model
    train_config = exp_config.training
    task = getattr(exp_config.data, "task", "classification")

    train_logger.info("-" * 70)
    train_logger.info("Dataset Configuration:")
    for key, value in exp_config.data.to_dict().items():
        train_logger.debug("  %s: %s", key, value)

    train_logger.info("-" * 70)
    train_logger.info("Model Configuration:")
    for field_name in (
        "num_dims",
        "patch_size",
        "window_size",
        "embed_dim",
        "depths",
        "num_heads",
        "num_classes",
    ):
        train_logger.info("  %s: %s", field_name, getattr(model_config, field_name))
    train_logger.debug("  drop_path_rate: %s", model_config.drop_path_rate)

    train_logger.info("-" * 70)
    train_logger.info("Training Configuration:")
    train_logger.info("  epochs: %s", train_config.num_epochs)
    train_logger.info("  batch_size: %s", train_config.batch_size)
    train_logger.info("  learning_rate: %s", train_config.learning_rate)
    train_logger.debug("  weight_decay: %s", train_config.weight_decay)
    train_logger.debug("  optimizer: %s", train_config.optimizer)
    train_logger.debug("  lr_schedule: %s", train_config.lr_schedule)
    train_logger.debug("  warmup_epochs: %s", train_config.warmup_epochs)
    train_logger.debug("  label_smoothing: %s", train_config.label_smoothing)

    if task == "segmentation":
        model = SwinForSegmentation(config=model_config, num_classes=model_config.num_classes or 2)
    else:
        model = NDSwinTransformer(config=model_config)

    input_shape = exp_config.data.image_size
    if len(input_shape) == exp_config.model.num_dims:
        input_shape = (exp_config.data.in_channels,) + input_shape
    dummy_input = jnp.ones((1,) + input_shape)
    variables = model.init(rng, dummy_input, deterministic=True)
    num_params = sum(param.size for param in jax.tree_util.tree_leaves(variables["params"]))
    train_logger.info("Total parameters: %s", f"{num_params:,}")

    train_logger.info("Loading data...")
    try:
        patch_size = exp_config.model.patch_size
        num_stages = exp_config.model.num_stages
        pad_to = tuple(int(patch_size[i] * (2 ** (num_stages - 1))) for i in range(len(patch_size)))
    except Exception:
        pad_to = None

    dataset_name = getattr(exp_config.data, "dataset", "unknown")
    try:
        train_loader = create_data_loader(
            config=exp_config.data,
            split="train",
            batch_size=exp_config.training.batch_size,
            pad_to=pad_to,
        )
        val_loader = create_data_loader(
            config=exp_config.data,
            split="validation",
            batch_size=exp_config.training.batch_size,
            pad_to=pad_to,
        )
    except Exception as exc:
        train_logger.error("Failed to create required data loaders: %s", exc)
        return 1

    split_counts = dataset_contract.get("split_counts") if dataset_contract is not None else None
    if hasattr(train_loader, "dataset_info"):
        train_logger.info("  Dataset: %s", train_loader.dataset_info.name)
        if isinstance(split_counts, dict):
            train_logger.info("  Train samples: %s", split_counts.get("train", 0))
            train_logger.info(
                "  Validation samples: %s",
                split_counts.get("validation", split_counts.get("val", 0)),
            )
            train_logger.info("  Test samples: %s", split_counts.get("test", 0))
        else:
            dataset_info = train_loader.dataset_info
            train_logger.info("  Train samples: %s", getattr(dataset_info, "num_train", 0))
            train_logger.info("  Validation samples: %s", getattr(dataset_info, "num_val", 0))
            train_logger.info("  Test samples: %s", getattr(dataset_info, "num_test", 0))
    else:
        train_logger.info("  Dataset: %s", dataset_name)

    mixup_transform = None
    mixup_alpha = getattr(exp_config.data, "mixup_alpha", 0.0)
    cutmix_alpha = getattr(exp_config.data, "cutmix_alpha", 0.0)
    if mixup_alpha > 0 or cutmix_alpha > 0:
        from .training.augmentation import MixupOrCutmix

        num_classes = model_config.num_classes or 100
        mixup_prob = (
            0.5 if (mixup_alpha > 0 and cutmix_alpha > 0) else (1.0 if mixup_alpha > 0 else 0.0)
        )
        mixup_transform = MixupOrCutmix(
            p=mixup_prob,
            mixup_alpha=mixup_alpha if mixup_alpha > 0 else 1.0,
            cutmix_alpha=cutmix_alpha if cutmix_alpha > 0 else 1.0,
            num_classes=num_classes,
        )

    devices = None
    if args.max_devices is not None:
        devices = list(jax.devices())[: args.max_devices]

    trainer = Trainer(
        model=model,
        config=train_config,
        seed=exp_config.training.seed,
        log_every=exp_config.training.log_interval,
        eval_every=exp_config.training.eval_interval,
        checkpoint_dir=str(checkpoint_dir),
        task=task,
        loss_name=getattr(exp_config.training, "loss", "cross_entropy"),
        mixup_transform=mixup_transform,
        use_tensorboard=True,
        devices=devices,
    )

    train_logger.info("=" * 70)
    train_logger.info("Starting training for %d epochs", train_config.num_epochs)
    train_logger.info("=" * 70)

    start_time = time.time()
    history = trainer.fit(train_loader, val_loader, max_steps_per_epoch=args.max_steps_per_epoch)
    total_time = time.time() - start_time

    final_metrics: dict[str, Any] = {}
    val_loader.reset()
    final_metrics = trainer.evaluate(val_loader)
    train_logger.info("=" * 70)
    train_logger.info("Final Validation Results (best-restored state):")
    train_logger.info("=" * 70)
    train_logger.info("  Loss: %.4f", final_metrics["loss"])
    if task == "segmentation":
        train_logger.info("  Dice: %.4f", final_metrics.get("val_dice", 0.0))
        train_logger.info("  Voxel Acc: %.4f", final_metrics.get("val_voxel_accuracy", 0.0))
    else:
        train_logger.info("  Accuracy: %.4f", final_metrics["accuracy"])
        train_logger.info("  Top-5 Accuracy: %.4f", final_metrics["top5_accuracy"])

    metrics_file = checkpoint_dir / "metrics.json"
    final_metrics_data: dict[str, float | None] = {"loss": float(final_metrics["loss"])}
    if task == "segmentation":
        final_metrics_data["val_dice"] = float(final_metrics.get("val_dice", 0.0))
        final_metrics_data["val_voxel_accuracy"] = float(
            final_metrics.get("val_voxel_accuracy", 0.0)
        )
    else:
        final_metrics_data["accuracy"] = float(final_metrics.get("accuracy", 0.0))
        final_metrics_data["top5_accuracy"] = float(final_metrics.get("top5_accuracy", 0.0))

    metrics_data = {
        "config": exp_config.to_dict(),
        "final_metrics": final_metrics_data,
        "best_epoch": trainer.best_epoch,
        "best_metric_name": trainer.best_metric_name,
        "best_metric_value": trainer.best_metric_value,
        "best_metrics": trainer.best_metrics,
        "training_time_seconds": total_time,
        "num_parameters": num_params,
        "history": {key: [float(value) for value in values] for key, values in history.items()},
    }
    metrics_file.write_text(json.dumps(metrics_data, indent=2))

    train_logger.info("Training completed in %.2fs", total_time)
    train_logger.info("Metrics saved to: %s", metrics_file)
    train_logger.info("Done!")
    return 0


def load_sweep(path: str) -> dict[str, Any]:
    sweep_path = Path(path)
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep file not found: {path}")
    if sweep_path.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML sweep files. Install pyyaml.")
        return cast(dict[str, Any], yaml.safe_load(sweep_path.read_text()))
    if sweep_path.suffix == ".json":
        return cast(dict[str, Any], json.loads(sweep_path.read_text()))
    raise ValueError("Unsupported sweep file format. Use .yaml/.yml or .json")


def sample_value(spec: dict[str, Any]) -> Any:
    kind = spec.get("kind", "choice")
    if kind == "choice":
        return random.choice(spec["values"])
    if kind == "uniform":
        return random.uniform(float(spec["min"]), float(spec["max"]))
    if kind == "log_uniform":
        minimum = float(spec["min"])
        maximum = float(spec["max"])
        return 10 ** random.uniform(math.log10(minimum), math.log10(maximum))
    raise ValueError(f"Unknown sampling kind: {kind}")


def materialize_experiment(base: Any, sampled: dict[str, Any], budget_epochs: int) -> Any:
    from .config import NDSwinConfig

    exp = deepcopy(base)
    apply_path_overrides(exp, list(sampled.items()))
    exp.training.epochs = int(budget_epochs)
    if hasattr(exp.training, "warmup_epochs"):
        exp.training.warmup_epochs = max(
            0,
            min(int(exp.training.warmup_epochs), int(exp.training.epochs)),
        )

    if getattr(exp.data, "dataset", "").lower() == "cifar100":
        exp.training.num_classes = 100
        exp.model.num_classes = 100

    def coerce_number(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value
        return value

    for field_name in (
        "learning_rate",
        "min_learning_rate",
        "weight_decay",
        "ema_decay",
        "stochastic_depth_rate",
        "label_smoothing",
    ):
        if hasattr(exp.training, field_name):
            setattr(exp.training, field_name, coerce_number(getattr(exp.training, field_name)))

    if isinstance(exp.training.batch_size, str):
        with suppress(ValueError):
            exp.training.batch_size = int(exp.training.batch_size)

    for field_name in ("mixup_alpha", "cutmix_alpha", "cutout_size"):
        if hasattr(exp.data, field_name):
            setattr(exp.data, field_name, coerce_number(getattr(exp.data, field_name)))

    if hasattr(exp.model, "drop_path_rate"):
        exp.model.drop_path_rate = coerce_number(exp.model.drop_path_rate)

    try:
        _ = NDSwinConfig.from_dict(exp.model.to_dict())
    except Exception as exc:  # pragma: no cover - validation depends on sampled values
        raise ValueError(f"Invalid model configuration after sampling: {exc}") from exc

    return exp


def build_sweep_summary_payload(
    results: list[dict[str, Any]],
    *,
    sweep: dict[str, Any],
    trials: int,
    budget_epochs: int,
    outdir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Build a sweep summary payload with selection metadata."""
    return {
        "metric": sweep.get("metric"),
        "trials": trials,
        "budget_epochs": budget_epochs,
        "output_dir": str(outdir),
        "mode": "dry-run" if dry_run else "run",
        "results": results,
    }


def write_sweep_summary(
    path: Path,
    results: list[dict[str, Any]],
    *,
    sweep: dict[str, Any],
    trials: int,
    budget_epochs: int,
    outdir: Path,
    dry_run: bool,
) -> None:
    """Write a sweep summary payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            build_sweep_summary_payload(
                results,
                sweep=sweep,
                trials=trials,
                budget_epochs=budget_epochs,
                outdir=outdir,
                dry_run=dry_run,
            ),
            indent=2,
        )
    )


def load_base_experiment(path: str | None) -> Any:
    from .config import ExperimentConfig, NDSwinConfig

    if path is None:
        cfg = ExperimentConfig()
        cfg.data.dataset = "cifar10"
        cfg.data.data_dir = "data"
        cfg.data.image_size = (32, 32)
        cfg.data.in_channels = 3
        cfg.training.batch_size = 128
        cfg.training.epochs = 100
        cfg.training.learning_rate = 1e-3
        cfg.model = NDSwinConfig.swin_tiny_2d(num_classes=100)
        cfg.name = "cifar10_sweep_base"
        return cfg

    exp_raw = cast(dict[str, Any], json.loads(Path(path).read_text()))
    exp_dict: dict[str, Any] = {"name": exp_raw.get("name", "sweep_base")}
    if "model" in exp_raw:
        exp_dict["model"] = exp_raw["model"]
    if "training" in exp_raw:
        exp_dict["training"] = exp_raw["training"]

    data = exp_raw.get("data") or exp_raw.get("dataset") or {}
    augmentation = exp_raw.get("augmentation", {})
    merged = {**data, **augmentation}
    for key in ("name", "description", "num_classes"):
        merged.pop(key, None)

    if "input_shape" in merged:
        input_shape = merged.pop("input_shape")
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2:
            merged["in_channels"] = int(input_shape[0])
            merged["image_size"] = tuple(int(x) for x in input_shape[1:])

    allowed_data_keys = {
        "dataset",
        "data_dir",
        "download",
        "image_size",
        "in_channels",
        "num_workers",
        "prefetch_size",
        "shuffle_buffer_size",
        "pin_memory",
        "drop_last",
        "train_split",
        "val_split",
        "test_split",
        "augmentation",
        "normalize",
        "mean",
        "std",
        "random_crop",
        "random_flip",
        "random_rotation",
        "color_jitter",
        "auto_augment",
        "mixup_alpha",
        "cutmix_alpha",
        "cutout_size",
        "task",
        "hf_id",
    }
    merged = {key: value for key, value in merged.items() if key in allowed_data_keys}
    if merged:
        exp_dict["data"] = merged
    return ExperimentConfig.from_dict(exp_dict)


def build_cli_invocation(
    command: str, *extra_args: str, python_exe: str | None = None
) -> list[str]:
    executable = python_exe or sys.executable
    return [executable, "-m", "ndswin.cli", command, *extra_args]


def run_trial(
    trial_idx: int,
    exp: Any,
    out_dir: Path,
    dry_run: bool = False,
    sweep_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_name = (
        getattr(exp.data, "hf_id", None) or getattr(exp.data, "dataset", None) or "dataset"
    )
    dataset_name = str(dataset_name).replace("/", "_").replace(":", "_")

    stamp_factory = getattr(exp, "get_stamp", None)
    stamp = stamp_factory() if callable(stamp_factory) else "unstamped"

    base_dir = out_dir / dataset_name / stamp
    if base_dir.exists() and dry_run:
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    trial_dir = base_dir / f"trial_{trial_idx:03d}"
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=False)

    config_path = trial_dir / "config.json"
    config_path.write_text(json.dumps(exp.to_dict(), indent=2))

    if dry_run:
        logger.info("[DRY] Trial %d config written to %s", trial_idx, config_path)
        return {
            "trial": trial_idx,
            "status": "dry",
            "config_path": str(config_path),
            "dataset": dataset_name,
            "stamp": stamp,
        }

    max_steps = str((sweep_config or {}).get("max_steps_per_epoch", 200))
    train_cmd = build_cli_invocation(
        "train",
        "--config",
        str(config_path),
        "--epochs",
        str(exp.training.epochs),
        "--stamp",
        f"{stamp}_trial_{trial_idx:03d}",
        "--max-steps-per-epoch",
        max_steps,
    )

    log_file_path = trial_dir / "train.log"
    logger.info("Trial %d started - writing logs to %s", trial_idx, log_file_path)
    start = time.time()
    try:
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        with log_file_path.open("w") as log_file:
            subprocess.run(
                train_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
                env=env,
                timeout=1800,
            )
        status = "success"
    except subprocess.CalledProcessError as exc:
        status = "error"
        logger.error("Trial %d training failed with exit code %d", trial_idx, exc.returncode)
    except KeyboardInterrupt:
        status = "error"
        logger.warning("Trial %d interrupted by user", trial_idx)
    except Exception as exc:  # pragma: no cover - depends on runtime
        status = "error"
        logger.error("Trial %d training error: %s", trial_idx, exc)

    elapsed = time.time() - start
    results: dict[str, Any] = {
        "trial": trial_idx,
        "dataset": dataset_name,
        "stamp": stamp,
        "config_path": str(config_path),
        "train_epochs": exp.training.epochs,
        "elapsed_seconds": elapsed,
        "status": status,
    }

    metrics_file = (
        Path("outputs")
        / dataset_name
        / f"{stamp}_trial_{trial_idx:03d}"
        / "checkpoints"
        / "metrics.json"
    )
    if status == "success" and metrics_file.exists():
        try:
            metrics_data = json.loads(metrics_file.read_text())
            final_metrics = metrics_data.get("final_metrics", {})
            if getattr(exp.data, "task", "classification") == "segmentation":
                results["val_dice"] = float(final_metrics.get("val_dice", 0.0))
                results["val_voxel_accuracy"] = float(final_metrics.get("val_voxel_accuracy", 0.0))
            else:
                results["val_accuracy"] = float(final_metrics.get("accuracy", 0.0))
                results["val_top5_accuracy"] = float(final_metrics.get("top5_accuracy", 0.0))
        except Exception as exc:  # pragma: no cover - file contents depend on runtime
            logger.error("Failed to parse metrics for trial %d: %s", trial_idx, exc)
            results["status"] = "error"
            results["message"] = f"Missing or invalid metrics: {exc}"
    elif status == "success":
        results["status"] = "error"
        results["message"] = f"Metrics file not found at {metrics_file}"
        logger.error(results["message"])

    (trial_dir / "results.json").write_text(json.dumps(results, indent=2))
    if status == "error" and "message" not in results:
        results["message"] = "Trial failed during execution"
    return results


def run_sweep_command(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    sweep = load_sweep(args.sweep)
    trials = args.trials or sweep.get("trials", 20)
    budget = sweep.get("budget_epochs", 25)
    outdir = Path(args.outdir or sweep.get("output_dir", "outputs/sweeps/unnamed"))
    outdir.mkdir(parents=True, exist_ok=True)

    seed = args.seed or sweep.get("seed", 42)
    random.seed(seed)

    base_cfg_path = args.base_config or sweep.get("base_config")
    space = sweep.get("param_space", {})
    summary: list[dict[str, Any]] = []

    logger.info("Starting sweep: %d trials, budget %d epochs, outdir=%s", trials, budget, outdir)
    if args.dry_run:
        if base_cfg_path is not None:
            base_raw = json.loads(Path(base_cfg_path).read_text())
        else:
            base_raw = {
                "model": {"num_dims": 2, "embed_dim": 96, "drop_path_rate": 0.1},
                "training": {"epochs": 100, "batch_size": 128, "learning_rate": 1e-3},
                "data": {"dataset": "cifar10", "data_dir": "data", "image_size": [32, 32]},
            }

        for idx in range(trials):
            sampled = {key: sample_value(spec) for key, spec in space.items()}
            conf = deepcopy(base_raw)
            for path, value in sampled.items():
                set_by_path(conf, path, value)
            conf.setdefault("training", {})["epochs"] = int(budget)
            dataset_name = (
                conf.get("data", {}).get("hf_id")
                or conf.get("data", {}).get("dataset")
                or "dataset"
            )
            dataset_name = str(dataset_name).replace("/", "_").replace(":", "_")
            stamp = conf.get("name", "dryrun") + "_dry"
            base_dir = outdir / dataset_name / stamp
            if base_dir.exists():
                shutil.rmtree(base_dir)
            base_dir.mkdir(parents=True, exist_ok=True)
            trial_dir = base_dir / f"trial_{idx:03d}"
            trial_dir.mkdir(parents=True, exist_ok=False)
            config_path = trial_dir / "config.json"
            config_path.write_text(json.dumps(conf, indent=2))
            result = {
                "trial": idx,
                "status": "dry",
                "config_path": str(config_path),
                "dataset": dataset_name,
                "stamp": stamp,
            }
            summary.append(result)
            write_sweep_summary(
                base_dir / "summary.json",
                summary,
                sweep=sweep,
                trials=trials,
                budget_epochs=budget,
                outdir=outdir,
                dry_run=True,
            )
            write_sweep_summary(
                outdir / "summary.json",
                summary,
                sweep=sweep,
                trials=trials,
                budget_epochs=budget,
                outdir=outdir,
                dry_run=True,
            )
        logger.info("Dry-run complete. Summary written to %s", base_dir / "summary.json")
        return 0

    base_exp = load_base_experiment(base_cfg_path)
    try:
        validate_experiment_dataset_contract(base_exp, require_validation_split=True)
    except Exception as exc:
        logger.error("Dataset/config validation failed before sweep start: %s", exc)
        return 1
    for idx in range(trials):
        max_attempts = 10
        exp = None
        trial_sampled: dict[str, Any]
        for attempt in range(max_attempts):
            trial_sampled = {key: sample_value(spec) for key, spec in space.items()}
            if "model.embed_dim" in trial_sampled and isinstance(
                trial_sampled["model.embed_dim"], str
            ):
                with suppress(ValueError):
                    trial_sampled["model.embed_dim"] = int(float(trial_sampled["model.embed_dim"]))
            try:
                exp = materialize_experiment(base_exp, trial_sampled, budget)
                _ = exp.model
                break
            except Exception as exc:
                logger.debug(
                    "Sampled invalid config on attempt %d/%d: %s; resampling...",
                    attempt + 1,
                    max_attempts,
                    exc,
                )
        if exp is None:
            err_msg = (
                f"Skipping trial {idx}: could not find valid config after {max_attempts} attempts"
            )
            logger.warning("%s", err_msg)
            summary.append({"trial": idx, "status": "error", "error": err_msg})
            write_sweep_summary(
                outdir / "summary.json",
                summary,
                sweep=sweep,
                trials=trials,
                budget_epochs=budget,
                outdir=outdir,
                dry_run=False,
            )
            continue

        try:
            result = run_trial(idx, exp, outdir, dry_run=False, sweep_config=sweep)
            summary.append(result)
            summary_path = (
                outdir
                / result.get("dataset", "dataset")
                / result.get("stamp", "stamp")
                / "summary.json"
            )
            write_sweep_summary(
                summary_path,
                summary,
                sweep=sweep,
                trials=trials,
                budget_epochs=budget,
                outdir=outdir,
                dry_run=False,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime
            logger.error("Trial %d failed with exception: %s", idx, exc)
            trial_dir = outdir / f"trial_{idx:03d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            error_path = trial_dir / "error.txt"
            error_path.write_text(str(exc))
            summary.append(
                {
                    "trial": idx,
                    "status": "error",
                    "error": str(exc),
                    "traceback_file": str(error_path),
                }
            )

        last = summary[-1]
        summary_path = (
            outdir / last.get("dataset", "_global") / last.get("stamp", "_global") / "summary.json"
        )
        write_sweep_summary(
            summary_path,
            summary,
            sweep=sweep,
            trials=trials,
            budget_epochs=budget,
            outdir=outdir,
            dry_run=False,
        )
        write_sweep_summary(
            outdir / "summary.json",
            summary,
            sweep=sweep,
            trials=trials,
            budget_epochs=budget,
            outdir=outdir,
            dry_run=False,
        )

        try:
            import jax

            jax.clear_caches()
        except Exception:
            pass
        gc.collect()

    logger.info("Sweep complete. Summary written to %s", outdir / "summary.json")
    return 0


def metric_direction(metric_name: str) -> str:
    metric_lower = metric_name.lower()
    if metric_lower.endswith("loss") or metric_lower == "loss":
        return "min"
    return "max"


def get_best_trial_from_summary(
    summary_path: Path,
    *,
    metric: str | None = None,
) -> dict[str, Any]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Sweep summary not found at {summary_path}")
    raw_summary = json.loads(summary_path.read_text())
    if isinstance(raw_summary, dict):
        summary = cast(list[dict[str, Any]], raw_summary.get("results", []))
        metric = metric or cast(str | None, raw_summary.get("metric"))
    else:
        summary = cast(list[dict[str, Any]], raw_summary)
    if not summary:
        raise ValueError(f"Sweep summary is empty: {summary_path}")
    valid_trials = [trial for trial in summary if trial.get("status") not in {"error", "dry"}]
    if not valid_trials:
        raise ValueError(f"No successful trials found in sweep summary: {summary_path}")

    if metric is not None:
        selected_metric = metric
        metric_trials = [trial for trial in valid_trials if selected_metric in trial]
        if not metric_trials:
            raise ValueError(
                f"Metric '{selected_metric}' not present in sweep summary: {summary_path}"
            )
        if metric_direction(selected_metric) == "min":
            best_trial = min(
                metric_trials,
                key=lambda trial: trial.get(selected_metric, float("inf")),
            )
        else:
            best_trial = max(
                metric_trials,
                key=lambda trial: trial.get(selected_metric, float("-inf")),
            )
        metric = selected_metric
    elif "val_dice" in valid_trials[0]:
        metric_name = "val_dice"
        best_trial = max(valid_trials, key=lambda trial: trial.get(metric_name, 0.0))
        metric = metric_name
    elif "val_accuracy" in valid_trials[0]:
        metric_name = "val_accuracy"
        best_trial = max(valid_trials, key=lambda trial: trial.get(metric_name, 0.0))
        metric = metric_name
    else:
        metric_name = "loss"
        best_trial = min(valid_trials, key=lambda trial: trial.get(metric_name, float("inf")))
        metric = metric_name
    logger.info(
        "Best trial found: Trial %s with %s = %.4f",
        best_trial.get("trial"),
        metric,
        best_trial.get(metric, 0.0),
    )
    return best_trial


def resolve_sweep_output_dir(sweep_path: str, explicit_outdir: str | None) -> Path:
    if explicit_outdir:
        return Path(explicit_outdir)
    try:
        sweep_data = load_sweep(sweep_path)
    except Exception:
        return Path("outputs/sweeps/unnamed")
    return Path(sweep_data.get("output_dir", "outputs/sweeps/unnamed"))


def run_auto_sweep_command(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    sweep_config = load_sweep(args.sweep)
    selection_metric = cast(str | None, sweep_config.get("metric"))
    planned_trials = args.trials or sweep_config.get("trials", 20)
    logger.info("=" * 80)
    logger.info("AUTOMATED SWEEP AND TRAIN WORKFLOW")
    logger.info("=" * 80)
    logger.info("1. Sweeping config: %s", args.sweep)
    logger.info("2. Number of trials: %d", planned_trials)
    logger.info("3. Final training epochs: %d", args.train_epochs)
    logger.info("-" * 80)

    sweep_cmd = build_cli_invocation("sweep", "--sweep", args.sweep)
    if args.base_config:
        sweep_cmd.extend(["--base-config", args.base_config])
    if args.trials is not None:
        sweep_cmd.extend(["--trials", str(args.trials)])
    if args.outdir:
        sweep_cmd.extend(["--outdir", args.outdir])
    if args.seed is not None:
        sweep_cmd.extend(["--seed", str(args.seed)])
    if args.log_level:
        sweep_cmd.extend(["--log-level", args.log_level])

    logger.info(">>> Launching Sweep...")
    try:
        subprocess.run(sweep_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Sweep failed with exit code %d. Aborting.", exc.returncode)
        return 1

    summary_path = resolve_sweep_output_dir(args.sweep, args.outdir) / "summary.json"
    if not summary_path.exists():
        logger.error(
            "Could not find sweep summary at %s. Cannot proceed to training.", summary_path
        )
        return 1

    try:
        best_trial = get_best_trial_from_summary(summary_path, metric=selection_metric)
    except Exception as exc:
        logger.error("Error parsing sweep summary: %s", exc)
        return 1

    best_config_path = best_trial.get("config_path")
    if not best_config_path or not Path(best_config_path).exists():
        logger.error("Best trial config not found at expected path: %s", best_config_path)
        return 1

    final_config_dir = Path("configs/auto_best")
    final_config_dir.mkdir(parents=True, exist_ok=True)
    trial_index = int(best_trial.get("trial", 0))
    dataset_name = best_trial.get("dataset", "unknown")
    stamp = best_trial.get("stamp", "unstamped")
    final_config_path = (
        final_config_dir / f"best_{dataset_name}_{stamp}_trial{trial_index:03d}.json"
    )
    shutil.copy(best_config_path, final_config_path)
    logger.info("Copied optimal configuration to: %s", final_config_path)

    train_cmd = build_cli_invocation(
        "train",
        "--config",
        str(final_config_path),
        "--epochs",
        str(args.train_epochs),
        "--log-level",
        args.log_level,
    )
    logger.info(">>> Launching Final Full Training...")
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Final training failed with exit code %d.", exc.returncode)
        return 1

    logger.info("=" * 80)
    logger.info("FULL WORKFLOW COMPLETED SUCCESSFULLY")
    logger.info("Best Configuration: %s", final_config_path)
    logger.info("=" * 80)
    return 0


def load_queue(path: str) -> list[dict[str, Any]]:
    queue_path = Path(path)
    if not queue_path.exists():
        raise FileNotFoundError(f"Queue file not found: {path}")
    text = queue_path.read_text()
    if queue_path.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML required for YAML queue files. Install pyyaml.")
        data = yaml.safe_load(text)
    elif queue_path.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported queue file format: {queue_path.suffix}")
    jobs = data if isinstance(data, list) else data.get("jobs", [])
    if not jobs:
        raise ValueError(f"No jobs found in queue file: {path}")
    return jobs


def build_command(job: dict[str, Any], python_exe: str) -> list[str]:
    job_type = job.get("type", "auto-sweep")
    extra_args = job.get("args", [])
    if isinstance(extra_args, str):
        extra_args = extra_args.split()

    if job_type == "auto-sweep":
        sweep_file = job.get("sweep")
        if not sweep_file:
            raise ValueError(f"auto-sweep job requires 'sweep' field: {job}")
        cmd = build_cli_invocation("auto-sweep", "--sweep", str(sweep_file), python_exe=python_exe)
        if "trials" in job:
            cmd.extend(["--trials", str(job["trials"])])
        if "train_epochs" in job:
            cmd.extend(["--train-epochs", str(job["train_epochs"])])
        if "outdir" in job:
            cmd.extend(["--outdir", str(job["outdir"])])
        cmd.extend(extra_args)
        return cmd

    if job_type == "sweep":
        sweep_file = job.get("sweep")
        if not sweep_file:
            raise ValueError(f"sweep job requires 'sweep' field: {job}")
        cmd = build_cli_invocation("sweep", "--sweep", str(sweep_file), python_exe=python_exe)
        if "trials" in job:
            cmd.extend(["--trials", str(job["trials"])])
        if "outdir" in job:
            cmd.extend(["--outdir", str(job["outdir"])])
        cmd.extend(extra_args)
        return cmd

    if job_type == "train":
        config_file = job.get("config")
        if not config_file:
            raise ValueError(f"train job requires 'config' field: {job}")
        cmd = build_cli_invocation("train", "--config", str(config_file), python_exe=python_exe)
        if "epochs" in job:
            cmd.extend(["--epochs", str(job["epochs"])])
        cmd.extend(extra_args)
        return cmd

    raise ValueError(f"Unknown job type: {job_type}")


def load_completed_jobs(results_dir: Path) -> set[str]:
    completed: set[str] = set()
    for result_file in sorted(results_dir.glob("queue_*.json")):
        try:
            data = json.loads(result_file.read_text())
        except Exception:
            continue
        for entry in data:
            if entry.get("status") == "completed":
                completed.add(entry.get("name", ""))
    return completed


def run_job(
    job_idx: int, job: dict[str, Any], python_exe: str, dry_run: bool = False, retry: int = 0
) -> dict[str, Any]:
    job_name = job.get("name", f"job_{job_idx:03d}")
    job_type = job.get("type", "auto-sweep")
    max_retries = job.get("retries", retry)

    logger.info("=" * 80)
    logger.info("JOB %d: %s (type=%s)", job_idx, job_name, job_type)
    logger.info("=" * 80)
    try:
        cmd = build_command(job, python_exe)
    except ValueError as exc:
        logger.error("  %s", exc)
        return {"job": job_idx, "name": job_name, "status": "error", "error": str(exc)}

    logger.info("  Command: %s", " ".join(cmd))
    if dry_run:
        logger.info("  [DRY RUN] Skipping execution")
        return {"job": job_idx, "name": job_name, "status": "dry", "command": " ".join(cmd)}

    attempt = 0
    last_error = None
    while attempt <= max_retries:
        if attempt > 0:
            logger.info("  Retry %d/%d...", attempt, max_retries)
            time.sleep(5)
        start = time.time()
        try:
            env = os.environ.copy()
            env["HF_HOME"] = ".hf_cache"
            env["HF_DATASETS_CACHE"] = ".hf_cache"
            subprocess.run(cmd, env=env, check=True, stdout=sys.stdout, stderr=sys.stderr)
            elapsed = time.time() - start
            logger.info("Job %s COMPLETED in %.1fs", job_name, elapsed)
            return {
                "job": job_idx,
                "name": job_name,
                "status": "completed",
                "elapsed_seconds": elapsed,
            }
        except subprocess.CalledProcessError as exc:
            elapsed = time.time() - start
            last_error = str(exc)
            logger.error(
                "Job %s FAILED (attempt %d/%d, %.1fs): %s",
                job_name,
                attempt + 1,
                max_retries + 1,
                elapsed,
                exc,
            )
            attempt += 1
        except KeyboardInterrupt:
            logger.warning("Job %s INTERRUPTED by user", job_name)
            return {"job": job_idx, "name": job_name, "status": "interrupted"}

    return {
        "job": job_idx,
        "name": job_name,
        "status": "failed",
        "error": last_error,
        "attempts": attempt,
    }


def run_queue_command(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    jobs = load_queue(args.queue)
    python_exe = sys.executable

    if args.jobs:
        job_filter = {name.strip() for name in args.jobs.split(",")}
        original_count = len(jobs)
        jobs = [job for job in jobs if job.get("name", "") in job_filter]
        logger.info(
            "Filtered to %d/%d jobs matching: %s",
            len(jobs),
            original_count,
            ", ".join(sorted(job_filter)),
        )

    if args.skip_completed:
        completed = load_completed_jobs(Path("logs"))
        skipped_names = {job.get("name", "") for job in jobs} & completed
        jobs = [job for job in jobs if job.get("name", "") not in completed]
        if skipped_names:
            logger.info(
                "Skipping %d already-completed jobs: %s",
                len(skipped_names),
                ", ".join(sorted(skipped_names)),
            )

    logger.info("=" * 80)
    logger.info("QUEUE RUNNER")
    logger.info("=" * 80)
    logger.info("Queue file: %s", args.queue)
    logger.info("Total jobs: %d", len(jobs))
    logger.info("Continue on error: %s", not args.stop_on_error)
    logger.info("Started: %s", datetime.now().isoformat())
    logger.info("-" * 80)

    results: list[dict[str, Any]] = []
    results_path = Path("logs") / f"queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, job in enumerate(jobs):
        result = run_job(idx, job, python_exe, dry_run=args.dry_run, retry=args.retry)
        results.append(result)
        results_path.write_text(json.dumps(results, indent=2))
        if result["status"] in {"failed", "error"} and args.stop_on_error:
            logger.error("Stopping queue due to job failure (--stop-on-error)")
            break

    completed_count = sum(1 for result in results if result["status"] == "completed")
    failed = sum(1 for result in results if result["status"] in {"failed", "error"})
    dry = sum(1 for result in results if result["status"] == "dry")
    logger.info("=" * 80)
    logger.info("QUEUE COMPLETE")
    logger.info("=" * 80)
    logger.info("Completed: %d/%d", completed_count, len(jobs))
    if failed:
        logger.warning("Failed: %d/%d", failed, len(jobs))
    if dry:
        logger.info("Dry-run: %d/%d", dry, len(jobs))
    logger.info("Results saved to: %s", results_path)
    return 0 if failed == 0 or not args.stop_on_error else 1


def canonicalize_label_token(label: Any) -> str:
    """Convert arbitrary labels into a deterministic string token."""
    if hasattr(label, "item"):
        with suppress(Exception):
            label = label.item()

    if isinstance(label, bool):
        return json.dumps(label)
    if isinstance(label, int):
        return str(label)
    if isinstance(label, float) and label.is_integer():
        return str(int(label))
    if isinstance(label, str):
        return label
    return json.dumps(label, sort_keys=True, ensure_ascii=True)


def is_scalar_classification_label(label: Any) -> bool:
    """Return True when a label represents a single classification target."""
    if hasattr(label, "item"):
        with suppress(Exception):
            label = label.item()

    return isinstance(label, (bool, int, float, str))


def class_dirname_for_label(label: Any) -> tuple[str, str]:
    """Return a deterministic class directory name and canonical label token."""
    token = canonicalize_label_token(label)
    if re.fullmatch(r"\d+", token):
        return f"class_{int(token):03d}", token

    safe_label = re.sub(r"[^A-Za-z0-9]+", "_", token).strip("_").lower() or "label"
    safe_label = safe_label[:32]
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:8]
    return f"class_{safe_label}_{digest}", token


def point_cloud_to_voxel(points: Any, resolution: int = 32) -> Any:
    import numpy as np

    p_min = points.min(axis=0)
    p_max = points.max(axis=0)
    points = (points - p_min) / (p_max - p_min + 1e-8)
    coords = np.clip(np.floor(points * resolution), 0, resolution - 1).astype(np.int32)
    voxel = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    voxel[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    return voxel


def save_classification(
    out_dir: Path,
    split: str,
    idx: int,
    image: Any,
    label: Any,
) -> tuple[str, str, bool]:
    import numpy as np

    class_dirname, label_token = class_dirname_for_label(label)
    class_dir = out_dir / split / class_dirname
    class_dir.mkdir(parents=True, exist_ok=True)
    image_array = np.array(image)
    voxelized = False
    if image_array.ndim == 2 and image_array.shape[1] == 3:
        image_array = point_cloud_to_voxel(image_array, resolution=32)
        voxelized = True
    np.savez_compressed(class_dir / f"{idx:05d}.npz", image=image_array)
    return class_dirname, label_token, voxelized


def save_segmentation(out_dir: Path, split: str, case_id: str, image: Any, label: Any) -> None:
    import numpy as np

    image_dir = out_dir / split / "images"
    label_dir = out_dir / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(image_dir / f"{case_id}.npz", image=image.astype(np.float32))
    np.savez_compressed(label_dir / f"{case_id}.npz", label=label.astype(np.int32))


def build_export_manifest(
    *,
    hf_id: str,
    outdir: str,
    task: str,
    split_counts: dict[str, int],
    label_tokens_by_dir: dict[str, str],
    point_cloud_voxelized: bool,
) -> dict[str, Any]:
    """Build a deterministic manifest for exported datasets."""
    class_dirs = sorted(label_tokens_by_dir)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "hf_id": hf_id,
        "data_dir": outdir,
        "task": task,
        "split_counts": split_counts,
        "sample_format": {
            "storage": "npz",
            "image_key": "image",
            "point_cloud_voxelized": point_cloud_voxelized,
        },
    }

    if task == "classification":
        manifest["class_to_idx"] = {name: idx for idx, name in enumerate(class_dirs)}
        manifest["class_labels"] = {name: label_tokens_by_dir[name] for name in class_dirs}
        manifest["num_classes"] = len(class_dirs)
    else:
        manifest["sample_format"]["label_key"] = "label"

    return manifest


def export_dataset(hf_id: str, outdir: str, limit: int | None = None) -> None:
    import datasets
    import numpy as np

    from ndswin.training.data import write_dataset_manifest

    out_dir = Path(outdir)
    split_counts: dict[str, int] = {}
    label_tokens_by_dir: dict[str, str] = {}
    exported_task: str | None = None
    point_cloud_voxelized = False
    print(f"Loading dataset {hf_id}...")
    try:
        ds = datasets.load_dataset(hf_id)
        print("Available splits:", list(ds.keys()))
        for split in ds.keys():
            print(f"Processing split: {split}")
            count = 0
            for example in ds[split]:
                image_data = None
                for key in ("image", "images", "img", "image0", "pixel_values", "inputs"):
                    if key in example and example[key] is not None:
                        image_data = example[key]
                        break
                if "label" in example and example["label"] is not None and image_data is not None:
                    image = image_data
                    label = example["label"]
                    if is_scalar_classification_label(label):
                        class_dirname, label_token, voxelized = save_classification(
                            out_dir, split, count, image, label
                        )
                        label_tokens_by_dir[class_dirname] = label_token
                        point_cloud_voxelized = point_cloud_voxelized or voxelized
                        current_task = "classification"
                    else:
                        img = np.array(image)
                        lbl = np.array(label)
                        if img.ndim == 3:
                            img = np.expand_dims(img, 0)
                        elif img.ndim == 4 and img.shape[-1] < 5:
                            img = np.moveaxis(img, -1, 0)
                        if lbl.ndim == 4 and lbl.shape[0] == 1:
                            lbl = lbl[0]
                        case_id = example.get("image_id") or example.get("id") or f"{split}_{count}"
                        save_segmentation(out_dir, split, str(case_id), img, lbl)
                        current_task = "segmentation"
                elif (
                    "images" in example
                    and "labels" in example
                    and example["images"]
                    and example["labels"]
                ):
                    img = np.array(example["images"][0])
                    lbl = np.array(example["labels"][0])
                    case_id = example.get("id") or f"{split}_{count}"
                    save_segmentation(out_dir, split, str(case_id), img, lbl)
                    current_task = "segmentation"
                else:
                    print(
                        f"Warning: skipping example {count} in split {split} (unrecognized format)"
                    )
                    continue
                if exported_task is None:
                    exported_task = current_task
                elif exported_task != current_task:
                    raise RuntimeError(
                        f"Mixed dataset task types detected during export: {exported_task} and {current_task}"
                    )
                count += 1
                if limit is not None and count >= limit:
                    break
            split_counts[split] = count
            print(f"Saved {count} examples under {out_dir / split}")
    except ValueError as exc:
        print(f"load_dataset failed with: {exc}; retrying using streaming fallback...")
        any_saved = False
        for split in ("train", "validation", "val", "test"):
            try:
                print(f"Attempting streaming export for split: {split}")
                iterator = datasets.load_dataset(hf_id, split=split, streaming=True)
                count = 0
                for example in iterator:
                    image_data = None
                    for key in ("image", "images", "img", "image0", "pixel_values", "inputs"):
                        if key in example and example[key] is not None:
                            image_data = example[key]
                            break
                    if (
                        "label" in example
                        and example["label"] is not None
                        and image_data is not None
                    ):
                        image = image_data
                        label = example["label"]
                        if is_scalar_classification_label(label):
                            class_dirname, label_token, voxelized = save_classification(
                                out_dir, split, count, image, label
                            )
                            label_tokens_by_dir[class_dirname] = label_token
                            point_cloud_voxelized = point_cloud_voxelized or voxelized
                            current_task = "classification"
                        else:
                            img = np.array(image)
                            lbl = np.array(label)
                            if img.ndim == 3:
                                img = np.expand_dims(img, 0)
                            elif img.ndim == 4 and img.shape[-1] < 5:
                                img = np.moveaxis(img, -1, 0)
                            if lbl.ndim == 4 and lbl.shape[0] == 1:
                                lbl = lbl[0]
                            case_id = (
                                example.get("image_id") or example.get("id") or f"{split}_{count}"
                            )
                            save_segmentation(out_dir, split, str(case_id), img, lbl)
                            current_task = "segmentation"
                    elif (
                        "images" in example
                        and "labels" in example
                        and example["images"]
                        and example["labels"]
                    ):
                        img = np.array(example["images"][0])
                        lbl = np.array(example["labels"][0])
                        case_id = example.get("id") or f"{split}_{count}"
                        save_segmentation(out_dir, split, str(case_id), img, lbl)
                        current_task = "segmentation"
                    else:
                        print(
                            f"Warning: skipping example {count} in split {split} (unrecognized format)"
                        )
                        continue
                    if exported_task is None:
                        exported_task = current_task
                    elif exported_task != current_task:
                        raise RuntimeError(
                            f"Mixed dataset task types detected during export: {exported_task} and {current_task}"
                        )
                    count += 1
                    if limit is not None and count >= limit:
                        break
                if count > 0:
                    print(f"Saved {count} examples under {out_dir / split}")
                    split_counts[split] = count
                    any_saved = True
            except Exception:
                continue
        if not any_saved:
            raise
    if exported_task is not None:
        manifest = build_export_manifest(
            hf_id=hf_id,
            outdir=outdir,
            task=exported_task,
            split_counts=split_counts,
            label_tokens_by_dir=label_tokens_by_dir,
            point_cloud_voxelized=point_cloud_voxelized,
        )
        manifest_path = write_dataset_manifest(outdir, manifest)
        print(f"Wrote dataset manifest: {manifest_path}")
    print("Done.")


def run_fetch_data_command(args: argparse.Namespace) -> int:
    export_dataset(args.hf_id, args.outdir, limit=args.limit)
    return 0


def find_summary_files(outputs_dir: Path) -> list[Path]:
    if not outputs_dir.exists():
        return []
    sweeps_dir = outputs_dir / "sweeps" if (outputs_dir / "sweeps").exists() else outputs_dir
    summary_files = [
        path for path in sweeps_dir.rglob("summary.json") if path.parent.parent == sweeps_dir
    ]
    return sorted(summary_files, key=lambda path: path.stat().st_mtime, reverse=True)


def infer_summary_metric(summary_path: Path) -> str | None:
    sweep_root = next(
        (parent for parent in summary_path.parents if parent.parent.name == "sweeps"),
        None,
    )
    if sweep_root is None:
        return None

    for suffix in (".yaml", ".yml", ".json"):
        sweep_path = Path("configs") / "sweeps" / f"{sweep_root.name}{suffix}"
        if sweep_path.exists():
            try:
                return cast(str | None, load_sweep(str(sweep_path)).get("metric"))
            except Exception:
                return None
    return None


def format_best_trial(best_trial: dict[str, Any], metric: str | None = None) -> list[str]:
    lines = [f"Best Trial: {best_trial.get('trial', 'N/A')}"]
    if metric is not None and metric in best_trial:
        lines.append(f"Selection Metric: {metric}")
        lines.append(f"Metric Value: {best_trial.get(metric, 0.0):.4f}")
    if "val_accuracy" in best_trial:
        lines.append(f"Val Accuracy: {best_trial.get('val_accuracy', 0.0):.4f}")
        lines.append(f"Val Top-5: {best_trial.get('val_top5_accuracy', 0.0):.4f}")
    elif "val_dice" in best_trial:
        lines.append(f"Val Dice: {best_trial.get('val_dice', 0.0):.4f}")
        lines.append(f"Val Voxel Acc: {best_trial.get('val_voxel_accuracy', 0.0):.4f}")
    elif "loss" in best_trial:
        lines.append(f"Loss: {best_trial.get('loss', 0.0):.4f}")
    lines.append(f"Config: {best_trial.get('config_path', 'N/A')}")
    lines.append(f"Elapsed: {best_trial.get('elapsed_seconds', 0):.0f}s")
    return lines


def run_show_best_command(args: argparse.Namespace) -> int:
    summaries = [Path(args.summary)] if args.summary else find_summary_files(Path(args.outputs_dir))
    print("Best results from sweep outputs:")
    print()
    if not summaries:
        print("  No sweep results found in outputs/. Run 'make optimize' first.")
        return 0
    printed_any = False
    for summary in summaries:
        try:
            selection_metric = infer_summary_metric(summary)
            best_trial = get_best_trial_from_summary(summary, metric=selection_metric)
        except Exception:
            continue
        if printed_any:
            print()
        print(f"  Source: {summary}")
        for line in format_best_trial(best_trial, selection_metric):
            print(f"  {line}")
        printed_any = True
    if not printed_any:
        print("  No successful trials found.")
    return 0


def run_validate_command(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    logger.info("Running pipeline validation...")
    logger.info("")

    if not args.skip_tests:
        logger.info("1. Running unit tests...")
        test_cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-v",
            "-x",
            "--tb=short",
            "-q",
            *args.pytest_args,
        ]
        result = subprocess.run(test_cmd)
        if result.returncode != 0:
            return result.returncode
        logger.info("")

    if not args.skip_train:
        logger.info("2. Smoke test: %d-epoch training...", args.train_epochs)
        train_args = argparse.Namespace(
            config=args.config,
            epochs=args.train_epochs,
            batch_size=None,
            lr=None,
            seed=None,
            data_dir=None,
            max_steps_per_epoch=args.max_steps_per_epoch,
            max_devices=args.max_devices,
            stamp=args.stamp,
            no_log_file=args.no_log_file,
            log_level=args.log_level,
            overrides=list(args.overrides),
        )
        previous_validation_fallback = os.environ.get("ALLOW_VALIDATION_SPLIT_FALLBACK")
        logger.warning(
            "Validation smoke test enables ALLOW_VALIDATION_SPLIT_FALLBACK=1 so configs "
            "without a dedicated validation split can still complete the short smoke run."
        )
        os.environ["ALLOW_VALIDATION_SPLIT_FALLBACK"] = "1"
        try:
            return_code = run_train_command(train_args)
        finally:
            if previous_validation_fallback is None:
                os.environ.pop("ALLOW_VALIDATION_SPLIT_FALLBACK", None)
            else:
                os.environ["ALLOW_VALIDATION_SPLIT_FALLBACK"] = previous_validation_fallback
        if return_code != 0:
            return return_code
        logger.info("")

    logger.info("✓ Pipeline validation passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ndswin",
        description="NDSwin-JAX package CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model from a JSON config")
    add_config_argument(train_parser)
    add_train_override_arguments(train_parser)
    add_overrides_argument(train_parser)
    add_log_level_argument(train_parser)
    train_parser.set_defaults(func=run_train_command)

    sweep_parser = subparsers.add_parser("sweep", help="Run a random-search hyperparameter sweep")
    add_sweep_argument(sweep_parser)
    sweep_parser.add_argument(
        "--base-config", type=str, default=None, help="Optional base experiment JSON file"
    )
    sweep_parser.add_argument(
        "--trials", type=int, default=None, help="Override number of trials in sweep file"
    )
    sweep_parser.add_argument(
        "--dry-run", action="store_true", help="Materialize configs without training"
    )
    sweep_parser.add_argument(
        "--outdir", type=str, default=None, help="Where to write sweep outputs"
    )
    sweep_parser.add_argument("--seed", type=int, default=None, help="Sweep random seed")
    add_log_level_argument(sweep_parser)
    sweep_parser.set_defaults(func=run_sweep_command)

    auto_parser = subparsers.add_parser("auto-sweep", help="Run a sweep and train the best config")
    add_sweep_argument(auto_parser)
    auto_parser.add_argument(
        "--base-config", type=str, default=None, help="Optional base experiment JSON file"
    )
    auto_parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override the number of sweep trials from the sweep file",
    )
    auto_parser.add_argument(
        "--train-epochs", type=int, default=100, help="Epochs for final training"
    )
    auto_parser.add_argument(
        "--outdir", type=str, default=None, help="Output directory for sweep artifacts"
    )
    auto_parser.add_argument("--seed", type=int, default=None, help="Sweep random seed")
    add_log_level_argument(auto_parser)
    auto_parser.set_defaults(func=run_auto_sweep_command)

    queue_parser = subparsers.add_parser("queue", help="Run a queue of train/sweep/auto-sweep jobs")
    add_queue_argument(queue_parser)
    queue_parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    queue_parser.add_argument("--retry", type=int, default=0, help="Default retries per job")
    queue_parser.add_argument(
        "--jobs", type=str, default=None, help="Comma-separated list of job names to run"
    )
    queue_parser.add_argument(
        "--skip-completed", action="store_true", help="Skip jobs completed in previous queue logs"
    )
    queue_parser.add_argument(
        "--stop-on-error", action="store_true", help="Stop queue if any job fails"
    )
    add_log_level_argument(queue_parser)
    queue_parser.set_defaults(func=run_queue_command)

    fetch_parser = subparsers.add_parser(
        "fetch-data", help="Fetch and export a Hugging Face dataset"
    )
    fetch_parser.add_argument("--hf-id", type=str, required=True, help="Hugging Face dataset id")
    fetch_parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    fetch_parser.add_argument(
        "--limit", type=int, default=None, help="Optional per-split example cap"
    )
    fetch_parser.set_defaults(func=run_fetch_data_command)

    show_best_parser = subparsers.add_parser(
        "show-best", help="Show the best trial from sweep summaries"
    )
    show_best_parser.add_argument(
        "--summary", type=str, default=None, help="Specific summary.json to inspect"
    )
    show_best_parser.add_argument(
        "--outputs-dir", type=str, default="outputs", help="Directory to search for summaries"
    )
    show_best_parser.set_defaults(func=run_show_best_command)

    validate_parser = subparsers.add_parser("validate", help="Run tests and a smoke training job")
    add_config_argument(validate_parser, required=False)
    validate_parser.set_defaults(config="configs/cifar10.json")
    validate_parser.add_argument(
        "--train-epochs", type=int, default=2, help="Epochs for the smoke training run"
    )
    validate_parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Cap steps per epoch during validation training",
    )
    validate_parser.add_argument(
        "--max-devices",
        type=int,
        default=1,
        help="Limit validation smoke training to at most this many visible JAX devices",
    )
    validate_parser.add_argument("--skip-tests", action="store_true", help="Skip pytest execution")
    validate_parser.add_argument("--skip-train", action="store_true", help="Skip smoke training")
    validate_parser.add_argument(
        "--pytest-args", nargs="*", default=[], help="Additional pytest arguments"
    )
    validate_parser.add_argument(
        "--stamp", type=str, default=None, help="Optional smoke-run stamp override"
    )
    validate_parser.add_argument(
        "--no-log-file", action="store_true", help="Disable training log file during smoke run"
    )
    add_overrides_argument(validate_parser)
    add_log_level_argument(validate_parser)
    validate_parser.set_defaults(func=run_validate_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


def train_main(argv: Sequence[str] | None = None) -> int:
    return main(["train", *(list(argv) if argv is not None else sys.argv[1:])])


def sweep_main(argv: Sequence[str] | None = None) -> int:
    return main(["sweep", *(list(argv) if argv is not None else sys.argv[1:])])


def auto_sweep_main(argv: Sequence[str] | None = None) -> int:
    return main(["auto-sweep", *(list(argv) if argv is not None else sys.argv[1:])])


def queue_main(argv: Sequence[str] | None = None) -> int:
    return main(["queue", *(list(argv) if argv is not None else sys.argv[1:])])


def fetch_data_main(argv: Sequence[str] | None = None) -> int:
    return main(["fetch-data", *(list(argv) if argv is not None else sys.argv[1:])])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
