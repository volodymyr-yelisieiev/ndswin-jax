#!/usr/bin/env python3
"""Unified training script for NDSwin-JAX.

This script provides config-driven training for any dimensionality (2D, 3D, 4D+).

Usage:
    python train/train.py --config configs/cifar100.json
    python train/train.py --config configs/cifar100.json --epochs 200
    make train CONFIG=configs/cifar100.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ndswin.utils.gpu import setup_optimal_gpus
setup_optimal_gpus()

import jax
import jax.numpy as jnp
import numpy as np

from ndswin import NDSwinConfig, NDSwinTransformer, TrainingConfig, ExperimentConfig
from ndswin.training import (
    Trainer,
    create_data_loader,
)

def setup_output_dirs(
    exp_config: ExperimentConfig,
    config_path: str,
    stamp: str | None = None,
) -> tuple[Path, Path]:
    """Setup output directories for checkpoints and logs."""
    if stamp is None:
        stamp = exp_config.get_stamp()

    # Dataset name for directory organization
    dataset_name = getattr(exp_config.data, "hf_id", None) or getattr(exp_config.data, "dataset", None) or "dataset"
    dataset_name = str(dataset_name).replace("/", "_").replace(":", "_")

    # Create a base outputs directory per dataset
    checkpoint_dir = Path("outputs") / dataset_name / stamp / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Copy config to output dir for reference
    config_copy_path = Path("outputs") / dataset_name / stamp / "config.json"
    shutil.copy(config_path, config_copy_path)

    # Create logs directory per dataset
    logs_dir = Path("logs") / dataset_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{stamp}.log"

    return checkpoint_dir, log_file


class TeeLogger:
    """Logger that writes to both stdout and a file."""

    def __init__(self, log_file: Path) -> None:
        self.terminal = sys.stdout
        self._log_file = log_file
        self._file_handle: Any = None

    def _ensure_open(self) -> None:
        """Ensure log file is open."""
        if self._file_handle is None:
            self._file_handle = self._log_file.open("a")  # noqa: SIM115

    def write(self, message: str) -> None:
        self._ensure_open()
        self.terminal.write(message)
        self._file_handle.write(message)
        self._file_handle.flush()

    def flush(self) -> None:
        self.terminal.flush()
        if self._file_handle is not None:
            self._file_handle.flush()

    def close(self) -> None:
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train NDSwin model with config file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file",
    )

    # Optional overrides
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable logging to file",
    )
    parser.add_argument(
        "--stamp",
        type=str,
        default=None,
        help="Override experiment stamp (used for directories and log files)",
    )

    return parser.parse_args()


def apply_overrides(exp_config: ExperimentConfig, args: argparse.Namespace) -> None:
    """Apply command-line overrides to experiment configuration."""
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


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config_dict = json.load(f)
    
    # Map old config keys to new config keys for compatibility
    if "dataset" in config_dict and "data" not in config_dict:
        config_dict["data"] = config_dict.pop("dataset")
    if "augmentation" in config_dict:
        config_dict.setdefault("data", {}).update(config_dict.pop("augmentation"))

    exp_config = ExperimentConfig.from_dict(config_dict)

    # Apply command-line overrides
    apply_overrides(exp_config, args)

    # Setup output directories
    checkpoint_dir, log_file = setup_output_dirs(exp_config, str(config_path), stamp=args.stamp)

    # Setup logging
    logger = None
    if not args.no_log_file:
        logger = TeeLogger(log_file)
        sys.stdout = logger  # type: ignore

    try:
        # Print header
        print("=" * 70)
        print("NDSwin-JAX Training")
        print("=" * 70)
        print(f"\nConfig file: {config_path}")
        print(f"Experiment: {exp_config.name}")
        print(f"Description: {exp_config.description}")
        print(f"Config hash: {exp_config.get_config_hash()}")
        print(f"\nCheckpoint dir: {checkpoint_dir}")
        print(f"Log file: {log_file}")
        print(f"Timestamp: {datetime.now().isoformat()}")

        # Set random seed
        rng = jax.random.PRNGKey(exp_config.training.seed)
        np.random.seed(exp_config.training.seed)

        # Create configurations
        model_config = exp_config.model
        train_config = exp_config.training

        # Print configurations
        print("\n" + "-" * 70)
        print("Dataset Configuration:")
        print("-" * 70)
        for key, value in exp_config.data.to_dict().items():
            print(f"  {key}: {value}")

        print("\n" + "-" * 70)
        print("Model Configuration:")
        print("-" * 70)
        print(f"  num_dims: {model_config.num_dims}")
        print(f"  patch_size: {model_config.patch_size}")
        print(f"  window_size: {model_config.window_size}")
        print(f"  embed_dim: {model_config.embed_dim}")
        print(f"  depths: {model_config.depths}")
        print(f"  num_heads: {model_config.num_heads}")
        print(f"  drop_path_rate: {model_config.drop_path_rate}")
        print(f"  num_classes: {model_config.num_classes}")

        print("\n" + "-" * 70)
        print("Training Configuration:")
        print("-" * 70)
        print(f"  epochs: {train_config.num_epochs}")
        print(f"  batch_size: {train_config.batch_size}")
        print(f"  learning_rate: {train_config.learning_rate}")
        print(f"  weight_decay: {train_config.weight_decay}")
        print(f"  optimizer: {train_config.optimizer}")
        print(f"  lr_schedule: {train_config.lr_schedule}")
        print(f"  warmup_epochs: {train_config.warmup_epochs}")
        print(f"  label_smoothing: {train_config.label_smoothing}")

        

        # Create model
        task = getattr(exp_config.data, "task", "classification")
        if task == "segmentation":
            from ndswin.models.classifier import SwinForSegmentation
            model = SwinForSegmentation(
                config=model_config,
                num_classes=model_config.num_classes or 2,
            )
        else:
            model = NDSwinTransformer(config=model_config)

        # Count parameters
        input_shape = exp_config.data.image_size
        if len(input_shape) == exp_config.model.num_dims:
            input_shape = (exp_config.data.in_channels,) + input_shape
        dummy_input = jnp.ones((1,) + input_shape)
        variables = model.init(rng, dummy_input, deterministic=True)
        num_params = sum(p.size for p in jax.tree_util.tree_leaves(variables["params"]))
        print(f"\nTotal parameters: {num_params:,}")

        # Create augmentation transforms
        # Create data loaders
        print("\nLoading data...")
        try:
            patch_size = exp_config.model.patch_size
            num_stages = exp_config.model.num_stages
            pad_to = tuple(int(patch_size[i] * (2 ** (num_stages - 1))) for i in range(len(patch_size)))
        except Exception:
            pad_to = None

        dataset_name = getattr(exp_config.data, "dataset", "unknown")
        train_loader = create_data_loader(
            config=exp_config.data,
            split="train",
            batch_size=exp_config.training.batch_size,
            pad_to=pad_to
        )
        try:
            val_loader = create_data_loader(
                config=exp_config.data,
                split="validation",
                batch_size=exp_config.training.batch_size,
                pad_to=pad_to
            )
        except Exception:
            try:
                val_loader = create_data_loader(
                    config=exp_config.data,
                    split="test",
                    batch_size=exp_config.training.batch_size,
                    pad_to=pad_to
                )
            except Exception as e:
                print(f"Warning: Could not load validation or test split: {e}")
                val_loader = None

        
        if hasattr(train_loader, "dataset_info"):
            print(f"  Dataset: {train_loader.dataset_info.name}")
            print(f"  Train samples: {train_loader.dataset_info.num_train}")
            print(f"  Test samples: {train_loader.dataset_info.num_test}")
        else:
            print(f"  Dataset: {dataset_name}")

        # Create Mixup/CutMix transform if configured
        mixup_transform = None
        mixup_alpha = getattr(exp_config.data, "mixup_alpha", 0.0)
        cutmix_alpha = getattr(exp_config.data, "cutmix_alpha", 0.0)
        if mixup_alpha > 0 or cutmix_alpha > 0:
            from ndswin.training.augmentation import MixupOrCutmix
            num_classes = model_config.num_classes or 100
            mixup_prob = 0.5 if (mixup_alpha > 0 and cutmix_alpha > 0) else (1.0 if mixup_alpha > 0 else 0.0)
            mixup_transform = MixupOrCutmix(
                p=mixup_prob,
                mixup_alpha=mixup_alpha if mixup_alpha > 0 else 1.0,
                cutmix_alpha=cutmix_alpha if cutmix_alpha > 0 else 1.0,
                num_classes=num_classes,
            )
            print(f"\n  Mixup alpha: {mixup_alpha}")
            print(f"  CutMix alpha: {cutmix_alpha}")

        # Create trainer
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
        )

        # Train
        print("\n" + "=" * 70)
        print("Starting training...")
        print("=" * 70 + "\n")

        start_time = time.time()
        history = trainer.fit(train_loader, val_loader)
        total_time = time.time() - start_time

        # Final evaluation
        if val_loader is not None:
            val_loader.reset()
            final_metrics = trainer.evaluate(val_loader)
            print("\n" + "=" * 70)
            print("Final Validation Results:")
            print("=" * 70)
            print(f"  Loss: {final_metrics['loss']:.4f}")
            if task == "segmentation":
                print(f"  Dice: {final_metrics.get('val_dice', 0.0):.4f}")
                print(f"  Voxel Acc: {final_metrics.get('val_voxel_accuracy', 0.0):.4f}")
            else:
                print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
                print(f"  Top-5 Accuracy: {final_metrics['top5_accuracy']:.4f}")

        # Save final metrics
        metrics_file = checkpoint_dir / "metrics.json"
        final_metrics_data: dict[str, float] = {"loss": float(final_metrics["loss"]) if val_loader else None}
        if task == "segmentation":
            final_metrics_data["val_dice"] = float(final_metrics.get("val_dice", 0.0))
            final_metrics_data["val_voxel_accuracy"] = float(final_metrics.get("val_voxel_accuracy", 0.0))
        else:
            final_metrics_data["accuracy"] = float(final_metrics.get("accuracy", 0.0))
            final_metrics_data["top5_accuracy"] = float(final_metrics.get("top5_accuracy", 0.0))

        metrics_data = {
            "config": exp_config.to_dict(),
            "final_metrics": final_metrics_data,
            "training_time_seconds": total_time,
            "num_parameters": num_params,
            "history": {k: [float(v) for v in vals] for k, vals in history.items()},
        }
        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=2)

        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Metrics saved to: {metrics_file}")
        print("\nDone!")

    finally:
        # Restore stdout and close logger
        if logger is not None:
            sys.stdout = logger.terminal
            logger.close()


if __name__ == "__main__":
    main()
