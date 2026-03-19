#!/usr/bin/env python3
"""Unified training script for NDSwin-JAX.

This script provides config-driven training for any dimensionality (2D, 3D, 4D+).

Usage:
    python train/train.py --config configs/cifar10.json
    python train/train.py --config configs/cifar10.json --epochs 200
    make train CONFIG=configs/cifar10.json
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
from ndswin.utils.logging import get_logger, setup_logging

logger = get_logger("train")


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
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level",
    )
    parser.add_argument(
        "--stamp",
        type=str,
        default=None,
        help="Override experiment stamp (used for directories and log files)",
    )

    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Cap the number of training steps per epoch (useful for sweep budget control)",
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

    # Setup structured logging
    setup_logging(
        level=args.log_level,
        log_file=None if args.no_log_file else str(log_file),
        log_to_file=not args.no_log_file,
        log_to_console=True,
        use_rich=False,  # Plain text for consistency in log files
    )

    # Print header
    logger.info("=" * 70)
    logger.info("NDSwin-JAX Training")
    logger.info("=" * 70)
    logger.info("Config file: %s", config_path)
    logger.info("Experiment: %s", exp_config.name)
    logger.info("Description: %s", exp_config.description)
    logger.info("Config hash: %s", exp_config.get_config_hash())
    logger.info("Checkpoint dir: %s", checkpoint_dir)
    logger.info("Log file: %s", log_file)
    logger.info("Timestamp: %s", datetime.now().isoformat())

    # Set random seed
    rng = jax.random.PRNGKey(exp_config.training.seed)
    np.random.seed(exp_config.training.seed)

    # Create configurations
    model_config = exp_config.model
    train_config = exp_config.training

    # Print configurations
    logger.info("-" * 70)
    logger.info("Dataset Configuration:")
    for key, value in exp_config.data.to_dict().items():
        logger.debug("  %s: %s", key, value)

    logger.info("-" * 70)
    logger.info("Model Configuration:")
    logger.info("  num_dims: %s", model_config.num_dims)
    logger.info("  patch_size: %s", model_config.patch_size)
    logger.info("  window_size: %s", model_config.window_size)
    logger.info("  embed_dim: %s", model_config.embed_dim)
    logger.info("  depths: %s", model_config.depths)
    logger.info("  num_heads: %s", model_config.num_heads)
    logger.debug("  drop_path_rate: %s", model_config.drop_path_rate)
    logger.info("  num_classes: %s", model_config.num_classes)

    logger.info("-" * 70)
    logger.info("Training Configuration:")
    logger.info("  epochs: %s", train_config.num_epochs)
    logger.info("  batch_size: %s", train_config.batch_size)
    logger.info("  learning_rate: %s", train_config.learning_rate)
    logger.debug("  weight_decay: %s", train_config.weight_decay)
    logger.debug("  optimizer: %s", train_config.optimizer)
    logger.debug("  lr_schedule: %s", train_config.lr_schedule)
    logger.debug("  warmup_epochs: %s", train_config.warmup_epochs)
    logger.debug("  label_smoothing: %s", train_config.label_smoothing)

    

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
    logger.info("Total parameters: %s", f"{num_params:,}")

    # Create augmentation transforms
    # Create data loaders
    logger.info("Loading data...")
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
            logger.warning("Could not load validation or test split: %s", e)
            val_loader = None

    
    if hasattr(train_loader, "dataset_info"):
        logger.info("  Dataset: %s", train_loader.dataset_info.name)
        logger.info("  Train samples: %s", train_loader.dataset_info.num_train)
        logger.info("  Test samples: %s", train_loader.dataset_info.num_test)
    else:
        logger.info("  Dataset: %s", dataset_name)

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
        logger.debug("  Mixup alpha: %s", mixup_alpha)
        logger.debug("  CutMix alpha: %s", cutmix_alpha)

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
        use_tensorboard=True,
    )

    # Train
    logger.info("=" * 70)
    logger.info("Starting training for %d epochs", train_config.num_epochs)
    logger.info("=" * 70)

    start_time = time.time()
    history = trainer.fit(train_loader, val_loader, max_steps_per_epoch=getattr(args, 'max_steps_per_epoch', None))
    total_time = time.time() - start_time

    # Final evaluation
    if val_loader is not None:
        val_loader.reset()
        final_metrics = trainer.evaluate(val_loader)
        logger.info("=" * 70)
        logger.info("Final Validation Results:")
        logger.info("=" * 70)
        logger.info("  Loss: %.4f", final_metrics['loss'])
        if task == "segmentation":
            logger.info("  Dice: %.4f", final_metrics.get('val_dice', 0.0))
            logger.info("  Voxel Acc: %.4f", final_metrics.get('val_voxel_accuracy', 0.0))
        else:
            logger.info("  Accuracy: %.4f", final_metrics['accuracy'])
            logger.info("  Top-5 Accuracy: %.4f", final_metrics['top5_accuracy'])

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

    logger.info("Training completed in %.2fs", total_time)
    logger.info("Metrics saved to: %s", metrics_file)
    logger.info("Done!")


if __name__ == "__main__":
    main()
