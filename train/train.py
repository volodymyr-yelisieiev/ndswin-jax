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

import jax
import jax.numpy as jnp
import numpy as np

from ndswin import NDSwinConfig, NDSwinTransformer, TrainingConfig
from ndswin.training import (
    CIFAR10DataLoader,
    CIFAR100DataLoader,
    Cutout,
    Cutmix,
    Mixup,
    MixupOrCutmix,
    RandomCrop,
    RandomHorizontalFlip,
    SyntheticDataLoader,
    NumpySegmentationFolderDataLoader,
    Trainer,
)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration loaded from JSON."""

    name: str
    description: str = ""

    # Dataset configuration
    dataset: dict[str, Any] = field(default_factory=dict)

    # Model configuration
    model: dict[str, Any] = field(default_factory=dict)

    # Training configuration
    training: dict[str, Any] = field(default_factory=dict)

    # Augmentation configuration
    augmentation: dict[str, Any] = field(default_factory=dict)

    # Checkpointing configuration
    checkpointing: dict[str, Any] = field(default_factory=dict)

    # Logging configuration
    logging: dict[str, Any] = field(default_factory=dict)

    # Random seed
    seed: int = 42

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            name=data.get("name", "experiment"),
            description=data.get("description", ""),
            dataset=data.get("dataset", {}),
            model=data.get("model", {}),
            training=data.get("training", {}),
            augmentation=data.get("augmentation", {}),
            checkpointing=data.get("checkpointing", {}),
            logging=data.get("logging", {}),
            seed=data.get("seed", 42),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "dataset": self.dataset,
            "model": self.model,
            "training": self.training,
            "augmentation": self.augmentation,
            "checkpointing": self.checkpointing,
            "logging": self.logging,
            "seed": self.seed,
        }

    def get_config_hash(self) -> str:
        """Generate a short hash of the config for identification."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def get_stamp(self) -> str:
        """Generate a unique stamp for this experiment run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.name}_{timestamp}"


def create_model_config(exp_config: ExperimentConfig) -> NDSwinConfig:
    """Create NDSwinConfig from experiment configuration."""
    model_cfg = exp_config.model
    dataset_cfg = exp_config.dataset

    # Get number of classes from dataset config
    num_classes = dataset_cfg.get("num_classes", 10)

    # Convert lists to tuples for config
    patch_size = tuple(model_cfg.get("patch_size", [4, 4]))
    window_size = tuple(model_cfg.get("window_size", [4, 4]))
    depths = tuple(model_cfg.get("depths", [2, 2, 6, 2]))
    num_heads = tuple(model_cfg.get("num_heads", [3, 6, 12, 24]))

    return NDSwinConfig(
        num_dims=model_cfg.get("num_dims", 2),
        patch_size=patch_size,
        window_size=window_size,
        in_channels=model_cfg.get("in_channels", 3),
        embed_dim=model_cfg.get("embed_dim", 96),
        depths=depths,
        num_heads=num_heads,
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        qkv_bias=model_cfg.get("qkv_bias", True),
        drop_rate=model_cfg.get("drop_rate", 0.0),
        attn_drop_rate=model_cfg.get("attn_drop_rate", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
        num_classes=num_classes,
    )


def create_training_config(exp_config: ExperimentConfig) -> TrainingConfig:
    """Create TrainingConfig from experiment configuration."""
    train_cfg = exp_config.training
    dataset_cfg = exp_config.dataset

    return TrainingConfig(
        epochs=train_cfg.get("epochs", 100),
        batch_size=train_cfg.get("batch_size", 128),
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        min_learning_rate=train_cfg.get("min_learning_rate", 1e-6),
        weight_decay=train_cfg.get("weight_decay", 0.05),
        optimizer=train_cfg.get("optimizer", "adamw"),
        scheduler=train_cfg.get("lr_schedule", "cosine"),
        lr_schedule=train_cfg.get("lr_schedule", "cosine"),
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
        label_smoothing=train_cfg.get("label_smoothing", 0.1),
        gradient_clip_norm=train_cfg.get("gradient_clip_norm", 1.0),
        max_grad_norm=train_cfg.get("gradient_clip_norm", 1.0),
        num_classes=dataset_cfg.get("num_classes", 10),
        loss=train_cfg.get("loss", "cross_entropy"),
        bce_pos_weight=train_cfg.get("bce_pos_weight", None),
    )


def create_augmentation_transform(
    exp_config: ExperimentConfig,
) -> tuple[Any | None, Mixup | None]:
    """Create augmentation transforms from experiment configuration.

    Returns:
        Tuple of (per_sample_transform, batch_mixup_transform)
    """
    aug_cfg = exp_config.augmentation
    dataset_cfg = exp_config.dataset

    transforms = []

    # Random crop with padding
    if aug_cfg.get("random_crop", False):
        input_shape = dataset_cfg.get("input_shape", [3, 32, 32])
        spatial_size = tuple(input_shape[1:])  # Exclude channel dim
        padding = aug_cfg.get("crop_padding", 4)
        transforms.append(RandomCrop(size=spatial_size, padding=padding))

    # Random horizontal flip
    if aug_cfg.get("random_flip", False):
        transforms.append(RandomHorizontalFlip(p=0.5))

    # Cutout
    cutout_size = aug_cfg.get("cutout_size", 0)
    if cutout_size > 0:
        transforms.append(Cutout(size=cutout_size, p=0.5))

    # Mixup and Cutmix (applied at batch level)
    mixup_transform = None
    mixup_alpha = aug_cfg.get("mixup_alpha", 0.0)
    cutmix_alpha = aug_cfg.get("cutmix_alpha", 0.0)
    num_classes = dataset_cfg.get("num_classes", 10)

    if mixup_alpha > 0 and cutmix_alpha > 0:
        mixup_transform = MixupOrCutmix(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            p=0.5,
            num_classes=num_classes,
        )
    elif mixup_alpha > 0:
        mixup_transform = Mixup(alpha=mixup_alpha, num_classes=num_classes)
    elif cutmix_alpha > 0:
        mixup_transform = Cutmix(alpha=cutmix_alpha, num_classes=num_classes)

    # Compose per-sample transforms
    per_sample_transform = None
    if transforms:

        @jax.jit
        def apply_transforms(images: jnp.ndarray, seed: int) -> jnp.ndarray:
            """Apply per-sample transforms to a batch."""
            rng = jax.random.PRNGKey(seed)
            batch_size = images.shape[0]
            keys = jax.random.split(rng, batch_size)

            def transform_single(img: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
                for t in transforms:
                    key, subkey = jax.random.split(key)
                    img = t(img, subkey)
                return img

            return jax.vmap(transform_single)(images, keys)

        per_sample_transform = apply_transforms

    return per_sample_transform, mixup_transform


def create_data_loaders(
    exp_config: ExperimentConfig,
    transform: Any | None = None,
) -> tuple[Any, Any | None]:
    """Create train and validation data loaders from experiment configuration."""
    dataset_cfg = exp_config.dataset
    train_cfg = exp_config.training

    dataset_name = dataset_cfg.get("name", "cifar10").lower()
    batch_size = train_cfg.get("batch_size", 128)
    data_dir = dataset_cfg.get("data_dir", "data")
    download = dataset_cfg.get("download", True)

    if dataset_name == "cifar10":
        train_loader = CIFAR10DataLoader(
            data_dir=data_dir,
            split="train",
            batch_size=batch_size,
            shuffle=True,
            download=download,
            transform=transform,
        )
        val_loader = CIFAR10DataLoader(
            data_dir=data_dir,
            split="test",
            batch_size=batch_size,
            shuffle=False,
            download=download,
        )
    elif dataset_name == "cifar100":
        train_loader = CIFAR100DataLoader(
            data_dir=data_dir,
            split="train",
            batch_size=batch_size,
            shuffle=True,
            download=download,
            transform=transform,
        )
        val_loader = CIFAR100DataLoader(
            data_dir=data_dir,
            split="test",
            batch_size=batch_size,
            shuffle=False,
            download=download,
        )
    elif dataset_name == "synthetic":
        input_shape = tuple(dataset_cfg.get("input_shape", [3, 32, 32]))
        num_classes = dataset_cfg.get("num_classes", 10)
        num_samples_train = dataset_cfg.get("num_samples_train", 1000)
        num_samples_val = dataset_cfg.get("num_samples_val", 200)

        train_loader = SyntheticDataLoader(
            num_samples=num_samples_train,
            input_shape=input_shape,
            num_classes=num_classes,
            batch_size=batch_size,
            shuffle=True,
            seed=exp_config.seed,
        )
        val_loader = SyntheticDataLoader(
            num_samples=num_samples_val,
            input_shape=input_shape,
            num_classes=num_classes,
            batch_size=batch_size,
            shuffle=False,
            seed=exp_config.seed + 1,
        )
    elif dataset_name in {"medseg_msd", "medseg", "medseg_brain_tumour"}:
        # Segmentation dataset saved as NPZ pairs by train/fetch_medseg_msd.py
        train_loader = NumpySegmentationFolderDataLoader(
            name=dataset_name,
            data_dir=data_dir,
            split="train",
            batch_size=batch_size,
            shuffle=True,
            in_channels=dataset_cfg.get("in_channels", 1),
            image_size=tuple(dataset_cfg.get("image_size", [64, 64, 64])),
            mean=tuple(dataset_cfg.get("mean", (0.0,))),
            std=tuple(dataset_cfg.get("std", (1.0,))),
        )
        val_loader = NumpySegmentationFolderDataLoader(
            name=dataset_name,
            data_dir=data_dir,
            split="validation" if os.path.exists(os.path.join(data_dir, "validation")) else "test",
            batch_size=batch_size,
            shuffle=False,
            in_channels=dataset_cfg.get("in_channels", 1),
            image_size=tuple(dataset_cfg.get("image_size", [64, 64, 64])),
            mean=tuple(dataset_cfg.get("mean", (0.0,))),
            std=tuple(dataset_cfg.get("std", (1.0,))),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_loader, val_loader


def setup_output_dirs(
    exp_config: ExperimentConfig,
    config_path: str,
    stamp: str | None = None,
) -> tuple[Path, Path]:
    """Setup output directories for checkpoints and logs.

    Standard layout:
        outputs/<dataset_name>/<stamp>/checkpoints/
        logs/<dataset_name>/<stamp>.log

    Returns:
        Tuple of (checkpoint_dir, log_file_path)
    """
    if stamp is None:
        stamp = exp_config.get_stamp()

    # Dataset name for directory organization
    dataset_name = exp_config.dataset.get("hf_id") or exp_config.dataset.get("dataset") or "dataset"
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
        exp_config.training["epochs"] = args.epochs
    if args.batch_size is not None:
        exp_config.training["batch_size"] = args.batch_size
    if args.lr is not None:
        exp_config.training["learning_rate"] = args.lr
    if args.seed is not None:
        exp_config.seed = args.seed
    if args.data_dir is not None:
        exp_config.dataset["data_dir"] = args.data_dir


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    exp_config = ExperimentConfig.from_json(config_path)

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
        rng = jax.random.PRNGKey(exp_config.seed)
        np.random.seed(exp_config.seed)

        # Create configurations
        model_config = create_model_config(exp_config)
        train_config = create_training_config(exp_config)

        # Print configurations
        print("\n" + "-" * 70)
        print("Dataset Configuration:")
        print("-" * 70)
        for key, value in exp_config.dataset.items():
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

        print("\n" + "-" * 70)
        print("Augmentation Configuration:")
        print("-" * 70)
        for key, value in exp_config.augmentation.items():
            print(f"  {key}: {value}")

        # Create model
        model = NDSwinTransformer(config=model_config)

        # Count parameters
        input_shape = tuple(exp_config.dataset.get("input_shape", [3, 32, 32]))
        dummy_input = jnp.ones((1,) + input_shape)
        variables = model.init(rng, dummy_input, deterministic=True)
        num_params = sum(p.size for p in jax.tree_util.tree_leaves(variables["params"]))
        print(f"\nTotal parameters: {num_params:,}")

        # Create augmentation transforms
        per_sample_transform, mixup_transform = create_augmentation_transform(exp_config)

        # Create data loaders
        print("\nLoading data...")
        train_loader, val_loader = create_data_loaders(exp_config, per_sample_transform)

        dataset_name = exp_config.dataset.get("name", "unknown")
        if hasattr(train_loader, "dataset_info"):
            print(f"  Dataset: {train_loader.dataset_info.name}")
            print(f"  Train samples: {train_loader.dataset_info.num_train}")
            print(f"  Test samples: {train_loader.dataset_info.num_test}")
        else:
            print(f"  Dataset: {dataset_name}")

        # Create trainer
        log_every = exp_config.logging.get("log_every", 50)
        checkpoint_every = exp_config.checkpointing.get("save_every", 5000)

        trainer = Trainer(
            model=model,
            config=train_config,
            seed=exp_config.seed,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            checkpoint_dir=str(checkpoint_dir),
            mixup_transform=mixup_transform,
            task=exp_config.dataset.get("task", "classification"),
            loss_name=train_config.loss,
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
            if exp_config.dataset.get("task", "classification") == "segmentation":
                print(f"  Dice: {final_metrics.get('val_dice', 0.0):.4f}")
                print(f"  Voxel Acc: {final_metrics.get('val_voxel_accuracy', 0.0):.4f}")
            else:
                print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
                print(f"  Top-5 Accuracy: {final_metrics['top5_accuracy']:.4f}")

        # Save final metrics
        metrics_file = checkpoint_dir / "metrics.json"
        final_metrics_data: dict[str, float] = {"loss": float(final_metrics["loss"]) if val_loader else None}
        if exp_config.dataset.get("task", "classification") == "segmentation":
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
