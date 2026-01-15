#!/usr/bin/env python3
"""Training script for 2D Swin Transformer on CIFAR-100.

This script demonstrates how to train an NDSwin-JAX model on the CIFAR-100 dataset.

Usage:
    python train/train_cifar100.py --epochs 100 --batch-size 128
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from ndswin import NDSwinConfig, NDSwinTransformer, TrainingConfig
from ndswin.training import (
    CIFAR100DataLoader,
    SyntheticDataLoader,
    Trainer,
)


def augment_data(images, seed):
    """Simple data augmentation for CIFAR-100."""
    rng = jax.random.PRNGKey(seed)

    # Horizontal flip
    rng, flip_rng = jax.random.split(rng)
    do_flip = jax.random.bernoulli(flip_rng, 0.5, (images.shape[0], 1, 1, 1))
    images = jnp.where(do_flip, jnp.flip(images, axis=-1), images)

    return images


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Swin on CIFAR-100")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--embed-dim", type=int, default=96, help="Embedding dimension")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_cifar100",
        help="Checkpoint directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic data for testing",
    )
    return parser.parse_args()


def create_model_config(args: argparse.Namespace) -> NDSwinConfig:
    """Create model configuration for CIFAR-100."""
    return NDSwinConfig(
        num_dims=2,
        patch_size=(4, 4),
        window_size=(4, 4),
        in_channels=3,
        embed_dim=args.embed_dim,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        num_classes=100,
        drop_path_rate=0.1,
        drop_rate=0.0,
    )


def create_training_config(args: argparse.Namespace) -> TrainingConfig:
    """Create training configuration."""
    return TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_epochs=5,
        lr_schedule="cosine",
        label_smoothing=0.1,
        num_classes=100,
    )


def main() -> None:
    """Main training function."""
    args = parse_args()

    print("=" * 60)
    print("NDSwin-JAX: Training on CIFAR-100")
    print("=" * 60)

    # Set random seed
    rng = jax.random.PRNGKey(args.seed)

    # Create configurations
    model_config = create_model_config(args)
    train_config = create_training_config(args)

    print("\nModel config:")
    print(f"  Patch size: {model_config.patch_size}")
    print(f"  Window size: {model_config.window_size}")
    print(f"  Embed dim: {model_config.embed_dim}")
    print(f"  Depths: {model_config.depths}")
    print(f"  Num heads: {model_config.num_heads}")

    print("\nTraining config:")
    print(f"  Epochs: {train_config.num_epochs}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Weight decay: {train_config.weight_decay}")

    # Create model
    model = NDSwinTransformer(config=model_config)

    # Count parameters
    dummy_input = jnp.ones((1, 3, 32, 32))
    variables = model.init(rng, dummy_input, deterministic=True)
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(variables["params"]))
    print(f"\nTotal parameters: {num_params:,}")

    # Create data loaders
    if args.use_synthetic:
        print("\nUsing synthetic data for testing...")
        train_loader = SyntheticDataLoader(
            num_samples=1000,
            input_shape=(3, 32, 32),
            num_classes=100,
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = SyntheticDataLoader(
            num_samples=200,
            input_shape=(3, 32, 32),
            num_classes=100,
            batch_size=args.batch_size,
            shuffle=False,
        )
    else:
        print("\nLoading CIFAR-100...")
        try:
            train_loader = CIFAR100DataLoader(
                data_dir=args.data_dir,
                split="train",
                batch_size=args.batch_size,
                shuffle=True,
                download=True,
                transform=augment_data,
            )
            val_loader = CIFAR100DataLoader(
                data_dir=args.data_dir,
                split="test",
                batch_size=args.batch_size,
                shuffle=False,
                download=True,
            )
            print(f"  Train samples: {train_loader.dataset_info.num_train}")
            print(f"  Test samples: {train_loader.dataset_info.num_test}")
        except ImportError:
            print("Warning: torchvision not available, using synthetic data")
            train_loader = SyntheticDataLoader(
                num_samples=1000,
                input_shape=(3, 32, 32),
                num_classes=100,
                batch_size=args.batch_size,
            )
            val_loader = None

    # Use absolute path for checkpoint directory
    checkpoint_dir = str(Path(args.checkpoint_dir).resolve())

    # Create trainer
    trainer = Trainer(
        model=model,
        config=train_config,
        seed=args.seed,
        log_every=50,
        checkpoint_dir=checkpoint_dir,
    )

    # Train
    print("\nStarting training...")
    _ = trainer.fit(train_loader, val_loader)

    # Final evaluation
    if val_loader is not None:
        val_loader.reset()
        final_metrics = trainer.evaluate(val_loader)
        print("\nFinal validation:")
        print(f"  Loss: {final_metrics['loss']:.4f}")
        print(f"  Accuracy: {final_metrics['accuracy']:.4f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
