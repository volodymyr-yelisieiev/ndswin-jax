#!/usr/bin/env python3
"""Test training script for synthetic datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from training import CheckpointManager, apply_overrides, load_config
from training.trainer import train as train_fn
from training.state import create_model, create_train_state
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm
import optax

# Import synthetic data loaders
from loader_synthetic import get_dataset_info_synthetic, create_input_iter_synthetic


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the ndswin-jax classifier on synthetic data")
    parser.add_argument(
        "--config",
        default="configs.cifar100_test",
        help="Python module path to a get_config() function.",
    )
    parser.add_argument("--workdir", required=True, help="Directory to store checkpoints and logs.")
    parser.add_argument(
        "--dataset",
        choices=["cifar100", "modelnet40"],
        required=True,
        help="Synthetic dataset to use",
    )
    return parser.parse_args()


def mean_metrics(metrics):
    """Average metrics."""
    if not metrics:
        return {"loss": float("nan"), "accuracy": float("nan")}
    import numpy as np
    loss = float(np.mean([m["loss"] for m in metrics]))
    acc = float(np.mean([m["accuracy"] for m in metrics]))
    return {"loss": loss, "accuracy": acc}


def build_train_step(num_classes: int, label_smoothing: float):
    """Build training step function."""
    @jax.jit
    def train_step(state, batch, rng):
        dropout_rng, drop_path_rng = jax.random.split(rng)

        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                batch["image"],
                deterministic=False,
                rngs={"dropout": dropout_rng, "drop_path": drop_path_rng},
            )
            labels = jax.nn.one_hot(batch["label"], num_classes)
            if label_smoothing > 0.0:
                labels = optax.smooth_labels(labels, label_smoothing)
            loss = optax.softmax_cross_entropy(logits, labels).mean()
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch["label"])
        return state, {"loss": loss, "accuracy": accuracy}

    return train_step


def build_eval_step(num_classes: int):
    """Build evaluation step function."""
    @jax.jit
    def eval_step(state, batch):
        logits = state.apply_fn({"params": state.params}, batch["image"], deterministic=True)
        labels = jax.nn.one_hot(batch["label"], num_classes)
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch["label"])
        return {"loss": loss, "accuracy": accuracy}

    return eval_step


def train_synthetic(config, checkpoint_manager, dataset_name: str) -> None:
    """Training loop for synthetic data."""
    # Save config
    checkpoint_manager.save_config(config)

    # Initialize RNG and model
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, train_data_rng, eval_data_rng, step_base_rng = jax.random.split(rng, 5)

    model = create_model(config)
    
    # Get dataset info
    train_info = get_dataset_info_synthetic(dataset_name, "train", config.batch_size)
    eval_info = get_dataset_info_synthetic(dataset_name, "test", config.eval_batch_size)
    steps_per_epoch = train_info.steps_per_epoch
    eval_steps = eval_info.steps_per_epoch
    
    print(f"Training on {train_info.examples} examples ({steps_per_epoch} steps per epoch)")
    print(f"Evaluating on {eval_info.examples} examples ({eval_steps} steps)")
    
    state = create_train_state(model, config, init_rng, steps_per_epoch)

    # Build training and evaluation functions
    train_step_fn = build_train_step(model.num_classes, float(config.label_smoothing))
    eval_step_fn = build_eval_step(model.num_classes)

    # Training loop
    best_eval_accuracy = 0.0
    for epoch in range(config.num_epochs):
        # Training phase
        train_iter = create_input_iter_synthetic(
            dataset_name,
            "train",
            config.batch_size,
            seed=config.seed + epoch,
            shuffle=True,
            repeat=False,
        )

        train_metrics = []
        progress = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for _ in progress:
            batch = next(train_iter)
            step_seed = jax.random.fold_in(step_base_rng, int(state.step))
            state, metrics = train_step_fn(state, batch, step_seed)
            metrics = jax.device_get(metrics)
            train_metrics.append(metrics)
            
            if len(train_metrics) % 10 == 0:
                summary = mean_metrics(train_metrics[-10:])
                progress.set_postfix(
                    train_loss=f"{summary['loss']:.4f}",
                    train_acc=f"{summary['accuracy']:.4f}",
                )

        train_summary = mean_metrics(train_metrics)

        # Evaluation phase
        eval_iter = create_input_iter_synthetic(
            dataset_name,
            "test",
            config.eval_batch_size,
            seed=config.seed + epoch,
            shuffle=False,
            repeat=False,
        )
        
        eval_metrics = []
        for _ in range(eval_steps):
            try:
                batch = next(eval_iter)
                metrics = jax.device_get(eval_step_fn(state, batch))
                eval_metrics.append(metrics)
            except StopIteration:
                break
        eval_summary = mean_metrics(eval_metrics)

        # Save checkpoint
        checkpoint_manager.save(state)
        
        # Save best model
        if eval_summary["accuracy"] > best_eval_accuracy:
            best_eval_accuracy = eval_summary["accuracy"]
            checkpoint_manager.save(state, is_best=True)
            print(f"  ⭐ New best accuracy: {best_eval_accuracy:.4f}")

        # Log metrics
        record = {
            "epoch": epoch + 1,
            "train_loss": train_summary["loss"],
            "train_accuracy": train_summary["accuracy"],
            "eval_loss": eval_summary["loss"],
            "eval_accuracy": eval_summary["accuracy"],
            "step": int(state.step),
            "best_eval_accuracy": best_eval_accuracy,
        }
        checkpoint_manager.save_metrics(record)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_summary['loss']:.4f}, train_acc={train_summary['accuracy']:.4f}, "
            f"eval_loss={eval_summary['loss']:.4f}, eval_acc={eval_summary['accuracy']:.4f}"
        )

    print(f"\nTraining complete! Best eval accuracy: {best_eval_accuracy:.4f}")


def main() -> None:
    """Main training entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    
    checkpoint_manager = CheckpointManager(Path(args.workdir), keep=config.keep_checkpoints)
    train_synthetic(config, checkpoint_manager, args.dataset)


if __name__ == "__main__":
    main()
