"""Main training loop and evaluation logic."""

from __future__ import annotations

from typing import Dict, Iterable

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from ml_collections import ConfigDict
from tqdm.auto import tqdm

from loader import create_input_iter, get_dataset_info

from .augmentation import apply_augmentation
from .checkpointing import CheckpointManager
from .state import create_model, create_train_state


def mean_metrics(metrics: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Average metrics emitted by the training/eval loop.
    
    Args:
        metrics: Iterable of metric dictionaries
        
    Returns:
        Dictionary with averaged metrics
    """
    metrics = list(metrics)
    if not metrics:
        return {"loss": float("nan"), "accuracy": float("nan")}
    loss = float(jnp.mean(jnp.array([m["loss"] for m in metrics])))
    acc = float(jnp.mean(jnp.array([m["accuracy"] for m in metrics])))
    return {"loss": loss, "accuracy": acc}


def _build_train_step(num_classes: int, label_smoothing: float, use_augmentation: bool = False):
    """Build JIT-compiled training step function.
    
    Args:
        num_classes: Number of output classes
        label_smoothing: Label smoothing factor
        use_augmentation: Whether data augmentation with mixup/cutmix is used
        
    Returns:
        Compiled training step function
    """
    @jax.jit
    def train_step(
        state: train_state.TrainState, batch: Dict[str, jnp.ndarray], rng: jax.Array
    ):
        dropout_rng, drop_path_rng = jax.random.split(rng)

        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                batch["image"],
                deterministic=False,
                rngs={"dropout": dropout_rng, "drop_path": drop_path_rng},
            )
            
            # Use pre-computed one-hot labels if augmentation was applied
            if use_augmentation and "label_onehot" in batch:
                labels = batch["label_onehot"]
                if label_smoothing > 0.0:
                    labels = optax.smooth_labels(labels, label_smoothing)
            else:
                labels = jax.nn.one_hot(batch["label"], num_classes)
                if label_smoothing > 0.0:
                    labels = optax.smooth_labels(labels, label_smoothing)
            
            loss = optax.softmax_cross_entropy(logits, labels).mean()
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch["label"])
        metrics = {"loss": loss, "accuracy": accuracy}
        return state, metrics

    return train_step


def _build_eval_step(num_classes: int):
    """Build JIT-compiled evaluation step function.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Compiled evaluation step function
    """
    @jax.jit
    def eval_step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray]):
        logits = state.apply_fn({"params": state.params}, batch["image"], deterministic=True)
        labels = jax.nn.one_hot(batch["label"], num_classes)
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch["label"])
        return {"loss": loss, "accuracy": accuracy}

    return eval_step


def train(config: ConfigDict, checkpoint_manager: CheckpointManager) -> None:
    """Full training loop, including checkpointing and logging.
    
    Args:
        config: Configuration dictionary with all training parameters
        checkpoint_manager: Manager for saving/loading checkpoints
    """
    # Save config
    checkpoint_manager.save_config(config)

    # Initialize RNG and model
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, train_data_rng, eval_data_rng, step_base_rng = jax.random.split(rng, 5)

    model = create_model(config)
    
    # Get dataset info first to create train state with lr schedule
    train_info = get_dataset_info(config, config.train_split, config.batch_size)
    eval_info = get_dataset_info(config, config.eval_split, config.eval_batch_size)
    steps_per_epoch = train_info.steps_per_epoch
    eval_steps = eval_info.steps_per_epoch
    if config.max_eval_batches is not None:
        eval_steps = min(eval_steps, int(config.max_eval_batches))
    
    state = create_train_state(model, config, init_rng, steps_per_epoch)

    # Restore from checkpoint if available
    state = checkpoint_manager.restore(state)
    initial_step = int(state.step)

    # Generate data seeds
    train_seed = int(jax.random.randint(train_data_rng, (), 0, 1_000_000).item())
    eval_seed = int(jax.random.randint(eval_data_rng, (), 0, 1_000_000).item())

    # Build training and evaluation functions
    use_augmentation = getattr(config, "use_augmentation", False)
    train_step_fn = _build_train_step(
        model.num_classes, 
        float(config.label_smoothing),
        use_augmentation
    )
    eval_step_fn = _build_eval_step(model.num_classes)

    start_epoch = initial_step // steps_per_epoch if steps_per_epoch else 0
    
    # Track best accuracy for model selection
    best_eval_accuracy = 0.0

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        # Training phase
        train_iter = create_input_iter(
            config,
            config.train_split,
            config.batch_size,
            seed=train_seed + epoch,
            shuffle=True,
            repeat=True,
        )

        train_metrics = []
        progress = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for _ in progress:
            batch = next(train_iter)
            
            # Apply augmentation if enabled
            if use_augmentation:
                aug_rng = jax.random.fold_in(step_base_rng, int(state.step) * 2)
                batch = apply_augmentation(aug_rng, batch, config, model.num_classes)
            step_seed = jax.random.fold_in(step_base_rng, int(state.step))
            state, metrics = train_step_fn(state, batch, step_seed)
            metrics = jax.device_get(metrics)
            train_metrics.append(metrics)
            if len(train_metrics) % config.log_every == 0:
                summary = mean_metrics(train_metrics[-config.log_every :])
                progress.set_postfix(
                    train_loss=f"{summary['loss']:.4f}",
                    train_acc=f"{summary['accuracy']:.4f}",
                )

        train_summary = mean_metrics(train_metrics)

        # Evaluation phase
        eval_iter = create_input_iter(
            config,
            config.eval_split,
            config.eval_batch_size,
            seed=eval_seed + epoch,
            shuffle=False,
            repeat=False,
        )
        eval_metrics = []
        for _ in range(eval_steps):
            try:
                batch = next(eval_iter)
            except StopIteration:
                break
            metrics = jax.device_get(eval_step_fn(state, batch))
            eval_metrics.append(metrics)
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


__all__ = ["mean_metrics", "train"]
