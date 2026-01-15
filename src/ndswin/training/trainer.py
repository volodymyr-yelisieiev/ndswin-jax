"""Training loop and utilities for NDSwin-JAX.

This module provides the main training infrastructure.
"""

import time
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training import train_state

from ndswin.config import TrainingConfig
from ndswin.training.checkpoint import CheckpointManager
from ndswin.training.losses import (
    cross_entropy_loss,
    dice_loss,
    binary_cross_entropy_with_logits,
)
from ndswin.training.metrics import MetricTracker, accuracy, top_k_accuracy, compute_segmentation_metrics
from ndswin.training.optimizer import create_optimizer
from ndswin.types import Array, Batch, PRNGKey


class TrainState(train_state.TrainState):
    """Extended train state with additional fields.

    Attributes:
        batch_stats: Optional batch normalization statistics.
        dropout_rng: Random key for dropout.
        epoch: Current epoch.
    """

    batch_stats: dict[str, Any] | None = None
    dropout_rng: PRNGKey | None = None
    epoch: int = 0


def create_train_state(
    model: nn.Module,
    config: TrainingConfig,
    rng: PRNGKey,
    input_shape: tuple[int, ...],
    num_steps: int,
    num_train_samples: int | None = None,
) -> TrainState:
    """Create initial training state.

    Args:
        model: Flax model.
        config: Training configuration.
        rng: Random key.
        input_shape: Shape of input (without batch dimension).
        num_steps: Total number of training steps.
        num_train_samples: Number of training samples.

    Returns:
        Initial training state.
    """
    # Split RNG
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)

    # Create model
    if hasattr(model, "config") and hasattr(config, "stochastic_depth_rate"):
        # Sync drop path rate if provided in training config
        model.config.drop_path_rate = config.stochastic_depth_rate

    # Initialize model
    dummy_input = jnp.ones((1,) + input_shape)
    variables = model.init(init_rng, dummy_input, deterministic=False)

    # Extract params and batch stats
    params = variables["params"]
    batch_stats = variables.get("batch_stats")

    # Create optimizer
    optimizer = create_optimizer(config, num_steps, num_train_samples)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats,
        dropout_rng=dropout_rng,
    )
    return cast(TrainState, state)


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def train_step(
    state: TrainState,
    batch: Batch,
    rng: PRNGKey,
    label_smoothing: float = 0.0,
    use_batch_stats: bool = False,
    num_classes: int = 10,
    task: str = "classification",
    loss_name: str = "cross_entropy",
) -> tuple[TrainState, dict[str, Any]]:
    """Perform a single training step.

    This supports both classification and segmentation tasks (dice/bce/dice_bce).
    """
    images = batch["image"]
    labels = batch["label"]

    # Split RNG for this step
    dropout_rng, new_dropout_rng = jax.random.split(rng)

    def loss_fn(params: Any) -> tuple[Array, dict[str, Any]]:
        """Compute loss and metrics."""
        # Forward pass
        if use_batch_stats:
            logits, updates = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                images,
                deterministic=False,
                mutable=["batch_stats"],
                rngs={"dropout": dropout_rng},
            )
            new_batch_stats = updates["batch_stats"]
        else:
            logits = state.apply_fn(
                {"params": params},
                images,
                deterministic=False,
                rngs={"dropout": dropout_rng},
            )
            new_batch_stats = None

        # Choose loss based on task
        if task == "segmentation":
            # labels expected (B, D, H, W) or (B, C, D, H, W)
            if labels.ndim == logits.ndim - 1:
                # (B, D, H, W) -> to one-hot (B, C, D, H, W)
                lbl_onehot = jax.nn.one_hot(labels, num_classes)
                lbl_onehot = jnp.moveaxis(lbl_onehot, -1, 1)
            else:
                lbl_onehot = labels

            if loss_name == "dice":
                loss = dice_loss(logits, lbl_onehot)
            elif loss_name == "bce":
                # binary segmentation: use channel 1 logits
                fg_logits = logits[:, 1, ...]
                fg_labels = lbl_onehot[:, 1, ...]
                loss = binary_cross_entropy_with_logits(fg_logits, fg_labels)
            elif loss_name in {"bce_dice", "dice_bce"}:
                fg_logits = logits[:, 1, ...]
                fg_labels = lbl_onehot[:, 1, ...]
                bce = binary_cross_entropy_with_logits(fg_logits, fg_labels)
                dloss = dice_loss(logits, lbl_onehot)
                loss = 0.5 * bce + 0.5 * dloss
            else:
                # default to dice
                loss = dice_loss(logits, lbl_onehot)

            metrics = compute_segmentation_metrics(logits, labels, prefix="train_")
            metrics["loss"] = loss
            return loss, (metrics, logits, new_batch_stats)

        else:
            # Classification path
            loss = cast(Array, cross_entropy_loss(logits, labels, label_smoothing))
            acc = cast(Array, accuracy(logits, labels))
            top5_acc = cast(Array, top_k_accuracy(logits, labels, k=5))
            metrics = {"loss": loss, "accuracy": acc, "top5_accuracy": top5_acc}
            return loss, (metrics, logits, new_batch_stats)

    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (metrics_aux, logits, new_batch_stats)), grads = grad_fn(state.params)

    # Update parameters
    state = state.apply_gradients(grads=grads)

    # Update batch stats if present
    if new_batch_stats is not None:
        state = state.replace(batch_stats=new_batch_stats)

    # Update dropout RNG
    state = state.replace(dropout_rng=new_dropout_rng)

    # Ensure metrics has expected scalar entries for Tracker
    if "loss" not in metrics_aux:
        metrics_aux["loss"] = loss

    return state, metrics_aux


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def eval_step(
    state: TrainState,
    batch: Batch,
    use_batch_stats: bool = False,
    num_classes: int = 10,
    task: str = "classification",
    loss_name: str = "cross_entropy",
) -> dict[str, Any]:
    """Perform a single evaluation step.

    Supports classification and segmentation metrics.
    """
    images = batch["image"]
    labels = batch["label"]

    # Forward pass (no dropout)
    if use_batch_stats:
        logits = state.apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats},
            images,
            deterministic=True,
        )
    else:
        logits = state.apply_fn(
            {"params": state.params},
            images,
            deterministic=True,
        )

    # Compute metrics depending on task
    if task == "segmentation":
        if labels.ndim == logits.ndim - 1:
            lbl_onehot = jax.nn.one_hot(labels, num_classes)
            lbl_onehot = jnp.moveaxis(lbl_onehot, -1, 1)
        else:
            lbl_onehot = labels

        if loss_name == "dice":
            loss = dice_loss(logits, lbl_onehot)
        elif loss_name == "bce":
            fg_logits = logits[:, 1, ...]
            fg_labels = lbl_onehot[:, 1, ...]
            loss = binary_cross_entropy_with_logits(fg_logits, fg_labels)
        elif loss_name in {"bce_dice", "dice_bce"}:
            fg_logits = logits[:, 1, ...]
            fg_labels = lbl_onehot[:, 1, ...]
            bce = binary_cross_entropy_with_logits(fg_logits, fg_labels)
            dloss = dice_loss(logits, lbl_onehot)
            loss = 0.5 * bce + 0.5 * dloss
        else:
            loss = dice_loss(logits, lbl_onehot)

        metrics = compute_segmentation_metrics(logits, labels, prefix="val_")
        metrics["loss"] = loss
        return metrics

    else:
        loss = cross_entropy_loss(logits, labels)
        acc = accuracy(logits, labels)
        top5_acc = top_k_accuracy(logits, labels, k=5)

        return {
            "loss": loss,
            "accuracy": acc,
            "top5_accuracy": top5_acc,
        }


class Trainer:
    """Main trainer class for NDSwin models.

    Handles the training loop, evaluation, logging, and checkpointing.

    Example:
        >>> trainer = Trainer(model, config)
        >>> trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        seed: int = 42,
        log_every: int = 50,
        eval_every: int = 1000,
        checkpoint_every: int = 5000,
        checkpoint_dir: str | None = None,
        mixup_transform: Any | None = None,
        task: str = "classification",
        loss_name: str = "cross_entropy",
    ) -> None:
        """Initialize trainer.

        Args:
            model: Flax model to train.
            config: Training configuration.
            seed: Random seed.
            log_every: Steps between logging.
            eval_every: Steps between evaluation.
            checkpoint_every: Steps between checkpoints.
            checkpoint_dir: Directory for checkpoints.
            mixup_transform: Optional Mixup transform applied per batch.
        """
        self.model = model
        self.config = config
        self.seed = seed
        self.log_every = log_every
        self.eval_every = eval_every
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir
        self.mixup_transform = mixup_transform
        self.task = task
        self.loss_name = loss_name

        self.rng = jax.random.PRNGKey(seed)
        self.state: TrainState | None = None
        self.step = 0
        self.epoch = 0

        # Checkpoint manager
        self.checkpoint_manager = None
        if checkpoint_dir is not None:
            self.checkpoint_manager = CheckpointManager(
                directory=checkpoint_dir,
                max_to_keep=config.max_checkpoints if hasattr(config, "max_checkpoints") else 5,
            )

        # Callbacks
        self.callbacks: list[Callable] = []

    def init_state(
        self,
        input_shape: tuple[int, ...],
        num_steps: int,
        num_train_samples: int | None = None,
    ) -> None:
        """Initialize training state.

        Args:
            input_shape: Shape of input (without batch).
            num_steps: Total training steps.
            num_train_samples: Number of training samples.
        """
        self.rng, init_rng = jax.random.split(self.rng)
        self.state = create_train_state(
            self.model,
            self.config,
            init_rng,
            input_shape,
            num_steps,
            num_train_samples,
        )

    def train_epoch(
        self,
        train_loader: Iterator[Batch],
        num_steps: int | None = None,
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data iterator.
            num_steps: Optional max steps (for debugging).

        Returns:
            Average metrics for epoch.
        """
        if self.state is None:
            raise RuntimeError("Must call init_state before training")

        tracker = MetricTracker()
        use_batch_stats = self.state.batch_stats is not None

        for i, batch in enumerate(train_loader):
            if num_steps is not None and i >= num_steps:
                break

            # Get RNG for this step
            self.rng, step_rng = jax.random.split(self.rng)

            # Apply batch-level Mixup if configured
            if self.mixup_transform is not None:
                # Use a subkey for mixup so dropout RNG is independent
                step_rng, mix_key = jax.random.split(step_rng)
                
                # JIT the mixup transform if it hasn't been
                if not hasattr(self, "_jitted_mixup"):
                    self._jitted_mixup = jax.jit(self.mixup_transform)
                
                mixed_x, mixed_y = self._jitted_mixup(batch["image"], batch["label"], mix_key)
                batch = {"image": mixed_x, "label": mixed_y}

            # Train step
            self.state, metrics = train_step(
                self.state,
                batch,
                step_rng,
                self.config.label_smoothing,
                use_batch_stats,
                self.config.num_classes,
                self.task,
                self.loss_name,
            )

            # Update tracking
            tracker.update(metrics, batch["image"].shape[0])
            self.step += 1

            # Logging
            if self.step % self.log_every == 0:
                _ = tracker.compute()
                print(f"Step {self.step}: {tracker}")

            # Checkpointing
            if self.checkpoint_dir is not None and self.step % self.checkpoint_every == 0:
                self._save_checkpoint()

        self.epoch += 1
        return tracker.compute()

    def evaluate(
        self,
        eval_loader: Iterator[Batch],
        num_steps: int | None = None,
    ) -> dict[str, float]:
        """Evaluate on a dataset.

        Args:
            eval_loader: Evaluation data iterator.
            num_steps: Optional max steps.

        Returns:
            Average metrics.
        """
        if self.state is None:
            raise RuntimeError("Must call init_state before evaluation")

        tracker = MetricTracker()
        use_batch_stats = self.state.batch_stats is not None

        for i, batch in enumerate(eval_loader):
            if num_steps is not None and i >= num_steps:
                break

            metrics = eval_step(
                self.state,
                batch,
                use_batch_stats,
                self.config.num_classes,
                self.task,
                self.loss_name,
            )
            tracker.update(metrics, batch["image"].shape[0])

        return tracker.compute()

    def fit(
        self,
        train_loader: Any,
        val_loader: Any | None = None,
        num_epochs: int | None = None,
    ) -> dict[str, list[float]]:
        """Run full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            num_epochs: Number of epochs (defaults to config.num_epochs).

        Returns:
            Training history.
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        # Initialize state if not done
        if self.state is None:
            # Get a sample batch to determine input shape
            sample_batch = next(iter(train_loader))
            input_shape = sample_batch["image"].shape[1:]
            num_steps = num_epochs * len(train_loader)
            self.init_state(input_shape, num_steps, len(train_loader) * self.config.batch_size)
            train_loader.reset()  # Reset after getting sample

        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "train_top5_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_top5_accuracy": [],
        }

        print(f"Starting training for {num_epochs} epochs")
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["train_top5_accuracy"].append(train_metrics["top5_accuracy"])

            # Evaluate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                history["val_top5_accuracy"].append(val_metrics["top5_accuracy"])

                print(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f}, "
                    f"Acc: {train_metrics['accuracy']:.4f}/{val_metrics['accuracy']:.4f}, "
                    f"Top-5: {train_metrics['top5_accuracy']:.4f}/{val_metrics['top5_accuracy']:.4f}, "
                    f"Time: {time.time() - epoch_start:.2f}s"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Time: {time.time() - epoch_start:.2f}s"
                )

            # Reset data loader
            if hasattr(train_loader, "reset"):
                train_loader.reset()
            if val_loader is not None and hasattr(val_loader, "reset"):
                val_loader.reset()

            # Run callbacks
            for callback in self.callbacks:
                callback(self, epoch, train_metrics, history)

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")

        return history

    def _save_checkpoint(self, metrics: dict[str, float] | None = None) -> None:
        """Save a training checkpoint."""
        if self.checkpoint_manager is None or self.state is None:
            return

        self.checkpoint_manager.save(
            params=self.state.params,
            step=self.step,
            epoch=self.epoch,
            metrics=metrics,
            batch_stats=self.state.batch_stats,
            optimizer_state=self.state.opt_state,
        )

    def load_checkpoint(self, path: str | None = None, step: int | None = None) -> None:
        """Load a training checkpoint.

        Args:
            path: Path to checkpoint file.
            step: Optional specific step to load.
        """
        if self.checkpoint_manager is None:
            if path is not None:
                # Direct load from path even if no manager initialized
                from ndswin.training.checkpoint import CheckpointManager as CM

                manager = CM(directory="/tmp/ndswin_temp_load")
                checkpoint = manager.load(path=path)
            else:
                raise RuntimeError("Must provide path or have checkpoint_dir initialized")
        else:
            checkpoint = self.checkpoint_manager.load(path=path, step=step)

        if self.state is None:
            raise RuntimeError("Must initialize state before loading checkpoint")

        self.state = self.state.replace(
            params=checkpoint["params"],
            batch_stats=checkpoint.get("batch_stats"),
            opt_state=checkpoint.get("optimizer_state", self.state.opt_state),
        )
        self.step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)

    def add_callback(self, callback: Callable) -> None:
        """Add a training callback.

        Args:
            callback: Function called after each epoch with
                (trainer, epoch, metrics, history).
        """
        self.callbacks.append(callback)


def train_model(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any | None = None,
    config: TrainingConfig | None = None,
    num_epochs: int = 100,
    seed: int = 42,
    checkpoint_dir: str | None = None,
) -> tuple[TrainState, dict[str, list[float]]]:
    """Convenience function to train a model.

    Args:
        model: Flax model.
        train_loader: Training data loader.
        val_loader: Optional validation loader.
        config: Training configuration.
        num_epochs: Number of epochs.
        seed: Random seed.
        checkpoint_dir: Directory for checkpoints.

    Returns:
        Tuple of (final state, training history).
    """
    if config is None:
        config = TrainingConfig(epochs=num_epochs)

    trainer = Trainer(
        model=model,
        config=config,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
    )

    history = trainer.fit(train_loader, val_loader, num_epochs)

    return cast(TrainState, trainer.state), history
