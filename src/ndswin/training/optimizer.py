"""Optimizer utilities for NDSwin-JAX.

This module provides optimizer creation and configuration.
"""

from typing import Any, cast

import optax

from ndswin.config import TrainingConfig
from ndswin.types import LearningRateSchedule


def create_learning_rate_schedule(
    base_lr: float,
    num_steps: int,
    warmup_steps: int = 0,
    schedule_type: str = "cosine",
    min_lr: float = 1e-6,
) -> LearningRateSchedule:
    """Create a learning rate schedule."""
    schedules = []
    boundaries = []

    # Warmup phase
    if warmup_steps > 0:
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_lr,
            transition_steps=warmup_steps,
        )
        schedules.append(warmup_fn)
        boundaries.append(warmup_steps)

    # Main schedule
    remaining_steps = max(1, num_steps - warmup_steps)

    if schedule_type == "cosine":
        main_fn = optax.cosine_decay_schedule(
            init_value=base_lr,
            decay_steps=remaining_steps,
            alpha=min_lr / base_lr if base_lr > 0 else 0.0,
        )
    elif schedule_type == "linear":
        main_fn = optax.linear_schedule(
            init_value=base_lr,
            end_value=min_lr,
            transition_steps=remaining_steps,
        )
    elif schedule_type == "constant":
        main_fn = optax.constant_schedule(base_lr)
    elif schedule_type == "step":
        step1 = remaining_steps // 3
        step2 = 2 * remaining_steps // 3
        main_fn = optax.piecewise_constant_schedule(
            init_value=base_lr,
            boundaries_and_scales={
                step1: 0.1,
                step2: 0.1,
            },
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    schedules.append(main_fn)

    if len(schedules) > 1:
        return cast(LearningRateSchedule, optax.join_schedules(schedules, boundaries))
    return cast(LearningRateSchedule, schedules[0])


def create_optimizer(
    config: TrainingConfig,
    num_steps: int,
    num_train_samples: int | None = None,
) -> optax.GradientTransformation:
    """Create optimizer from training configuration."""
    steps_per_epoch = (
        (num_train_samples // config.batch_size)
        if num_train_samples and config.batch_size > 0
        else 1
    )
    warmup_steps = config.warmup_epochs * steps_per_epoch
    if config.warmup_steps > 0:
        warmup_steps = config.warmup_steps

    schedule = create_learning_rate_schedule(
        base_lr=config.learning_rate,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        schedule_type=config.lr_schedule,
        min_lr=config.min_learning_rate,
    )

    optimizer_name = config.optimizer.lower()
    if optimizer_name == "adamw":
        base_opt = optax.adamw(
            learning_rate=schedule,
            b1=config.adam_beta1,
            b2=config.adam_beta2,
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
    elif optimizer_name == "adam":
        base_opt = optax.adam(
            learning_rate=schedule,
            b1=config.adam_beta1,
            b2=config.adam_beta2,
            eps=config.adam_epsilon,
        )
    elif optimizer_name == "sgd":
        base_opt = optax.sgd(
            learning_rate=schedule,
            momentum=config.momentum,
            nesterov=config.nesterov,
        )
    elif optimizer_name == "lamb":
        base_opt = optax.lamb(
            learning_rate=schedule,
            b1=config.adam_beta1,
            b2=config.adam_beta2,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    transforms = []
    if config.max_grad_norm > 0:
        transforms.append(optax.clip_by_global_norm(config.max_grad_norm))
    transforms.append(base_opt)

    return optax.chain(*transforms)


def get_current_learning_rate(
    state: Any,  # TrainState
    step: int,
) -> float:
    """Get current learning rate from state."""
    # Since we use optax schedules, we can just call the schedule if we had it,
    # but TrainState doesn't store the schedule.
    # Usually we can look at opt_state if it's a simple scale_by_schedule.
    return 0.0  # Placeholder as it depends on having the schedule function handy.
