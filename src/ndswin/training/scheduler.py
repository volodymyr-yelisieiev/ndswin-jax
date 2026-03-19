"""Learning rate schedules for NDSwin-JAX.

This module provides various learning rate scheduling strategies.
"""

from typing import Any, cast

import jax.numpy as jnp
import optax

from ndswin.types import LearningRate, LearningRateSchedule


class CosineAnnealingSchedule:
    """Cosine annealing learning rate schedule.

    Decreases the learning rate following a cosine curve.

    Example:
        >>> schedule = CosineAnnealingSchedule(
        ...     base_lr=1e-3,
        ...     total_steps=10000,
        ...     min_lr=1e-6,
        ... )
        >>> lr = schedule(step)
    """

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        min_lr: float = 0.0,
        warmup_steps: int = 0,
    ) -> None:
        """Initialize cosine annealing schedule.

        Args:
            base_lr: Base learning rate.
            total_steps: Total number of training steps.
            min_lr: Minimum learning rate.
            warmup_steps: Number of warmup steps.
        """
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> LearningRate:
        """Get learning rate for step.

        Args:
            step: Current training step.

        Returns:
            Learning rate.
        """
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * step / max(1, self.warmup_steps)

        # Cosine decay
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress_arr = jnp.clip(progress, 0.0, 1.0)

        cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * progress_arr))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

    def to_optax(self) -> LearningRateSchedule:
        """Convert to optax schedule.

        Returns:
            Optax learning rate schedule function.
        """
        schedules = []
        boundaries = []

        if self.warmup_steps > 0:
            warmup = optax.linear_schedule(
                init_value=0.0,
                end_value=self.base_lr,
                transition_steps=self.warmup_steps,
            )
            schedules.append(warmup)
            boundaries.append(self.warmup_steps)

        decay = optax.cosine_decay_schedule(
            init_value=self.base_lr,
            decay_steps=self.total_steps - self.warmup_steps,
            alpha=self.min_lr / self.base_lr if self.base_lr > 0 else 0.0,
        )
        schedules.append(decay)

        if len(schedules) > 1:
            return cast(LearningRateSchedule, optax.join_schedules(schedules, boundaries))
        return cast(LearningRateSchedule, schedules[0])


class WarmupSchedule:
    """Linear warmup schedule.

    Linearly increases learning rate from 0 to base_lr over warmup steps.
    """

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
    ) -> None:
        """Initialize warmup schedule.

        Args:
            base_lr: Target learning rate after warmup.
            warmup_steps: Number of warmup steps.
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> LearningRate:
        """Get learning rate for step.

        Args:
            step: Current training step.

        Returns:
            Learning rate.
        """
        if step >= self.warmup_steps:
            return self.base_lr
        return self.base_lr * step / max(1, self.warmup_steps)

    def to_optax(self) -> LearningRateSchedule:
        """Convert to optax schedule.

        Returns:
            Optax learning rate schedule function.
        """
        warmup = optax.linear_schedule(
            init_value=0.0,
            end_value=self.base_lr,
            transition_steps=self.warmup_steps,
        )
        constant = optax.constant_schedule(self.base_lr)

        return cast(
            LearningRateSchedule, optax.join_schedules([warmup, constant], [self.warmup_steps])
        )


class LinearSchedule:
    """Linear decay learning rate schedule.

    Linearly decreases learning rate from base_lr to min_lr.
    """

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        min_lr: float = 0.0,
        warmup_steps: int = 0,
    ) -> None:
        """Initialize linear schedule.

        Args:
            base_lr: Base learning rate.
            total_steps: Total number of training steps.
            min_lr: Minimum learning rate.
            warmup_steps: Number of warmup steps.
        """
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> LearningRate:
        """Get learning rate for step.

        Args:
            step: Current training step.

        Returns:
            Learning rate.
        """
        if step < self.warmup_steps:
            return self.base_lr * step / max(1, self.warmup_steps)

        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress_arr = jnp.clip(progress, 0.0, 1.0)

        return self.base_lr - (self.base_lr - self.min_lr) * progress_arr

    def to_optax(self) -> LearningRateSchedule:
        """Convert to optax schedule.

        Returns:
            Optax learning rate schedule function.
        """
        schedules = []
        boundaries = []

        if self.warmup_steps > 0:
            warmup = optax.linear_schedule(
                init_value=0.0,
                end_value=self.base_lr,
                transition_steps=self.warmup_steps,
            )
            schedules.append(warmup)
            boundaries.append(self.warmup_steps)

        decay = optax.linear_schedule(
            init_value=self.base_lr,
            end_value=self.min_lr,
            transition_steps=self.total_steps - self.warmup_steps,
        )
        schedules.append(decay)

        if len(schedules) > 1:
            return cast(LearningRateSchedule, optax.join_schedules(schedules, boundaries))
        return cast(LearningRateSchedule, schedules[0])


class StepSchedule:
    """Step decay learning rate schedule.

    Decreases learning rate by a factor at specified milestones.
    """

    def __init__(
        self,
        base_lr: float,
        milestones: tuple[int, ...],
        gamma: float = 0.1,
        warmup_steps: int = 0,
    ) -> None:
        """Initialize step schedule.

        Args:
            base_lr: Base learning rate.
            milestones: Steps at which to decay.
            gamma: Decay factor.
            warmup_steps: Number of warmup steps.
        """
        self.base_lr = base_lr
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> LearningRate:
        """Get learning rate for step.

        Args:
            step: Current training step.

        Returns:
            Learning rate.
        """
        if step < self.warmup_steps:
            return self.base_lr * step / max(1, self.warmup_steps)

        lr = self.base_lr
        for milestone in self.milestones:
            if step >= milestone:
                lr *= self.gamma
        return lr

    def to_optax(self) -> LearningRateSchedule:
        """Convert to optax schedule.

        Returns:
            Optax learning rate schedule function.
        """
        schedules = []
        boundaries = []

        if self.warmup_steps > 0:
            warmup = optax.linear_schedule(
                init_value=0.0,
                end_value=self.base_lr,
                transition_steps=self.warmup_steps,
            )
            schedules.append(warmup)
            boundaries.append(self.warmup_steps)

        # Create piecewise constant
        boundaries_and_scales = dict.fromkeys(self.milestones, self.gamma)
        step_schedule = optax.piecewise_constant_schedule(
            init_value=self.base_lr,
            boundaries_and_scales=boundaries_and_scales,
        )
        schedules.append(step_schedule)

        if len(schedules) > 1:
            return cast(LearningRateSchedule, optax.join_schedules(schedules, boundaries))
        return cast(LearningRateSchedule, schedules[0])


class OneCycleLR:
    """One-cycle learning rate schedule.

    Increases then decreases learning rate following a cosine cycle.
    """

    def __init__(
        self,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
    ) -> None:
        """Initialize one-cycle schedule.

        Args:
            max_lr: Maximum learning rate.
            total_steps: Total number of training steps.
            pct_start: Percentage of steps for increasing phase.
            div_factor: Factor to determine initial LR (max_lr / div_factor).
            final_div_factor: Factor to determine final LR.
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        self.step_up = int(total_steps * pct_start)
        self.step_down = total_steps - self.step_up

    def __call__(self, step: int) -> LearningRate:
        """Get learning rate for step.

        Args:
            step: Current training step.

        Returns:
            Learning rate.
        """
        if step < self.step_up:
            # Increasing phase
            progress = step / max(1, self.step_up)
            return (
                self.initial_lr
                + (self.max_lr - self.initial_lr) * (1 - jnp.cos(jnp.pi * progress)) / 2
            )
        else:
            # Decreasing phase
            progress = (step - self.step_up) / max(1, self.step_down)
            return (
                self.final_lr + (self.max_lr - self.final_lr) * (1 + jnp.cos(jnp.pi * progress)) / 2
            )

    def to_optax(self) -> LearningRateSchedule:
        """Convert to optax schedule.

        Returns:
            Optax learning rate schedule function.
        """
        # Use piecewise interpolation
        # Increasing phase
        up_schedule = optax.cosine_decay_schedule(
            init_value=self.initial_lr,
            decay_steps=self.step_up,
            alpha=self.max_lr / self.initial_lr if self.initial_lr > 0 else 1.0,
        )

        # Decreasing phase
        down_schedule = optax.cosine_decay_schedule(
            init_value=self.max_lr,
            decay_steps=self.step_down,
            alpha=self.final_lr / self.max_lr if self.max_lr > 0 else 0.0,
        )

        return cast(
            LearningRateSchedule,
            optax.join_schedules(
                [up_schedule, down_schedule],
                [self.step_up],
            ),
        )


def create_schedule(
    name: str,
    base_lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr: float = 0.0,
    **kwargs: Any,
) -> LearningRateSchedule:
    """Create a learning rate schedule.

    Args:
        name: Schedule name ('cosine', 'linear', 'step', 'warmup', 'one_cycle').
        base_lr: Base learning rate.
        total_steps: Total number of training steps.
        warmup_steps: Number of warmup steps.
        min_lr: Minimum learning rate.
        **kwargs: Additional schedule-specific arguments.

    Returns:
        Learning rate schedule function.
    """
    name = name.lower()

    if name == "cosine":
        return cast(
            LearningRateSchedule,
            CosineAnnealingSchedule(
                base_lr=base_lr,
                total_steps=total_steps,
                min_lr=min_lr,
                warmup_steps=warmup_steps,
            ),
        )
    elif name == "linear":
        return cast(
            LearningRateSchedule,
            LinearSchedule(
                base_lr=base_lr,
                total_steps=total_steps,
                min_lr=min_lr,
                warmup_steps=warmup_steps,
            ),
        )
    elif name == "step":
        milestones = kwargs.get("milestones", (total_steps // 3, 2 * total_steps // 3))
        gamma = kwargs.get("gamma", 0.1)
        return cast(
            LearningRateSchedule,
            StepSchedule(
                base_lr=base_lr,
                milestones=milestones,
                gamma=gamma,
                warmup_steps=warmup_steps,
            ),
        )
    elif name == "warmup":
        return cast(
            LearningRateSchedule,
            WarmupSchedule(
                base_lr=base_lr,
                warmup_steps=warmup_steps,
            ),
        )
    elif name == "one_cycle":
        return cast(
            LearningRateSchedule,
            OneCycleLR(
                max_lr=base_lr,
                total_steps=total_steps,
                pct_start=kwargs.get("pct_start", 0.3),
                div_factor=kwargs.get("div_factor", 25.0),
                final_div_factor=kwargs.get("final_div_factor", 1e4),
            ),
        )
    elif name == "constant":
        return cast(LearningRateSchedule, lambda _step: base_lr)
    else:
        raise ValueError(f"Unknown schedule: {name}")
