"""Learning rate scheduling utilities."""

from __future__ import annotations

import optax
from ml_collections import ConfigDict


def create_learning_rate_schedule(config: ConfigDict, steps_per_epoch: int) -> optax.Schedule:
    """Create learning rate schedule with warmup and cosine decay.
    
    Args:
        config: Configuration with learning rate parameters
        steps_per_epoch: Number of training steps per epoch
        
    Returns:
        Optax learning rate schedule
    """
    warmup_epochs = getattr(config, "warmup_epochs", 0)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = config.num_epochs * steps_per_epoch
    decay_steps = total_steps - warmup_steps
    
    # If warmup is too long or epochs too few, just use constant learning rate
    if warmup_steps >= total_steps or decay_steps <= 0:
        return config.learning_rate
    
    min_lr = getattr(config, "min_learning_rate", 0.0)
    
    schedules = []
    boundaries = []
    
    if warmup_steps > 0:
        # Linear warmup
        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=config.learning_rate,
            transition_steps=warmup_steps,
        )
        schedules.append(warmup_schedule)
        boundaries.append(warmup_steps)
    
    # Cosine decay after warmup
    cosine_schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=decay_steps,
        alpha=min_lr / config.learning_rate if config.learning_rate > 0 else 0.0,
    )
    schedules.append(cosine_schedule)
    
    if warmup_steps > 0:
        # Join warmup and decay schedules
        schedule = optax.join_schedules(schedules, boundaries)
    else:
        schedule = cosine_schedule
    
    return schedule


__all__ = ["create_learning_rate_schedule"]
