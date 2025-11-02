"""Training state initialization and management."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from ml_collections import ConfigDict

from src.models import NDSwinClassifier
from .lr_schedule import create_learning_rate_schedule


def create_model(config: ConfigDict) -> NDSwinClassifier:
    """Instantiate ``NDSwinClassifier`` from the ``config`` block.
    
    Args:
        config: Configuration dictionary containing model parameters
        
    Returns:
        Initialized NDSwinClassifier model
    """
    model_cfg = config.model
    return NDSwinClassifier(
        dim=int(model_cfg.dim),
        resolution=tuple(config.input_resolution),
        space=int(model_cfg.space),
        in_channels=int(model_cfg.in_channels),
        num_classes=int(model_cfg.num_classes),
        patch_size=tuple(model_cfg.patch_size),
        window_size=tuple(model_cfg.window_size),
        depth=tuple(model_cfg.depths),
        num_heads=tuple(model_cfg.num_heads),
        mlp_ratio=float(model_cfg.mlp_ratio),
        drop_path_rate=float(model_cfg.drop_path_rate),
        head_drop=float(model_cfg.head_drop),
        use_conv=bool(model_cfg.use_conv),
        use_abs_pos=bool(model_cfg.use_abs_pos),
    )


def create_train_state(
    model: NDSwinClassifier, config: ConfigDict, rng: jax.Array, steps_per_epoch: int | None = None
) -> train_state.TrainState:
    """Initialize model parameters and optimizer state.
    
    Args:
        model: Model instance to initialize
        config: Configuration with training parameters
        rng: Random number generator key
        steps_per_epoch: Number of steps per epoch (for lr scheduling)
        
    Returns:
        Initialized training state
    """
    input_shape = (
        config.batch_size,
        model.in_channels,
        *config.input_resolution,
    )
    dummy = jnp.zeros(input_shape, dtype=jnp.float32)
    params = model.init(rng, dummy, deterministic=True)["params"]

    # Create learning rate schedule if warmup_epochs is specified
    if steps_per_epoch and hasattr(config, "warmup_epochs") and config.warmup_epochs > 0:
        learning_rate = create_learning_rate_schedule(config, steps_per_epoch)
    else:
        learning_rate = config.learning_rate
    
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=config.weight_decay)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


__all__ = ["create_model", "create_train_state"]
