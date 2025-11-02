"""Relative position bias utilities for Swin attention."""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp


def relative_coords_table(window_size: Sequence[int]) -> jnp.ndarray:
    """Generate continuous relative position bias table.
    
    Uses log-spaced coordinates as described in Swin Transformer V2.
    The coordinates are normalized and transformed using log-spacing for
    better extrapolation to different window sizes.
    
    Args:
        window_size: Size of the attention window for each dimension
        
    Returns:
        Relative coordinates table of shape (num_relative_positions, num_dims)
        where num_relative_positions = prod(2*w-1 for w in window_size)
    """
    coords = [jnp.arange(-(w - 1), w, dtype=jnp.float32) for w in window_size]
    table = jnp.stack(jnp.meshgrid(*coords, indexing="ij"), axis=-1)
    
    # Normalize and apply log-spacing
    for axis, w in enumerate(window_size):
        denom = max(w - 1, 1)
        table = table.at[..., axis].set(table[..., axis] / denom)
    
    # Apply log-spacing transformation
    table = 8.0 * table
    table = jnp.sign(table) * jnp.log2(jnp.abs(table) + 1.0) / jnp.log2(8.0)
    
    return table.reshape((-1, len(window_size)))


def relative_position_index(window_size: Sequence[int]) -> jnp.ndarray:
    """Generate relative position index for gathering bias values.
    
    Creates an index tensor that maps each pair of positions in a window
    to their corresponding entry in the relative position bias table.
    
    Args:
        window_size: Size of the attention window for each dimension
        
    Returns:
        Index tensor of shape (window_tokens, window_tokens)
        where window_tokens = prod(window_size)
    """
    coords = jnp.stack(
        jnp.meshgrid(*[jnp.arange(w, dtype=jnp.int32) for w in window_size], indexing="ij"),
        axis=0,
    )
    coords_flat = coords.reshape(coords.shape[0], -1)
    relative = coords_flat[:, :, None] - coords_flat[:, None, :]

    # Compute strides for flattening multi-dimensional indices
    import numpy as np
    scales = []
    for axis in range(len(window_size)):
        tail = window_size[axis + 1 :]
        if tail:
            scale = int(np.prod([(2 * w - 1) for w in tail]))
        else:
            scale = 1
        scales.append(scale)

    result = jnp.zeros(relative.shape[1:], dtype=jnp.int32)
    for axis, scale in enumerate(scales):
        result = result + (relative[axis] + window_size[axis] - 1) * scale
    return result


__all__ = ["relative_coords_table", "relative_position_index"]
