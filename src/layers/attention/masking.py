"""Window partitioning and masking utilities for Swin attention."""

from __future__ import annotations

from math import ceil
from typing import Sequence

import jax.numpy as jnp
import numpy as np


def window_partition(x: jnp.ndarray, window_size: Sequence[int]) -> jnp.ndarray:
    """Reshape tensors into non-overlapping local windows.
    
    Args:
        x: Input tensor of shape (batch, *spatial_dims, channels)
        window_size: Size of each window for each spatial dimension
        
    Returns:
        Windows tensor of shape (batch * num_windows, window_tokens, channels)
        where window_tokens = prod(window_size)
        
    Raises:
        ValueError: If spatial dimensions are not divisible by window size
    """
    space = len(window_size)
    batch = x.shape[0]
    spatial = x.shape[1:-1]
    channels = x.shape[-1]

    for dim, win in zip(spatial, window_size):
        if dim % win != 0:
            raise ValueError("Spatial dimensions must be divisible by window size.")

    counts = [dim // win for dim, win in zip(spatial, window_size)]
    reshape_dims = []
    for count, win in zip(counts, window_size):
        reshape_dims.extend([count, win])

    x = x.reshape(batch, *reshape_dims, channels)

    count_axes = [1 + 2 * i for i in range(space)]
    window_axes = [1 + 2 * i + 1 for i in range(space)]
    permute_axes = [0] + count_axes + window_axes + [1 + 2 * space]
    x = x.transpose(permute_axes)

    windows = x.reshape(batch * int(np.prod(np.array(counts))), int(np.prod(np.array(window_size))), channels)
    return windows


def window_reverse(windows: jnp.ndarray, window_size: Sequence[int], dims: Sequence[int]) -> jnp.ndarray:
    """Reverse :func:`window_partition` to recover the original layout.
    
    Args:
        windows: Windows tensor from window_partition
        window_size: Size of each window for each spatial dimension
        dims: Original tensor dimensions (batch, *spatial_dims)
        
    Returns:
        Reconstructed tensor of shape (batch, *spatial_dims, channels)
    """
    space = len(window_size)
    batch = dims[0]
    spatial = dims[1:]
    channels = windows.shape[-1]

    counts = [size // win for size, win in zip(spatial, window_size)]
    windows = windows.reshape(batch, *counts, *window_size, channels)

    count_axes = [1 + i for i in range(space)]
    window_axes = [1 + space + i for i in range(space)]
    permute_axes = [0]
    for idx in range(space):
        permute_axes.extend([count_axes[idx], window_axes[idx]])
    permute_axes.append(1 + 2 * space)

    x = windows.transpose(permute_axes)
    x = x.reshape(batch, *spatial, channels)
    return x


def compute_mask(dims: Sequence[int], window_size: Sequence[int], shift_size: Sequence[int]) -> jnp.ndarray:
    """Compute attention mask for shifted windows.
    
    This mask ensures that attention is only computed within the same window region
    after applying cyclic shifts, preventing information leakage across window boundaries.
    
    Args:
        dims: Spatial dimensions after padding
        window_size: Size of each window
        shift_size: Shift amount for each dimension
        
    Returns:
        Attention mask of shape (num_windows, window_tokens, window_tokens)
        where masked positions have value -1e9 (effectively -inf for softmax)
    """
    space = len(window_size)
    axis_regions = []
    for dim, win, shift in zip(dims, window_size, shift_size):
        region = jnp.zeros((dim,), dtype=jnp.int32)
        if shift > 0:
            region = region.at[(dim - win) : (dim - shift)].set(1)
            region = region.at[(dim - shift) :].set(2)
        axis_regions.append(region)

    multipliers = [3 ** (space - axis - 1) for axis in range(space)]
    region_grid = jnp.zeros((1, *dims, 1), dtype=jnp.int32)
    for axis, (region, multiplier) in enumerate(zip(axis_regions, multipliers)):
        shape = [1] * space
        shape[axis] = region.shape[0]
        region_grid = region_grid + region.reshape((1, *shape, 1)) * multiplier

    mask_windows = window_partition(region_grid.astype(jnp.float32), window_size)
    mask_windows = mask_windows[..., 0]
    mask = mask_windows[:, None, :] - mask_windows[:, :, None]
    mask = jnp.where(mask == 0, 0.0, -1e9)
    return mask


__all__ = ["window_partition", "window_reverse", "compute_mask"]
