"""Utility functions for NDSwin-JAX core components.

This module provides helper functions for shape manipulation,
indexing, and other utilities used across core components.
"""

from collections.abc import Callable
from functools import reduce
from operator import mul
from typing import Any

import jax
import jax.numpy as jnp

from ndswin.types import Array, PRNGKey


def to_ntuple(x: int | tuple[int, ...], n: int) -> tuple[int, ...]:
    """Convert a value to an n-tuple.

    If x is already a tuple/list of length n, it is returned as-is.
    If x is an integer, it is repeated n times.

    Args:
        x: Value to convert.
        n: Target length.

    Returns:
        Tuple of length n.

    Raises:
        ValueError: If x is a sequence but has wrong length.
    """
    if isinstance(x, (tuple, list)):
        if len(x) != n:
            raise ValueError(f"Expected tuple of length {n}, got {len(x)}")
        return tuple(x)
    return tuple([x] * n)


def compute_num_patches(
    input_size: tuple[int, ...],
    patch_size: tuple[int, ...],
) -> int:
    """Compute the total number of patches.

    Args:
        input_size: Spatial dimensions of input.
        patch_size: Size of each patch.

    Returns:
        Total number of patches.

    Raises:
        ValueError: If input_size is not divisible by patch_size.
    """
    if len(input_size) != len(patch_size):
        raise ValueError(
            f"input_size and patch_size must have same length, "
            f"got {len(input_size)} and {len(patch_size)}"
        )

    patches_per_dim = []
    for i, (inp, pat) in enumerate(zip(input_size, patch_size)):
        if inp % pat != 0:
            raise ValueError(
                f"Dimension {i}: input_size ({inp}) must be divisible by patch_size ({pat})"
            )
        patches_per_dim.append(inp // pat)

    return reduce(mul, patches_per_dim, 1)


def get_grid_shape(
    input_size: tuple[int, ...],
    patch_size: tuple[int, ...],
) -> tuple[int, ...]:
    """Get the grid shape after patching.

    Args:
        input_size: Spatial dimensions of input.
        patch_size: Size of each patch.

    Returns:
        Tuple of grid dimensions.
    """
    return tuple(i // p for i, p in zip(input_size, patch_size))


def get_relative_position_index(
    window_size: tuple[int, ...],
) -> Array:
    """Generate relative position index for n-dimensional window.

    This creates an index tensor that maps each pair of positions in a window
    to a unique index based on their relative position.

    Args:
        window_size: Size of window in each dimension.

    Returns:
        Index tensor of shape (*window_size, *window_size).
    """
    num_dims = len(window_size)

    # Create coordinate grids for each dimension
    coords = [jnp.arange(ws) for ws in window_size]
    coord_grids = jnp.meshgrid(*coords, indexing="ij")
    coords_flatten = jnp.stack(
        [c.flatten() for c in coord_grids], axis=0
    )  # (num_dims, num_positions)

    # Compute relative coordinates
    # (num_dims, num_positions, num_positions)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]

    # Shift to start from 0
    for i in range(num_dims):
        relative_coords = relative_coords.at[i].add(window_size[i] - 1)

    # Multiply by strides to get unique indices
    strides: list[int] = []
    stride = 1
    for i in range(num_dims - 1, -1, -1):
        strides.insert(0, stride)
        stride *= 2 * window_size[i] - 1

    # Compute final index
    relative_position_index = jnp.zeros(
        (relative_coords.shape[1], relative_coords.shape[2]),
        dtype=jnp.int32,
    )
    for i in range(num_dims):
        relative_position_index = relative_position_index + relative_coords[i] * strides[i]

    return relative_position_index


def get_relative_position_bias_table_size(window_size: tuple[int, ...]) -> int:
    """Get the size of relative position bias table.

    Args:
        window_size: Size of window in each dimension.

    Returns:
        Size of the bias table.
    """
    size = 1
    for ws in window_size:
        size *= 2 * ws - 1
    return size


def pad_to_multiple(
    x: Array,
    multiples: tuple[int, ...],
    spatial_dims: tuple[int, ...] | None = None,
) -> tuple[Array, tuple[int, ...]]:
    """Pad array so spatial dimensions are multiples of given values.

    Args:
        x: Input array.
        multiples: Values to pad to multiples of.
        spatial_dims: Indices of spatial dimensions. If None, assumes
            last len(multiples) dimensions are spatial.

    Returns:
        Tuple of (padded array, original spatial sizes).
    """
    ndim = x.ndim
    num_spatial = len(multiples)

    if spatial_dims is None:
        spatial_dims = tuple(range(ndim - num_spatial, ndim))

    original_sizes = tuple(x.shape[d] for d in spatial_dims)

    # Calculate padding
    pad_sizes = []
    for i, (size, mult) in enumerate(zip(original_sizes, multiples)):
        remainder = size % mult
        pad = (mult - remainder) % mult
        pad_sizes.append(pad)

    # Build padding specification
    if any(p > 0 for p in pad_sizes):
        pad_width = [(0, 0)] * ndim
        for dim, pad in zip(spatial_dims, pad_sizes):
            pad_width[dim] = (0, pad)
        x = jnp.pad(x, pad_width, mode="constant", constant_values=0)

    return x, original_sizes


def unpad(
    x: Array,
    original_sizes: tuple[int, ...],
    spatial_dims: tuple[int, ...] | None = None,
) -> Array:
    """Remove padding from array.

    Args:
        x: Padded array.
        original_sizes: Original sizes of spatial dimensions.
        spatial_dims: Indices of spatial dimensions.

    Returns:
        Unpadded array.
    """
    ndim = x.ndim
    num_spatial = len(original_sizes)

    if spatial_dims is None:
        spatial_dims = tuple(range(ndim - num_spatial, ndim))

    slices = [slice(None)] * ndim
    for dim, size in zip(spatial_dims, original_sizes):
        slices[dim] = slice(0, size)

    return x[tuple(slices)]


def split_along_dim(x: Array, dim: int, num_splits: int) -> list[Array]:
    """Split array along a dimension.

    Args:
        x: Input array.
        dim: Dimension to split along.
        num_splits: Number of splits.

    Returns:
        List of split arrays.
    """
    return jnp.split(x, num_splits, axis=dim)


def reshape_for_broadcast(
    x: Array,
    target_ndim: int,
    dims: tuple[int, ...],
) -> Array:
    """Reshape array for broadcasting.

    Args:
        x: Input array.
        target_ndim: Target number of dimensions.
        dims: Dimensions where x should be placed.

    Returns:
        Reshaped array with shape suitable for broadcasting.
    """
    shape = [1] * target_ndim
    for i, d in enumerate(dims):
        shape[d] = x.shape[i]
    return x.reshape(shape)


def flatten_spatial(x: Array, num_spatial_dims: int) -> Array:
    """Flatten spatial dimensions into a single sequence dimension.

    Args:
        x: Input array of shape (B, *spatial_dims, C).
        num_spatial_dims: Number of spatial dimensions.

    Returns:
        Array of shape (B, num_patches, C).
    """
    batch_size = x.shape[0]
    channels = x.shape[-1]
    return x.reshape(batch_size, -1, channels)


def unflatten_spatial(
    x: Array,
    spatial_shape: tuple[int, ...],
) -> Array:
    """Unflatten sequence dimension back to spatial dimensions.

    Args:
        x: Input array of shape (B, num_patches, C).
        spatial_shape: Target spatial shape.

    Returns:
        Array of shape (B, *spatial_shape, C).
    """
    batch_size = x.shape[0]
    channels = x.shape[-1]
    return x.reshape(batch_size, *spatial_shape, channels)


def einsum_attention(
    query: Array,
    key: Array,
    value: Array,
    scale: float,
    mask: Array | None = None,
    dropout_fn: Callable[[Array], Array] | None = None,
) -> Array:
    """Compute scaled dot-product attention using einsum.

    Args:
        query: Query tensor of shape (..., num_heads, seq_len, head_dim).
        key: Key tensor of shape (..., num_heads, seq_len, head_dim).
        value: Value tensor of shape (..., num_heads, seq_len, head_dim).
        scale: Scaling factor (typically 1/sqrt(head_dim)).
        mask: Optional attention mask.
        dropout_fn: Optional dropout function.

    Returns:
        Attention output of shape (..., num_heads, seq_len, head_dim).
    """
    # Compute attention scores
    attn = jnp.einsum("...hqd,...hkd->...hqk", query, key) * scale

    # Apply mask
    if mask is not None:
        attn = jnp.where(mask, attn, jnp.finfo(attn.dtype).min)

    # Softmax
    attn = jax.nn.softmax(attn, axis=-1)

    # Apply dropout
    if dropout_fn is not None:
        attn = dropout_fn(attn)

    # Compute output
    output = jnp.einsum("...hqk,...hkd->...hqd", attn, value)

    return output


def compute_window_count(
    spatial_size: tuple[int, ...],
    window_size: tuple[int, ...],
) -> int:
    """Compute total number of windows.

    Args:
        spatial_size: Size of spatial dimensions.
        window_size: Size of each window.

    Returns:
        Total number of windows.
    """
    count = 1
    for s, w in zip(spatial_size, window_size):
        count *= (s + w - 1) // w  # Ceiling division
    return count


def compute_tokens_per_window(window_size: tuple[int, ...]) -> int:
    """Compute number of tokens per window.

    Args:
        window_size: Size of window in each dimension.

    Returns:
        Number of tokens per window.
    """
    return reduce(mul, window_size, 1)


def trunc_normal_init(key: PRNGKey, shape: tuple[int, ...], std: float = 0.02) -> Array:
    """Truncated normal initialization.

    Args:
        key: JAX PRNG key.
        shape: Output shape.
        std: Standard deviation.

    Returns:
        Initialized array.
    """
    return jax.random.truncated_normal(key, -2, 2, shape) * std


def zeros_init(key: PRNGKey, shape: tuple[int, ...], dtype: Any = jnp.float32) -> Array:
    """Zero initialization.

    Args:
        key: JAX PRNG key (unused).
        shape: Output shape.
        dtype: Data type.

    Returns:
        Zero-initialized array.
    """
    return jnp.zeros(shape, dtype=dtype)


def ones_init(key: PRNGKey, shape: tuple[int, ...], dtype: Any = jnp.float32) -> Array:
    """Ones initialization.

    Args:
        key: JAX PRNG key (unused).
        shape: Output shape.
        dtype: Data type.

    Returns:
        Ones-initialized array.
    """
    return jnp.ones(shape, dtype=dtype)
