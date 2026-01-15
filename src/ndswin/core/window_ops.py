"""N-dimensional window operations for Swin Transformer.

This module implements generalized window partitioning and shifting operations
that work with arbitrary number of spatial dimensions (2D, 3D, 4D, and beyond).
"""

from functools import reduce
from operator import mul

import jax.numpy as jnp

from ndswin.types import Array


def get_window_grid_shape(
    spatial_shape: tuple[int, ...],
    window_size: tuple[int, ...],
) -> tuple[int, ...]:
    """Compute the grid shape of windows.

    Args:
        spatial_shape: Shape of spatial dimensions.
        window_size: Size of window in each dimension.

    Returns:
        Tuple of number of windows in each dimension.
    """
    return tuple(s // w for s, w in zip(spatial_shape, window_size))


def partition_windows(
    x: Array,
    window_size: tuple[int, ...],
) -> Array:
    """Partition input into non-overlapping windows.

    This function works with arbitrary number of spatial dimensions.
    The input is expected to have shape (B, *spatial_dims, C) where
    spatial_dims is a tuple of spatial sizes.

    Args:
        x: Input tensor of shape (B, *spatial_dims, C).
        window_size: Size of window in each spatial dimension.

    Returns:
        Tensor of shape (B * num_windows, *window_size, C) where
        num_windows = prod(spatial_dims[i] // window_size[i]).

    Raises:
        ValueError: If spatial dimensions are not divisible by window size.

    Example:
        >>> # 2D case
        >>> x = jnp.ones((2, 56, 56, 96))  # (B, H, W, C)
        >>> windows = partition_windows(x, (7, 7))
        >>> print(windows.shape)  # (2 * 64, 7, 7, 96) = (128, 7, 7, 96)

        >>> # 3D case
        >>> x = jnp.ones((2, 28, 28, 28, 96))  # (B, D, H, W, C)
        >>> windows = partition_windows(x, (7, 7, 7))
        >>> print(windows.shape)  # (2 * 64, 7, 7, 7, 96) = (128, 7, 7, 7, 96)
    """
    batch_size = x.shape[0]
    channels = x.shape[-1]
    spatial_shape = x.shape[1:-1]
    num_dims = len(spatial_shape)

    if len(window_size) != num_dims:
        raise ValueError(
            f"window_size must have {num_dims} elements to match spatial dims, "
            f"got {len(window_size)}"
        )

    # Validate divisibility
    for i, (s, w) in enumerate(zip(spatial_shape, window_size)):
        if s % w != 0:
            raise ValueError(f"Spatial dimension {i} ({s}) must be divisible by window_size ({w})")

    # Compute grid shape (number of windows in each dimension)
    grid_shape = get_window_grid_shape(spatial_shape, window_size)
    num_windows = reduce(mul, grid_shape, 1)

    # Reshape to separate windows
    # From: (B, S0, S1, ..., Sn, C)
    # To: (B, G0, W0, G1, W1, ..., Gn, Wn, C)
    new_shape = [batch_size]
    for g, w in zip(grid_shape, window_size):
        new_shape.extend([g, w])
    new_shape.append(channels)

    x = x.reshape(new_shape)

    # Permute to group window dimensions together
    # From: (B, G0, W0, G1, W1, ..., Gn, Wn, C)
    # To: (B, G0, G1, ..., Gn, W0, W1, ..., Wn, C)
    perm = [0]
    # Grid dimensions
    perm.extend(range(1, 2 * num_dims + 1, 2))
    # Window dimensions
    perm.extend(range(2, 2 * num_dims + 2, 2))
    # Channel dimension
    perm.append(2 * num_dims + 1)

    x = jnp.transpose(x, perm)

    # Flatten batch and grid dimensions
    # From: (B, G0, G1, ..., Gn, W0, W1, ..., Wn, C)
    # To: (B * num_windows, W0, W1, ..., Wn, C)
    output_shape = [batch_size * num_windows] + list(window_size) + [channels]
    x = x.reshape(output_shape)

    return x


def reverse_partition_windows(
    windows: Array,
    spatial_shape: tuple[int, ...],
    window_size: tuple[int, ...],
) -> Array:
    """Reverse the window partitioning operation.

    Args:
        windows: Windowed tensor of shape (B * num_windows, *window_size, C).
        spatial_shape: Original spatial dimensions.
        window_size: Size of window in each dimension.

    Returns:
        Tensor of shape (B, *spatial_shape, C).

    Example:
        >>> # 2D case
        >>> windows = jnp.ones((128, 7, 7, 96))
        >>> x = reverse_partition_windows(windows, (56, 56), (7, 7))
        >>> print(x.shape)  # (2, 56, 56, 96)
    """
    channels = windows.shape[-1]
    num_dims = len(window_size)

    # Compute grid shape
    grid_shape = get_window_grid_shape(spatial_shape, window_size)
    num_windows = reduce(mul, grid_shape, 1)

    # Infer batch size
    total_windows = windows.shape[0]
    batch_size = total_windows // num_windows

    # Reshape to separate batch and grid dimensions
    # From: (B * num_windows, W0, W1, ..., Wn, C)
    # To: (B, G0, G1, ..., Gn, W0, W1, ..., Wn, C)
    new_shape = [batch_size] + list(grid_shape) + list(window_size) + [channels]
    x = windows.reshape(new_shape)

    # Permute to interleave grid and window dimensions
    # From: (B, G0, G1, ..., Gn, W0, W1, ..., Wn, C)
    # To: (B, G0, W0, G1, W1, ..., Gn, Wn, C)
    perm = [0]
    for i in range(num_dims):
        perm.append(1 + i)  # Gi
        perm.append(1 + num_dims + i)  # Wi
    perm.append(1 + 2 * num_dims)  # C

    x = jnp.transpose(x, perm)

    # Merge grid and window dimensions
    # From: (B, G0, W0, G1, W1, ..., Gn, Wn, C)
    # To: (B, S0, S1, ..., Sn, C)
    output_shape = [batch_size] + list(spatial_shape) + [channels]
    x = x.reshape(output_shape)

    return x


def cyclic_shift(
    x: Array,
    shift_size: tuple[int, ...],
) -> Array:
    """Apply cyclic shift to spatial dimensions.

    This rolls the tensor along each spatial dimension by the specified
    shift amount. Used for shifted window attention in Swin Transformer.

    Args:
        x: Input tensor of shape (B, *spatial_dims, C).
        shift_size: Shift amount for each spatial dimension.
            Negative values shift towards the start.

    Returns:
        Shifted tensor of same shape.

    Example:
        >>> x = jnp.ones((2, 56, 56, 96))
        >>> shifted = cyclic_shift(x, (-3, -3))
    """
    # Shift each spatial dimension
    for i, shift in enumerate(shift_size):
        if shift != 0:
            x = jnp.roll(x, shift=-shift, axis=i + 1)

    return x


def reverse_cyclic_shift(
    x: Array,
    shift_size: tuple[int, ...],
) -> Array:
    """Reverse cyclic shift operation.

    This is the inverse of cyclic_shift.

    Args:
        x: Shifted tensor of shape (B, *spatial_dims, C).
        shift_size: Original shift amounts.

    Returns:
        Unshifted tensor.
    """
    # Reverse shift is just negating the shift amounts
    return cyclic_shift(x, tuple(-s for s in shift_size))


def create_attention_mask(
    spatial_shape: tuple[int, ...],
    window_size: tuple[int, ...],
    shift_size: tuple[int, ...],
) -> Array:
    """Create attention mask for shifted window attention.

    When using shifted windows, tokens from different original windows
    end up in the same shifted window. This mask prevents attention
    between tokens that were in different original windows.

    Args:
        spatial_shape: Shape of spatial dimensions.
        window_size: Size of attention window.
        shift_size: Amount of shift applied.

    Returns:
        Attention mask of shape (num_windows, window_area, window_area)
        where True indicates positions that should attend to each other.

    Example:
        >>> mask = create_attention_mask((56, 56), (7, 7), (3, 3))
        >>> print(mask.shape)  # (64, 49, 49)
    """
    num_dims = len(spatial_shape)
    window_area = reduce(mul, window_size, 1)
    grid_shape = get_window_grid_shape(spatial_shape, window_size)
    num_windows = reduce(mul, grid_shape, 1)

    # Check if any shift is non-zero
    if all(s == 0 for s in shift_size):
        # No shift - all positions can attend to each other
        return jnp.ones((num_windows, window_area, window_area), dtype=jnp.bool_)

    # Create window indices
    # Each position gets an ID based on which original window it belongs to
    # after the shift

    # Start with zeros
    window_ids = jnp.zeros(spatial_shape, dtype=jnp.int32)

    # For each dimension, mark regions that wrap around due to shift
    id_increment = 1

    # We need to track which "slice" each position belongs to
    # A slice is defined by whether it wraps around in each dimension

    for dim in range(num_dims):
        s = shift_size[dim]
        if s == 0:
            continue

        size = spatial_shape[dim]
        # Create indices for this dimension
        indices = jnp.arange(size)

        # Positions that wrap around vs those that don't
        # After shift of s, positions [0, s) wrap to the end
        # and positions [s, size) stay in order
        wrap_mask = indices < s

        # Create slice for broadcasting
        shape = [1] * num_dims
        shape[dim] = size
        wrap_mask = wrap_mask.reshape(shape)

        # Add to window IDs
        window_ids = window_ids + wrap_mask.astype(jnp.int32) * id_increment
        id_increment *= 2

    # Now partition window_ids the same way we partition the data
    # After shift
    window_ids_shifted = cyclic_shift(window_ids[None, ..., None].astype(jnp.float32), shift_size)[
        0, ..., 0
    ].astype(jnp.int32)

    # Add channel dimension for partitioning
    window_ids_expanded = window_ids_shifted[..., None]

    # Partition into windows
    # Need to match the partition_windows function signature
    windows = partition_windows(
        window_ids_expanded[None].astype(jnp.float32), window_size
    )  # (num_windows, *window_size, 1)

    # Flatten window dimensions
    windows = windows.reshape(num_windows, window_area)  # (num_windows, window_area)

    # Create mask: positions with same ID can attend to each other
    # (num_windows, window_area, 1) == (num_windows, 1, window_area)
    mask = windows[:, :, None] == windows[:, None, :]

    return mask


def create_shifted_window_mask(
    spatial_shape: tuple[int, ...],
    window_size: tuple[int, ...],
    shift_size: tuple[int, ...],
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Create attention mask as additive bias for shifted windows.

    This creates a mask that can be added to attention logits,
    with 0 for valid positions and -inf for invalid positions.

    Args:
        spatial_shape: Shape of spatial dimensions.
        window_size: Size of attention window.
        shift_size: Amount of shift applied.
        dtype: Output dtype.

    Returns:
        Mask of shape (1, num_windows, 1, window_area, window_area)
        suitable for adding to attention logits.
    """
    bool_mask = create_attention_mask(spatial_shape, window_size, shift_size)

    # Convert to additive mask (0 for attend, -inf for don't attend)
    mask = jnp.where(bool_mask, 0.0, -jnp.inf).astype(dtype)

    # Reshape for broadcasting with attention
    # (num_windows, window_area, window_area) ->
    # (1, num_windows, 1, window_area, window_area)
    mask = mask[None, :, None, :, :]

    return mask


def compute_shift_size(
    window_size: tuple[int, ...],
    shift_ratio: float = 0.5,
) -> tuple[int, ...]:
    """Compute shift size as a fraction of window size.

    Standard Swin Transformer uses shift_size = window_size // 2.

    Args:
        window_size: Size of attention window.
        shift_ratio: Fraction of window size to shift.

    Returns:
        Shift size for each dimension.
    """
    return tuple(int(w * shift_ratio) for w in window_size)


def get_valid_spatial_shapes(
    window_size: tuple[int, ...],
    min_windows: int = 1,
    max_windows: int = 16,
) -> list:
    """Get valid spatial shapes for given window size.

    This is useful for testing and debugging.

    Args:
        window_size: Size of attention window.
        min_windows: Minimum number of windows per dimension.
        max_windows: Maximum number of windows per dimension.

    Returns:
        List of valid spatial shapes.
    """
    shapes = []
    for n in range(min_windows, max_windows + 1):
        shape = tuple(w * n for w in window_size)
        shapes.append(shape)
    return shapes


def pad_for_window(
    x: Array,
    window_size: tuple[int, ...],
) -> tuple[Array, tuple[int, ...]]:
    """Pad input to be divisible by window size.

    Args:
        x: Input tensor of shape (B, *spatial_dims, C).
        window_size: Size of attention window.

    Returns:
        Tuple of (padded tensor, original spatial shape).
    """
    spatial_shape = x.shape[1:-1]

    # Check if already divisible
    needs_padding = False
    pad_amounts = []
    for s, w in zip(spatial_shape, window_size):
        remainder = s % w
        if remainder != 0:
            needs_padding = True
            pad_amounts.append(w - remainder)
        else:
            pad_amounts.append(0)

    if not needs_padding:
        return x, spatial_shape

    # Build pad specification
    # jnp.pad wants [(before, after), ...] for each dimension
    pad_width = [(0, 0)]  # Batch dim
    for p in pad_amounts:
        pad_width.append((0, p))  # Spatial dims
    pad_width.append((0, 0))  # Channel dim

    x_padded = jnp.pad(x, pad_width, mode="constant", constant_values=0)

    return x_padded, spatial_shape


def unpad_from_window(
    x: Array,
    original_shape: tuple[int, ...],
) -> Array:
    """Remove padding added by pad_for_window.

    Args:
        x: Padded tensor.
        original_shape: Original spatial shape before padding.

    Returns:
        Unpadded tensor.
    """
    slices = [slice(None)]  # Batch dim
    for s in original_shape:
        slices.append(slice(0, s))  # Spatial dims
    slices.append(slice(None))  # Channel dim

    return x[tuple(slices)]
