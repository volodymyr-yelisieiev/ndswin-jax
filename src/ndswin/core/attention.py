"""Attention mechanisms for NDSwin Transformer.

This module implements Window-based Multi-Head Self-Attention (W-MSA) and
Shifted Window Multi-Head Self-Attention (SW-MSA) for n-dimensional inputs.
"""

from functools import reduce
from operator import mul

import jax
import jax.numpy as jnp
from flax import linen as nn

from ndswin.core.utils import get_relative_position_bias_table_size, get_relative_position_index
from ndswin.core.window_ops import (
    create_shifted_window_mask,
    cyclic_shift,
    pad_for_window,
    partition_windows,
    reverse_partition_windows,
    unpad_from_window,
)
from ndswin.types import Array


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention layer.

    This implements the standard multi-head attention mechanism without
    windowing. It's used as a building block for window attention.

    Attributes:
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head. If None, computed from
            embed_dim // num_heads.
        qkv_bias: Whether to use bias in QKV projections.
        attn_drop_rate: Dropout rate for attention weights.
        proj_drop_rate: Dropout rate for output projection.
        use_rel_pos_bias: Whether to use relative position bias.
        dtype: Computation dtype.
    """

    num_heads: int
    head_dim: int | None = None
    qkv_bias: bool = True
    attn_drop_rate: float = 0.0
    proj_drop_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: Array,
        mask: Array | None = None,
        deterministic: bool = True,
    ) -> Array:
        """Apply multi-head attention.

        Args:
            x: Input tensor of shape (B, N, C) where N is sequence length.
            mask: Optional attention mask of shape (B, 1, N, N) or (1, 1, N, N).
                True values indicate valid attention positions.
            deterministic: Whether to apply dropout.

        Returns:
            Output tensor of shape (B, N, C).
        """
        batch_size, seq_len, embed_dim = x.shape

        head_dim = self.head_dim or embed_dim // self.num_heads
        inner_dim = self.num_heads * head_dim
        scale = head_dim**-0.5

        # QKV projection
        qkv = nn.Dense(
            inner_dim * 3,
            use_bias=self.qkv_bias,
            dtype=self.dtype,
            name="qkv",
        )(x)

        # Reshape and split into Q, K, V
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

        # Apply mask if provided
        if mask is not None:
            # Mask shape: (B, 1, N, N) or (1, 1, N, N)
            # attn shape: (B, num_heads, N, N)
            attn = jnp.where(mask, attn, jnp.finfo(self.dtype).min)

        # Softmax
        attn = jax.nn.softmax(attn, axis=-1)

        # Attention dropout
        if not deterministic and self.attn_drop_rate > 0:
            attn = nn.Dropout(rate=self.attn_drop_rate)(attn, deterministic=False)

        # Apply attention to values
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)

        # Reshape back
        out = jnp.transpose(out, (0, 2, 1, 3))  # (B, N, num_heads, head_dim)
        out = out.reshape(batch_size, seq_len, inner_dim)

        # Output projection
        out = nn.Dense(embed_dim, dtype=self.dtype, name="proj")(out)

        # Output dropout
        if not deterministic and self.proj_drop_rate > 0:
            out = nn.Dropout(rate=self.proj_drop_rate)(out, deterministic=False)

        return out


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention for n-dimensional inputs.

    This implements W-MSA (Window Multi-Head Self-Attention) and optionally
    SW-MSA (Shifted Window Multi-Head Self-Attention) for arbitrary dimensions.

    Attributes:
        num_heads: Number of attention heads.
        window_size: Size of attention window in each spatial dimension.
        qkv_bias: Whether to use bias in QKV projections.
        attn_drop_rate: Dropout rate for attention weights.
        proj_drop_rate: Dropout rate for output projection.
        use_rel_pos_bias: Whether to use relative position bias.
        dtype: Computation dtype.
    """

    num_heads: int
    window_size: tuple[int, ...]
    head_dim: int | None = None
    qkv_bias: bool = True
    attn_drop_rate: float = 0.0
    proj_drop_rate: float = 0.0
    use_rel_pos_bias: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        """Initialize attention parameters."""
        self.window_area = reduce(mul, self.window_size, 1)

        # Relative position bias table
        if self.use_rel_pos_bias:
            self.relative_position_bias_table_size = get_relative_position_bias_table_size(
                self.window_size
            )

    @nn.compact
    def __call__(
        self,
        x: Array,
        mask: Array | None = None,
        deterministic: bool = True,
    ) -> Array:
        """Apply window attention.

        Args:
            x: Input tensor of shape (num_windows * B, window_area, C).
            mask: Optional attention mask of shape
                (num_windows, window_area, window_area) for shifted windows.
            deterministic: Whether to apply dropout.

        Returns:
            Output tensor of same shape as input.
        """
        batch_windows, seq_len, embed_dim = x.shape
        head_dim = self.head_dim or embed_dim // self.num_heads
        inner_dim = self.num_heads * head_dim
        scale = head_dim**-0.5

        # QKV projection
        qkv = nn.Dense(
            inner_dim * 3,
            use_bias=self.qkv_bias,
            dtype=self.dtype,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            name="qkv",
        )(x)

        # Reshape to (B*num_windows, N, 3, num_heads, head_dim)
        qkv = qkv.reshape(batch_windows, seq_len, 3, self.num_heads, head_dim)
        # Transpose to (3, B*num_windows, num_heads, N, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scale query
        q = q * scale

        # Compute attention scores: (B*num_windows, num_heads, N, N)
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k)

        # Add relative position bias if enabled
        if self.use_rel_pos_bias:
            # Get relative position index
            relative_position_index = get_relative_position_index(self.window_size)

            # Create bias table
            bias_table = self.param(
                "relative_position_bias_table",
                nn.initializers.truncated_normal(stddev=0.02),
                (self.relative_position_bias_table_size, self.num_heads),
            )

            # Gather biases using index
            relative_position_bias = bias_table[relative_position_index.flatten()]
            relative_position_bias = relative_position_bias.reshape(
                self.window_area, self.window_area, self.num_heads
            )
            # Transpose to (num_heads, N, N)
            relative_position_bias = jnp.transpose(relative_position_bias, (2, 0, 1))

            # Add to attention (broadcasting over batch dimension)
            attn = attn + relative_position_bias[None, :, :, :]

        # Apply attention mask for shifted windows
        if mask is not None:
            # mask shape: (num_windows, 1, window_area, window_area)
            # Reshape to match attention: need to handle batch dimension
            num_windows = mask.shape[0]
            # Infer batch size
            batch_size = batch_windows // num_windows

            # Reshape attention for masking
            attn = attn.reshape(batch_size, num_windows, self.num_heads, seq_len, seq_len)
            # Expand mask for batch and heads
            mask = mask[None, :, :, :, :]  # (1, num_windows, 1, N, N)
            attn = attn + mask
            attn = attn.reshape(batch_windows, self.num_heads, seq_len, seq_len)

        # Softmax
        attn = jax.nn.softmax(attn, axis=-1)

        # Attention dropout
        if not deterministic and self.attn_drop_rate > 0:
            attn = nn.Dropout(rate=self.attn_drop_rate)(attn, deterministic=False)

        # Apply attention to values
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)

        # Reshape: (B*num_windows, num_heads, N, head_dim) -> (B*num_windows, N, C)
        out = jnp.transpose(out, (0, 2, 1, 3))
        out = out.reshape(batch_windows, seq_len, embed_dim)

        # Output projection
        out = nn.Dense(
            embed_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            name="proj",
        )(out)

        # Output dropout
        if not deterministic and self.proj_drop_rate > 0:
            out = nn.Dropout(rate=self.proj_drop_rate)(out, deterministic=False)

        return out


class ShiftedWindowAttention(nn.Module):
    """Shifted Window Attention wrapper for Swin Transformer blocks.

    This module handles the complete shifted window attention operation:
    1. Cyclic shift (if shift_size > 0)
    2. Pad to window size
    3. Partition into windows
    4. Apply window attention with mask
    5. Reverse partitioning
    6. Reverse cyclic shift

    Attributes:
        num_heads: Number of attention heads.
        window_size: Size of attention window.
        shift_size: Size of shift for SW-MSA. Use (0, 0, ...) for W-MSA.
        qkv_bias: Whether to use bias in QKV projections.
        attn_drop_rate: Dropout rate for attention weights.
        proj_drop_rate: Dropout rate for output projection.
        use_rel_pos_bias: Whether to use relative position bias.
        dtype: Computation dtype.
    """

    num_heads: int
    window_size: tuple[int, ...]
    shift_size: tuple[int, ...]
    qkv_bias: bool = True
    attn_drop_rate: float = 0.0
    proj_drop_rate: float = 0.0
    use_rel_pos_bias: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: Array,
        deterministic: bool = True,
    ) -> Array:
        """Apply shifted window attention.

        Args:
            x: Input tensor of shape (B, *spatial_dims, C).
            deterministic: Whether to apply dropout.

        Returns:
            Output tensor of same shape as input.
        """
        spatial_shape = x.shape[1:-1]
        channels = x.shape[-1]
        window_area = reduce(mul, self.window_size, 1)

        # Check if we need shifting
        do_shift = any(s > 0 for s in self.shift_size)

        # Store original shape for unpadding
        original_spatial = spatial_shape

        # Pad if needed
        x, original_spatial = pad_for_window(x, self.window_size)
        padded_spatial = x.shape[1:-1]

        # Apply cyclic shift if needed
        if do_shift:
            x = cyclic_shift(x, tuple(-s for s in self.shift_size))

            # Create attention mask
            attn_mask = create_shifted_window_mask(
                padded_spatial,
                self.window_size,
                self.shift_size,
                dtype=self.dtype,
            )
            # Reshape: (1, num_windows, 1, N, N) -> (num_windows, 1, N, N)
            attn_mask = attn_mask[0]
        else:
            attn_mask = None

        # Partition into windows: (B, *spatial, C) -> (B*num_windows, *window_size, C)
        x_windows = partition_windows(x, self.window_size)

        # Flatten window spatial dims: (B*num_windows, *window_size, C) -> (B*num_windows, window_area, C)
        num_windows_total = x_windows.shape[0]
        x_windows = x_windows.reshape(num_windows_total, window_area, channels)

        # Apply window attention
        attn_windows = WindowAttention(
            num_heads=self.num_heads,
            window_size=self.window_size,
            qkv_bias=self.qkv_bias,
            attn_drop_rate=self.attn_drop_rate,
            proj_drop_rate=self.proj_drop_rate,
            use_rel_pos_bias=self.use_rel_pos_bias,
            dtype=self.dtype,
            name="attn",
        )(x_windows, mask=attn_mask, deterministic=deterministic)

        # Reshape windows back: (B*num_windows, window_area, C) -> (B*num_windows, *window_size, C)
        attn_windows = attn_windows.reshape(num_windows_total, *self.window_size, channels)

        # Reverse window partitioning: (B*num_windows, *window_size, C) -> (B, *spatial, C)
        x = reverse_partition_windows(attn_windows, padded_spatial, self.window_size)

        # Reverse cyclic shift
        if do_shift:
            x = cyclic_shift(x, self.shift_size)

        # Remove padding if we added any
        if original_spatial != padded_spatial:
            x = unpad_from_window(x, original_spatial)

        return x


def get_attention_mask_cache_key(
    spatial_shape: tuple[int, ...],
    window_size: tuple[int, ...],
    shift_size: tuple[int, ...],
) -> str:
    """Generate cache key for attention mask.

    Args:
        spatial_shape: Shape of spatial dimensions.
        window_size: Size of attention window.
        shift_size: Amount of shift.

    Returns:
        String cache key.
    """
    return f"{spatial_shape}_{window_size}_{shift_size}"
