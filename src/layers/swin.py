"""Swin Transformer primitives implemented in JAX/Flax."""

from __future__ import annotations

from math import ceil
from typing import Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from .attention import WindowAttention, compute_mask, window_partition, window_reverse
from .patching import pad_to_blocks, unpad
from ..utils.mlp import MLP
from ..utils.stochastic import stochastic_depth


def get_window_size(
    x_size: Sequence[int], window_size: Sequence[int], shift_size: Optional[Sequence[int]] = None
) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
    """Clip window and shift sizes when inputs are smaller than a window.
    
    This is useful when the input spatial dimensions are smaller than the
    configured window size, which can happen at lower resolutions.
    
    Args:
        x_size: Spatial dimensions of the input
        window_size: Configured window size for each dimension
        shift_size: Configured shift size for each dimension (optional)
        
    Returns:
        Tuple of (adjusted_window_size, adjusted_shift_size)
    """
    use_window = list(window_size)
    use_shift = list(shift_size) if shift_size is not None else None

    for idx, size in enumerate(x_size):
        if size <= window_size[idx]:
            use_window[idx] = size
            if use_shift is not None:
                use_shift[idx] = 0

    return tuple(use_window), (tuple(use_shift) if use_shift is not None else None)


class SwinTransformerBlock(nn.Module):
    """Single Swin Transformer block supporting arbitrary dimensional data.
    
    This block applies window-based multi-head self-attention followed by
    a feed-forward MLP, with residual connections and optional stochastic depth.
    
    Attributes:
        space: Number of spatial dimensions
        dim: Number of input channels
        num_heads: Number of attention heads
        grid_size: Spatial grid dimensions
        window_size: Window size for each dimension
        shift_size: Shift amount for each dimension (0 for no shift)
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        qkv_bias: If True, add bias to QKV projection
        mlp_drop: Dropout rate for MLP and attention projection
        attn_drop: Dropout rate for attention weights
        drop_path_rate: Stochastic depth rate
        norm_layer: Normalization layer factory
        activation: Activation function
    """

    space: int
    dim: int
    num_heads: int
    grid_size: Sequence[int]
    window_size: Sequence[int]
    shift_size: Sequence[int]
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    mlp_drop: float = 0.0
    attn_drop: float = 0.0
    drop_path_rate: float = 0.0
    norm_layer: Callable[[], nn.Module] = nn.LayerNorm
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu

    def setup(self) -> None:
        self.norm1 = self.norm_layer()
        self.attn = WindowAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            proj_drop=self.mlp_drop,
        )
        self.norm2 = self.norm_layer()
        hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = MLP(
            features=[self.dim, hidden_dim, self.dim],
            activation=self.activation,
            dropout_rate=self.mlp_drop,
        )

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray], *, deterministic: bool) -> jnp.ndarray:
        """Apply Swin Transformer block.
        
        Args:
            x: Input tensor of shape (batch, *spatial_dims, channels)
            mask: Optional attention mask for shifted windows
            deterministic: If True, disable dropout and stochastic depth
            
        Returns:
            Output tensor of the same shape as input
        """
        residual = x
        y = self.norm1(x)
        base_shape = y.shape[1:-1]
        y, pad_axes = pad_to_blocks(y, self.window_size)
        padded_shape = y.shape

        # Apply cyclic shift if needed
        if any(self.shift_size):
            shift_axes = tuple(range(1, self.space + 1))
            y = jnp.roll(y, shift=tuple(-s for s in self.shift_size), axis=shift_axes)

        # Window partition and attention
        windows = window_partition(y, self.window_size)
        attn_mask = mask if any(self.shift_size) else None
        y = self.attn(windows, attn_mask, deterministic=deterministic)
        y = y.reshape((-1, *self.window_size, self.dim))
        y = window_reverse(y, self.window_size, (padded_shape[0], *padded_shape[1:-1]))

        # Reverse cyclic shift
        if any(self.shift_size):
            shift_axes = tuple(range(1, self.space + 1))
            y = jnp.roll(y, shift=self.shift_size, axis=shift_axes)

        y = unpad(y, pad_axes, base_shape)

        # Apply stochastic depth and add residual
        mlp_rng = None
        if self.drop_path_rate > 0.0 and not deterministic:
            drop_rng = self.make_rng("drop_path")
            drop_rng, mlp_rng = jax.random.split(drop_rng)
            y = stochastic_depth(y, self.drop_path_rate, deterministic=False, rng=drop_rng)

        x = residual + y

        # MLP block
        z = self.norm2(x)
        z = self.mlp(z, deterministic=deterministic)

        if self.drop_path_rate > 0.0 and not deterministic:
            z = stochastic_depth(z, self.drop_path_rate, deterministic=False, rng=mlp_rng)

        return x + z


class SwinLayer(nn.Module):
    """Stack of Swin Transformer blocks with optional attention mask caching.
    
    This layer stacks multiple Swin Transformer blocks, alternating between
    regular and shifted windowing patterns for better information flow.
    
    Attributes:
        space: Number of spatial dimensions
        dim: Number of input channels
        depth: Number of transformer blocks in this layer
        num_heads: Number of attention heads
        grid_size: Spatial grid dimensions
        window_size: Window size for each dimension
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        qkv_bias: If True, add bias to QKV projection
        drop_path: Stochastic depth rate (float or sequence)
        mlp_drop: Dropout rate for MLP and attention projection
        attn_drop: Dropout rate for attention weights
        norm_layer: Normalization layer factory
        activation: Activation function
    """

    space: int
    dim: int
    depth: int
    num_heads: int
    grid_size: Sequence[int]
    window_size: Sequence[int]
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    drop_path: float | Sequence[float] = 0.0
    mlp_drop: float = 0.0
    attn_drop: float = 0.0
    norm_layer: Callable[[], nn.Module] = nn.LayerNorm
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu

    def setup(self) -> None:
        if isinstance(self.drop_path, float):
            drop_rates = [self.drop_path] * self.depth
        else:
            drop_rates = list(self.drop_path)
            if len(drop_rates) != self.depth:
                raise ValueError("drop_path sequence length must match depth")

        self.shift_size = tuple(w // 2 for w in self.window_size)
        self.no_shift = (0,) * len(self.window_size)

        self.blocks = [
            SwinTransformerBlock(
                space=self.space,
                dim=self.dim,
                num_heads=self.num_heads,
                grid_size=self.grid_size,
                window_size=self.window_size,
                shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                mlp_drop=self.mlp_drop,
                attn_drop=self.attn_drop,
                drop_path_rate=drop_rates[i],
                norm_layer=self.norm_layer,
                activation=self.activation,
            )
            for i in range(self.depth)
        ]

        window_size, shift_size = get_window_size(self.grid_size, self.window_size, self.shift_size)
        if shift_size is not None and any(shift_size):
            mask_dims = [ceil(self.grid_size[i] / window_size[i]) * window_size[i] for i in range(len(window_size))]
            attn_mask = compute_mask(mask_dims, window_size, shift_size)
            self.attn_mask = attn_mask
        else:
            self.attn_mask = None

    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        """Apply stack of Swin Transformer blocks.
        
        Args:
            x: Input tensor of shape (batch, *spatial_dims, channels)
            deterministic: If True, disable dropout and stochastic depth
            
        Returns:
            Output tensor of the same shape as input
        """
        mask = self.attn_mask
        for block in self.blocks:
            x = block(x, mask, deterministic=deterministic)
        return x


__all__ = [
    "get_window_size",
    "SwinTransformerBlock",
    "SwinLayer",
]
