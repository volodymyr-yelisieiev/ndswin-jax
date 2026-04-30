"""Building blocks for Swin Transformer.

This module implements the core building blocks including Swin Transformer blocks,
MLP layers, DropPath, and other components.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

from ndswin.core.attention import ShiftedWindowAttention
from ndswin.types import Array


class DropPath(nn.Module):
    """Drop Path (Stochastic Depth) regularization.

    Randomly drops entire samples during training, similar to dropout but
    drops entire residual paths instead of individual activations.

    Attributes:
        drop_prob: Probability of dropping the path.
        scale_by_keep: Whether to scale output by keep probability.
    """

    drop_prob: float = 0.0
    scale_by_keep: bool = True

    @nn.compact
    def __call__(
        self,
        x: Array,
        deterministic: bool = True,
    ) -> Array:
        """Apply drop path.

        Args:
            x: Input tensor.
            deterministic: If True, no drop path is applied.

        Returns:
            Output tensor, potentially with some samples zeroed out.
        """
        if self.drop_prob == 0.0 or deterministic:
            return x

        keep_prob = 1.0 - self.drop_prob

        # Generate random mask (one per sample in batch)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rng = self.make_rng("dropout")
        random_tensor = jax.random.uniform(rng, shape, dtype=x.dtype)
        binary_mask = jnp.floor(random_tensor + keep_prob)

        if self.scale_by_keep:
            output = x / keep_prob * binary_mask
        else:
            output = x * binary_mask

        return output


class MLPBlock(nn.Module):
    """MLP block with GELU activation.

    Standard feed-forward network used in transformers:
    Linear -> GELU -> Dropout -> Linear -> Dropout

    Attributes:
        hidden_dim: Hidden layer dimension.
        out_dim: Output dimension. If None, uses input dimension.
        drop_rate: Dropout rate.
        act_fn: Activation function.
        dtype: Computation dtype.
    """

    hidden_dim: int
    out_dim: int | None = None
    drop_rate: float = 0.0
    act_fn: Callable[[Array], Array] = nn.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: Array,
        deterministic: bool = True,
    ) -> Array:
        """Apply MLP block.

        Args:
            x: Input tensor of shape (..., in_dim).
            deterministic: Whether to apply dropout.

        Returns:
            Output tensor of shape (..., out_dim).
        """
        in_dim = x.shape[-1]
        out_dim = self.out_dim or in_dim

        # First linear layer
        x = nn.Dense(
            self.hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            name="fc1",
        )(x)

        # Activation
        x = self.act_fn(x)

        # Dropout
        if not deterministic and self.drop_rate > 0:
            x = nn.Dropout(rate=self.drop_rate)(x, deterministic=False)

        # Second linear layer
        x = nn.Dense(
            out_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            name="fc2",
        )(x)

        # Dropout
        if not deterministic and self.drop_rate > 0:
            x = nn.Dropout(rate=self.drop_rate)(x, deterministic=False)

        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.

    This block implements:
    1. Layer Norm + (Shifted) Window Attention + Residual
    2. Layer Norm + MLP + Residual

    Attributes:
        dim: Input/output dimension.
        num_heads: Number of attention heads.
        window_size: Size of attention window.
        shift_size: Size of shift for SW-MSA. Use (0, 0, ...) for W-MSA.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        qkv_bias: Whether to use bias in QKV projections.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Drop path rate.
        norm_layer: Normalization layer type.
        dtype: Computation dtype.
    """

    dim: int
    num_heads: int
    window_size: tuple[int, ...]
    shift_size: tuple[int, ...] = ()
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    norm_layer: str = "layernorm"
    use_rel_pos_bias: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        """Set up default shift size if not provided."""
        if not self.shift_size:
            self._shift_size = tuple(0 for _ in self.window_size)
        else:
            self._shift_size = self.shift_size

    @nn.compact
    def __call__(
        self,
        x: Array,
        deterministic: bool = True,
    ) -> Array:
        """Apply Swin Transformer block.

        Args:
            x: Input tensor of shape (B, *spatial_dims, C).
            deterministic: Whether to apply dropout.

        Returns:
            Output tensor of same shape.
        """
        # Store shortcut for residual connection
        shortcut = x

        # Layer Norm 1
        if self.norm_layer == "layernorm":
            x = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, name="norm1")(x)
        else:
            x = nn.RMSNorm(epsilon=1e-6, dtype=self.dtype, name="norm1")(x)

        # (Shifted) Window Attention
        x = ShiftedWindowAttention(
            num_heads=self.num_heads,
            window_size=self.window_size,
            shift_size=self._shift_size,
            qkv_bias=self.qkv_bias,
            attn_drop_rate=self.attn_drop_rate,
            proj_drop_rate=self.drop_rate,
            use_rel_pos_bias=self.use_rel_pos_bias,
            dtype=self.dtype,
            name="attn",
        )(x, deterministic=deterministic)

        # Drop Path
        if self.drop_path_rate > 0:
            x = DropPath(drop_prob=self.drop_path_rate, name="drop_path1")(
                x, deterministic=deterministic
            )

        # Residual connection 1
        x = shortcut + x

        # Store shortcut for second residual
        shortcut = x

        # Layer Norm 2
        if self.norm_layer == "layernorm":
            x = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, name="norm2")(x)
        else:
            x = nn.RMSNorm(epsilon=1e-6, dtype=self.dtype, name="norm2")(x)

        # MLP
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        x = MLPBlock(
            hidden_dim=mlp_hidden_dim,
            out_dim=self.dim,
            drop_rate=self.drop_rate,
            dtype=self.dtype,
            name="mlp",
        )(x, deterministic=deterministic)

        # Drop Path
        if self.drop_path_rate > 0:
            x = DropPath(drop_prob=self.drop_path_rate, name="drop_path2")(
                x, deterministic=deterministic
            )

        # Residual connection 2
        x = shortcut + x

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    This layer consists of multiple Swin Transformer blocks with alternating
    W-MSA and SW-MSA, followed by optional downsampling.

    Attributes:
        dim: Input/output dimension.
        depth: Number of Swin Transformer blocks.
        num_heads: Number of attention heads.
        window_size: Size of attention window.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        qkv_bias: Whether to use bias in QKV projections.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rates: Drop path rates for each block.
        downsample: Whether to apply patch merging at the end.
        norm_layer: Normalization layer type.
        dtype: Computation dtype.
    """

    dim: int
    depth: int
    num_heads: int
    window_size: tuple[int, ...]
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rates: tuple[float, ...] = ()
    downsample: bool = False
    out_dim: int | None = None
    norm_layer: str = "layernorm"
    use_rel_pos_bias: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: Array,
        deterministic: bool = True,
    ) -> Array:
        """Apply basic layer.

        Args:
            x: Input tensor of shape (B, *spatial_dims, C).
            deterministic: Whether to apply dropout.

        Returns:
            Output tensor, possibly with reduced spatial dims if downsample=True.
        """
        num_dims = len(self.window_size)

        # Compute shift size (half of window size)
        shift_size = tuple(w // 2 for w in self.window_size)

        # Ensure drop_path_rates has correct length
        drop_path_rates = self.drop_path_rates or tuple(0.0 for _ in range(self.depth))

        # Apply Swin Transformer blocks
        for i in range(self.depth):
            # Alternate between W-MSA and SW-MSA
            block_shift = shift_size if i % 2 == 1 else tuple(0 for _ in range(num_dims))

            x = SwinTransformerBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                window_size=self.window_size,
                shift_size=block_shift,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                norm_layer=self.norm_layer,
                use_rel_pos_bias=self.use_rel_pos_bias,
                dtype=self.dtype,
                name=f"block{i}",
            )(x, deterministic=deterministic)

        # Apply downsampling if requested
        if self.downsample:
            from ndswin.core.patch_embed import PatchMerging

            out_dim = self.out_dim or self.dim * 2
            x = PatchMerging(
                dim=out_dim,
                norm_layer=True,
                dtype=self.dtype,
                name="downsample",
            )(x)

        return x


class ClassificationHead(nn.Module):
    """Classification head for Swin Transformer.

    Applies global pooling and a linear classifier.

    Attributes:
        num_classes: Number of output classes.
        pool_type: Pooling type ('avg', 'max', or 'token').
        dropout_rate: Dropout rate before classifier.
        dtype: Computation dtype.
    """

    num_classes: int
    pool_type: str = "avg"
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: Array,
        deterministic: bool = True,
    ) -> Array:
        """Apply classification head.

        Args:
            x: Input tensor of shape (B, *spatial_dims, C) or (B, N, C).
            deterministic: Whether to apply dropout.

        Returns:
            Logits of shape (B, num_classes).
        """
        batch_size = x.shape[0]
        channels = x.shape[-1]

        # Determine spatial dimensions
        if len(x.shape) == 3:
            # Already flattened: (B, N, C)
            spatial_flat = x
        else:
            # Has spatial dimensions: (B, *spatial, C)
            spatial_flat = x.reshape(batch_size, -1, channels)

        # Apply pooling
        if self.pool_type == "avg":
            pooled = jnp.mean(spatial_flat, axis=1)  # (B, C)
        elif self.pool_type == "max":
            pooled = jnp.max(spatial_flat, axis=1)  # (B, C)
        elif self.pool_type == "token":
            # Use first token (if using class token)
            pooled = spatial_flat[:, 0]  # (B, C)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        # Layer normalization
        pooled = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, name="norm")(pooled)

        # Dropout
        if not deterministic and self.dropout_rate > 0:
            pooled = nn.Dropout(rate=self.dropout_rate)(pooled, deterministic=False)

        # Classifier
        logits = nn.Dense(
            self.num_classes,
            dtype=self.dtype,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
            name="head",
        )(pooled)

        return logits


class FeatureExtractor(nn.Module):
    """Feature extraction head (no classification).

    Applies global pooling and optional projection.

    Attributes:
        out_dim: Output feature dimension. If None, uses input dimension.
        pool_type: Pooling type ('avg', 'max').
        norm: Whether to normalize output.
        dtype: Computation dtype.
    """

    out_dim: int | None = None
    pool_type: str = "avg"
    norm: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: Array,
        deterministic: bool = True,
    ) -> Array:
        """Extract features.

        Args:
            x: Input tensor of shape (B, *spatial_dims, C) or (B, N, C).
            deterministic: Not used, included for API consistency.

        Returns:
            Features of shape (B, out_dim or C).
        """
        batch_size = x.shape[0]
        channels = x.shape[-1]

        # Flatten spatial dimensions
        if len(x.shape) > 3:
            x = x.reshape(batch_size, -1, channels)

        # Pool
        if self.pool_type == "avg":
            x = jnp.mean(x, axis=1)
        elif self.pool_type == "max":
            x = jnp.max(x, axis=1)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        # Optional normalization
        if self.norm:
            x = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, name="norm")(x)

        # Optional projection
        if self.out_dim is not None and self.out_dim != channels:
            x = nn.Dense(
                self.out_dim,
                dtype=self.dtype,
                name="proj",
            )(x)

        return x
