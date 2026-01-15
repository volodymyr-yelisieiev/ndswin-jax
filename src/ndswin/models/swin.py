"""N-Dimensional Swin Transformer model.

This module implements the main NDSwinTransformer class that supports
arbitrary spatial dimensions (2D, 3D, 4D, and beyond).
"""

from functools import reduce
from operator import mul
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from ndswin.config import NDSwinConfig
from ndswin.core.blocks import BasicLayer, ClassificationHead
from ndswin.core.patch_embed import LearnedPositionalEmbedding, PatchEmbed
from ndswin.types import Array, PRNGKey


class NDSwinTransformer(nn.Module):
    """N-Dimensional Swin Transformer.

    A hierarchical vision transformer using shifted windows for efficient
    self-attention. This implementation supports arbitrary spatial dimensions.

    The architecture consists of:
    1. Patch Embedding: Convert raw input to patch tokens
    2. Optional Positional Embedding: Add learnable position information
    3. Transformer Stages: Multiple stages with increasing channels and
       decreasing spatial resolution
    4. Classification Head: Global pooling and linear classifier

    Attributes:
        config: NDSwinConfig with model hyperparameters.

    Example:
        >>> # 2D Image Classification
        >>> config = NDSwinConfig.swin_tiny_2d(num_classes=1000)
        >>> model = NDSwinTransformer(config)
        >>> x = jnp.ones((2, 3, 224, 224))  # (B, C, H, W)
        >>> variables = model.init(jax.random.PRNGKey(0), x)
        >>> logits = model.apply(variables, x)
        >>> print(logits.shape)  # (2, 1000)

        >>> # 3D Volume Classification
        >>> config = NDSwinConfig.swin_tiny_3d(num_classes=10)
        >>> model = NDSwinTransformer(config)
        >>> x = jnp.ones((2, 1, 64, 64, 64))  # (B, C, D, H, W)
        >>> variables = model.init(jax.random.PRNGKey(0), x)
        >>> logits = model.apply(variables, x)
    """

    config: NDSwinConfig

    def setup(self) -> None:
        """Initialize model components."""
        self.num_stages = self.config.num_stages
        self.num_dims = self.config.num_dims

    @nn.compact
    def __call__(
        self,
        x: Array,
        deterministic: bool = True,
        return_features: bool = False,
    ) -> Array | dict[str, Any]:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (B, C, *spatial_dims).
                For 2D: (B, C, H, W)
                For 3D: (B, C, D, H, W)
                For 4D: (B, C, T, D, H, W)
            deterministic: Whether to use deterministic operations (no dropout).
            return_features: If True, return intermediate features from each stage.

        Returns:
            If return_features is False:
                Logits of shape (B, num_classes) if num_classes is set,
                otherwise features of shape (B, final_embed_dim).
            If return_features is True:
                Dictionary with 'logits' and 'features' (list of stage outputs).
        """
        batch_size = x.shape[0]
        spatial_shape = x.shape[2:]

        # Validate input
        if len(spatial_shape) != self.config.num_dims:
            raise ValueError(
                f"Expected {self.config.num_dims} spatial dimensions, got {len(spatial_shape)}"
            )

        # Note: PatchEmbed handles channel-first to channel-last conversion internally
        # so we pass x directly without converting here

        # Store intermediate features if requested
        features = []

        # Patch Embedding (expects channel-first: B, C, *spatial)
        x = PatchEmbed(
            patch_size=self.config.patch_size,
            embed_dim=self.config.embed_dim,
            norm_layer=True,
            flatten=False,  # Keep spatial structure
            dtype=self.config.dtype,
            name="patch_embed",
        )(x)

        # Compute grid shape after patching
        grid_shape = tuple(s // p for s, p in zip(spatial_shape, self.config.patch_size))
        num_patches = reduce(mul, grid_shape, 1)

        # Optional absolute positional embedding
        if self.config.use_abs_pos_embed:
            x = LearnedPositionalEmbedding(
                num_positions=num_patches,
                embed_dim=self.config.embed_dim,
                dropout_rate=self.config.drop_rate,
                dtype=self.config.dtype,
                name="pos_embed",
            )(x, deterministic=deterministic)

        # Positional dropout
        if not deterministic and self.config.drop_rate > 0:
            x = nn.Dropout(rate=self.config.drop_rate, name="pos_drop")(x, deterministic=False)

        # Compute drop path rates for each block (use numpy for static values to avoid JIT tracing issues)
        total_blocks = sum(self.config.depths)
        dpr = np.linspace(0, self.config.drop_path_rate, total_blocks).tolist()

        # Process through stages
        block_idx = 0
        current_dim = self.config.embed_dim

        for stage_idx in range(self.num_stages):
            depth = self.config.depths[stage_idx]
            num_heads = self.config.num_heads[stage_idx]

            # Get drop path rates for this stage
            stage_dpr = tuple(dpr[block_idx : block_idx + depth])
            block_idx += depth

            # Determine if we should downsample after this stage
            # (all stages except the last)
            downsample = stage_idx < self.num_stages - 1

            # Compute output dimension for this stage
            if downsample:
                out_dim = current_dim * 2
            else:
                out_dim = current_dim

            x = BasicLayer(
                dim=current_dim,
                depth=depth,
                num_heads=num_heads,
                window_size=self.config.window_size,
                mlp_ratio=self.config.mlp_ratio,
                qkv_bias=self.config.qkv_bias,
                drop_rate=self.config.drop_rate,
                attn_drop_rate=self.config.attn_drop_rate,
                drop_path_rates=stage_dpr,
                downsample=downsample,
                out_dim=out_dim if downsample else None,
                norm_layer=self.config.norm_layer,
                use_rel_pos_bias=self.config.use_rel_pos_bias,
                dtype=self.config.dtype,
                name=f"layer{stage_idx}",
            )(x, deterministic=deterministic)

            # Store features
            if return_features:
                features.append(x)

            # Update current dimension after downsampling
            if downsample:
                current_dim = out_dim

        # Final layer normalization
        x = nn.LayerNorm(
            epsilon=self.config.norm_eps,
            dtype=self.config.dtype,
            name="norm",
        )(x)

        # Classification head
        if self.config.num_classes is not None:
            logits = ClassificationHead(
                num_classes=self.config.num_classes,
                pool_type=self.config.pool_type,
                dropout_rate=self.config.drop_rate,
                dtype=self.config.dtype,
                name="head",
            )(x, deterministic=deterministic)
        else:
            # Global average pooling for feature extraction
            logits = jnp.mean(x.reshape(batch_size, -1, current_dim), axis=1)

        if return_features:
            return {"logits": logits, "features": features}
        else:
            return logits

    def get_num_params(self, variables: dict[str, Any]) -> int:
        """Count total number of parameters.

        Args:
            variables: Model variables dictionary.

        Returns:
            Total number of parameters.
        """
        return sum(p.size for p in jax.tree_util.tree_leaves(variables["params"]))

    def get_feature_dims(self) -> list[int]:
        """Get feature dimensions at each stage.

        Returns:
            List of feature dimensions for each stage.
        """
        dims = []
        current_dim = self.config.embed_dim
        for i in range(self.num_stages):
            dims.append(current_dim)
            if i < self.num_stages - 1:
                current_dim *= 2
        return dims


def create_swin_model(
    config: NDSwinConfig | None = None,
    **kwargs: Any,
) -> NDSwinTransformer:
    """Factory function to create a Swin Transformer model.

    Args:
        config: NDSwinConfig object. If None, creates from kwargs.
        **kwargs: Keyword arguments passed to NDSwinConfig if config is None.

    Returns:
        NDSwinTransformer instance.

    Example:
        >>> # Using config object
        >>> config = NDSwinConfig.swin_tiny_2d()
        >>> model = create_swin_model(config)

        >>> # Using kwargs
        >>> model = create_swin_model(
        ...     num_dims=2,
        ...     embed_dim=96,
        ...     depths=(2, 2, 6, 2),
        ...     num_heads=(3, 6, 12, 24),
        ... )
    """
    if config is None:
        config = NDSwinConfig(**kwargs)

    return NDSwinTransformer(config)


def init_swin_model(
    model: NDSwinTransformer,
    key: PRNGKey,
    input_shape: tuple[int, ...],
) -> dict[str, Any]:
    """Initialize model parameters.

    Args:
        model: NDSwinTransformer instance.
        key: JAX PRNG key.
        input_shape: Shape of input tensor (including batch dimension).

    Returns:
        Initialized model variables.

    Example:
        >>> model = NDSwinTransformer(config)
        >>> variables = init_swin_model(
        ...     model,
        ...     jax.random.PRNGKey(0),
        ...     (1, 3, 224, 224),
        ... )
    """
    dummy_input = jnp.ones(input_shape)
    return cast(dict[str, Any], model.init(key, dummy_input))


# Convenience functions for common configurations
def swin_tiny_2d(num_classes: int = 1000) -> NDSwinTransformer:
    """Create Swin-Tiny model for 2D images.

    Args:
        num_classes: Number of output classes.

    Returns:
        NDSwinTransformer instance.
    """
    config = NDSwinConfig.swin_tiny_2d(num_classes)
    return NDSwinTransformer(config)


def swin_small_2d(num_classes: int = 1000) -> NDSwinTransformer:
    """Create Swin-Small model for 2D images.

    Args:
        num_classes: Number of output classes.

    Returns:
        NDSwinTransformer instance.
    """
    config = NDSwinConfig.swin_small_2d(num_classes)
    return NDSwinTransformer(config)


def swin_base_2d(num_classes: int = 1000) -> NDSwinTransformer:
    """Create Swin-Base model for 2D images.

    Args:
        num_classes: Number of output classes.

    Returns:
        NDSwinTransformer instance.
    """
    config = NDSwinConfig.swin_base_2d(num_classes)
    return NDSwinTransformer(config)


def swin_tiny_3d(num_classes: int = 10) -> NDSwinTransformer:
    """Create Swin-Tiny model for 3D volumes.

    Args:
        num_classes: Number of output classes.

    Returns:
        NDSwinTransformer instance.
    """
    config = NDSwinConfig.swin_tiny_3d(num_classes)
    return NDSwinTransformer(config)


def swin_tiny_4d(num_classes: int = 400) -> NDSwinTransformer:
    """Create Swin-Tiny model for 4D spatiotemporal data.

    Args:
        num_classes: Number of output classes.

    Returns:
        NDSwinTransformer instance.
    """
    config = NDSwinConfig.swin_tiny_4d(num_classes)
    return NDSwinTransformer(config)
