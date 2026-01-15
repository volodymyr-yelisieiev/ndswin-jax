"""Classifier wrapper for NDSwin Transformer.

This module provides a higher-level classifier interface that wraps
the NDSwinTransformer model.
"""

from typing import Any, cast

import jax
import jax.numpy as jnp
from flax import linen as nn

from ndswin.config import NDSwinConfig
from ndswin.models.swin import NDSwinTransformer
from ndswin.types import Array


class SwinClassifier(nn.Module):
    """Classification model wrapping NDSwinTransformer.

    This provides a convenient interface for classification tasks,
    handling model creation and configuration.

    Attributes:
        config: NDSwinConfig for the backbone.
        num_classes: Number of output classes. Overrides config.num_classes.

    Example:
        >>> config = NDSwinConfig.swin_tiny_2d()
        >>> classifier = SwinClassifier(config, num_classes=10)
        >>> x = jnp.ones((2, 3, 224, 224))
        >>> variables = classifier.init(jax.random.PRNGKey(0), x)
        >>> logits = classifier.apply(variables, x)
    """

    config: NDSwinConfig
    num_classes: int | None = None

    def setup(self) -> None:
        """Set up the model."""
        # Override num_classes if specified
        if self.num_classes is not None:
            # Create modified config
            config_dict = self.config.to_dict()
            config_dict["num_classes"] = self.num_classes
            self._config = NDSwinConfig.from_dict(config_dict)
        else:
            self._config = self.config

    @nn.compact
    def __call__(
        self,
        x: Array,
        deterministic: bool = True,
        return_features: bool = False,
    ) -> Array | dict[str, Array]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, *spatial_dims).
            deterministic: Whether to use deterministic operations.
            return_features: Whether to return intermediate features.

        Returns:
            Logits or dictionary with logits and features.
        """
        return NDSwinTransformer(self._config, name="backbone")(
            x, deterministic=deterministic, return_features=return_features
        )

    def extract_features(
        self,
        x: Array,
        deterministic: bool = True,
    ) -> Array:
        """Extract features without classification head.

        Args:
            x: Input tensor.
            deterministic: Whether to use deterministic operations.

        Returns:
            Feature tensor before classification head.
        """
        # Create config without classification head
        config_dict = self._config.to_dict()
        config_dict["num_classes"] = None
        feature_config = NDSwinConfig.from_dict(config_dict)

        return cast(
            Array,
            NDSwinTransformer(feature_config, name="backbone_features")(
                x, deterministic=deterministic
            ),
        )


class MultiScaleSwinClassifier(nn.Module):
    """Multi-scale classifier using features from multiple stages.

    This classifier aggregates features from multiple stages of the
    Swin Transformer for improved classification.

    Attributes:
        config: NDSwinConfig for the backbone.
        num_classes: Number of output classes.
        fusion_type: How to fuse multi-scale features ('concat', 'add', 'attention').

    Example:
        >>> config = NDSwinConfig.swin_tiny_2d()
        >>> classifier = MultiScaleSwinClassifier(config, num_classes=10)
        >>> x = jnp.ones((2, 3, 224, 224))
        >>> variables = classifier.init(jax.random.PRNGKey(0), x)
        >>> logits = classifier.apply(variables, x)
    """

    config: NDSwinConfig
    num_classes: int
    fusion_type: str = "concat"

    @nn.compact
    def __call__(
        self,
        x: Array,
        deterministic: bool = True,
    ) -> Array:
        """Forward pass with multi-scale feature fusion.

        Args:
            x: Input tensor.
            deterministic: Whether to use deterministic operations.

        Returns:
            Logits of shape (B, num_classes).
        """
        # Get features from all stages
        result = cast(
            dict[str, Any],
            NDSwinTransformer(self.config, name="backbone")(
                x, deterministic=deterministic, return_features=True
            ),
        )
        features = result["features"]

        batch_size = x.shape[0]

        # Pool each feature map
        pooled_features = []
        for i, feat in enumerate(features):
            # Global average pooling
            pooled = jnp.mean(feat.reshape(batch_size, -1, feat.shape[-1]), axis=1)
            pooled_features.append(pooled)

        # Fuse features
        if self.fusion_type == "concat":
            fused = jnp.concatenate(pooled_features, axis=-1)
        elif self.fusion_type == "add":
            # Project all to same dimension first
            proj_features = []
            target_dim = pooled_features[-1].shape[-1]
            for i, feat in enumerate(pooled_features):
                if feat.shape[-1] != target_dim:
                    feat = nn.Dense(target_dim, name=f"proj_{i}")(feat)
                proj_features.append(feat)
            fused = sum(proj_features)
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        # Classification head
        fused = nn.LayerNorm(name="norm")(fused)
        if not deterministic:
            fused = nn.Dropout(rate=self.config.drop_rate)(fused, deterministic=False)
        logits = nn.Dense(self.num_classes, name="head")(fused)

        return logits


class SwinForSegmentation(nn.Module):
    """Swin Transformer for semantic segmentation.

    This model uses the Swin Transformer as an encoder and adds
    a decoder for dense prediction.

    Attributes:
        config: NDSwinConfig for the encoder.
        num_classes: Number of segmentation classes.
        decoder_channels: Number of channels in decoder layers.

    Example:
        >>> config = NDSwinConfig.swin_tiny_2d()
        >>> model = SwinForSegmentation(config, num_classes=21)
        >>> x = jnp.ones((2, 3, 224, 224))
        >>> variables = model.init(jax.random.PRNGKey(0), x)
        >>> masks = model.apply(variables, x)
        >>> print(masks.shape)  # (2, 21, 224, 224)
    """

    config: NDSwinConfig
    num_classes: int
    decoder_channels: int = 256

    @nn.compact
    def __call__(
        self,
        x: Array,
        deterministic: bool = True,
    ) -> Array:
        """Forward pass for segmentation.

        Args:
            x: Input tensor of shape (B, C, H, W) for 2D.
            deterministic: Whether to use deterministic operations.

        Returns:
            Segmentation masks of shape (B, num_classes, H, W).
        """
        batch_size = x.shape[0]
        input_spatial = x.shape[2:]

        # Get multi-scale features
        result = cast(
            dict[str, Any],
            NDSwinTransformer(self.config, name="encoder")(
                x, deterministic=deterministic, return_features=True
            ),
        )
        features = result["features"]

        # Simple FPN-style decoder for 2D
        # Start from deepest features and progressively upsample
        decoded = features[-1]

        for i in range(len(features) - 2, -1, -1):
            skip = features[i]
            skip_channels = skip.shape[-1]

            # Upsample decoded features

            # Calculate upsampling factor
            # Use transpose convolution or resize
            # For simplicity, use resize
            target_shape = skip.shape[:-1]
            decoded = jax.image.resize(
                decoded,
                target_shape + (decoded.shape[-1],),
                method="bilinear",
            )

            # Project to same channels
            decoded = nn.Dense(skip_channels, name=f"lateral_{i}")(decoded)

            # Add skip connection
            decoded = decoded + skip

            # Refine
            decoded = nn.Conv(
                skip_channels, (3,) * self.config.num_dims, padding="SAME", name=f"refine_{i}"
            )(decoded)
            decoded = nn.gelu(decoded)

        # Final upsampling to input resolution
        decoded = jax.image.resize(
            decoded,
            (batch_size,) + input_spatial + (decoded.shape[-1],),
            method="bilinear",
        )

        # Classification head
        logits = nn.Conv(self.num_classes, (1,) * self.config.num_dims, name="seg_head")(decoded)

        # Convert to channel-first (B, *spatial, C) -> (B, C, *spatial)
        perm = [0, logits.ndim - 1] + list(range(1, logits.ndim - 1))
        logits = jnp.transpose(logits, perm)

        return logits


def create_classifier(
    model_name: str = "swin_tiny",
    num_dims: int = 2,
    num_classes: int = 1000,
    **kwargs: Any,
) -> SwinClassifier:
    """Create a Swin classifier by name.

    Args:
        model_name: Model variant name ('swin_tiny', 'swin_small', 'swin_base', 'swin_large').
        num_dims: Number of spatial dimensions.
        num_classes: Number of output classes.
        **kwargs: Additional configuration options.

    Returns:
        SwinClassifier instance.
    """
    # Get base config
    if model_name == "swin_tiny":
        if num_dims == 2:
            config = NDSwinConfig.swin_tiny_2d(num_classes)
        elif num_dims == 3:
            config = NDSwinConfig.swin_tiny_3d(num_classes)
        elif num_dims == 4:
            config = NDSwinConfig.swin_tiny_4d(num_classes)
        else:
            raise ValueError(f"No preset for {num_dims}D swin_tiny")
    elif model_name == "swin_small":
        config = NDSwinConfig.swin_small_2d(num_classes)
    elif model_name == "swin_base":
        config = NDSwinConfig.swin_base_2d(num_classes)
    elif model_name == "swin_large":
        config = NDSwinConfig.swin_large_2d(num_classes)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Apply overrides
    if kwargs:
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        config = NDSwinConfig.from_dict(config_dict)

    return SwinClassifier(config, num_classes=num_classes)
