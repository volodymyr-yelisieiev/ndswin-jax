"""N-dimensional patch embedding for Swin Transformer.

This module implements patch embedding that projects raw input pixels/voxels
into patch tokens for transformer processing.
"""

from functools import reduce
from operator import mul

import jax.numpy as jnp
from flax import linen as nn

from ndswin.types import Array


class PatchEmbed(nn.Module):
    """N-dimensional Patch Embedding layer.

    This layer converts the input tensor into a sequence of patch embeddings
    using a strided convolution. It works with arbitrary spatial dimensions.

    For a 2D input of shape (B, C, H, W), patches of size (patch_size, patch_size)
    are extracted and linearly embedded to embed_dim dimensions.

    Attributes:
        patch_size: Size of each patch in each spatial dimension.
        embed_dim: Dimension of patch embeddings.
        norm_layer: Whether to apply layer normalization after embedding.
        flatten: Whether to flatten spatial dimensions into a sequence.
        bias: Whether to use bias in the projection.
        dtype: Computation dtype.

    Example:
        >>> # 2D image embedding
        >>> embed = PatchEmbed(patch_size=(4, 4), embed_dim=96)
        >>> x = jnp.ones((2, 3, 224, 224))  # (B, C, H, W)
        >>> tokens = embed(x)  # (2, 3136, 96) or (2, 56, 56, 96)

        >>> # 3D volume embedding
        >>> embed = PatchEmbed(patch_size=(4, 4, 4), embed_dim=96)
        >>> x = jnp.ones((2, 1, 64, 64, 64))  # (B, C, D, H, W)
        >>> tokens = embed(x)  # (2, 4096, 96) or (2, 16, 16, 16, 96)
    """

    patch_size: tuple[int, ...]
    embed_dim: int
    norm_layer: bool = True
    flatten: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Apply patch embedding.

        Args:
            x: Input tensor of shape (B, C, *spatial_dims) in channel-first format.

        Returns:
            If flatten=True: (B, num_patches, embed_dim)
            If flatten=False: (B, *grid_shape, embed_dim)

        Raises:
            ValueError: If input spatial dimensions are not divisible by patch_size.
        """
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        spatial_shape = x.shape[2:]
        num_dims = len(spatial_shape)

        if len(self.patch_size) != num_dims:
            raise ValueError(
                f"patch_size must have {num_dims} elements to match input spatial dims, "
                f"got {len(self.patch_size)}"
            )

        # Validate divisibility
        for i, (s, p) in enumerate(zip(spatial_shape, self.patch_size)):
            if s % p != 0:
                raise ValueError(
                    f"Spatial dimension {i} ({s}) must be divisible by patch_size ({p})"
                )

        # Compute grid shape (number of patches per dimension)
        grid_shape = tuple(s // p for s, p in zip(spatial_shape, self.patch_size))
        num_patches = reduce(mul, grid_shape, 1)

        # Convert to channel-last for convolution
        # From (B, C, *spatial) to (B, *spatial, C)
        perm = [0] + list(range(2, 2 + num_dims)) + [1]
        x = jnp.transpose(x, perm)

        # Use appropriate convolution based on dimensionality
        if num_dims == 1:
            conv_fn = nn.Conv
            x = x[..., None]  # Add dummy dim for Conv1D
        elif num_dims == 2 or num_dims == 3:
            conv_fn = nn.Conv
        else:
            # For higher dimensions, use a general approach with reshape
            conv_fn = None

        if conv_fn is not None:
            # Use strided convolution for patch extraction and projection
            x = conv_fn(
                features=self.embed_dim,
                kernel_size=self.patch_size,
                strides=self.patch_size,
                padding="VALID",
                use_bias=self.bias,
                dtype=self.dtype,
                kernel_init=nn.initializers.truncated_normal(stddev=0.02),
                name="proj",
            )(x)
        else:
            # Fallback for higher dimensions: manual patch extraction
            x = self._manual_patch_embed(x, in_channels, grid_shape)

        # x is now (B, *grid_shape, embed_dim)

        # Apply layer normalization if requested
        if self.norm_layer:
            x = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, name="norm")(x)

        # Optionally flatten spatial dimensions
        if self.flatten:
            x = x.reshape(batch_size, num_patches, self.embed_dim)

        return x

    def _manual_patch_embed(
        self,
        x: Array,
        in_channels: int,
        grid_shape: tuple[int, ...],
    ) -> Array:
        """Manual patch embedding for high-dimensional inputs.

        Args:
            x: Input tensor (B, *spatial, C).
            in_channels: Number of input channels.
            grid_shape: Number of patches per dimension.

        Returns:
            Embedded patches (B, *grid_shape, embed_dim).
        """
        batch_size = x.shape[0]
        spatial_shape = x.shape[1:-1]
        num_dims = len(spatial_shape)

        # Reshape to separate patches
        # (B, *spatial, C) -> (B, G0, P0, G1, P1, ..., Gn, Pn, C)
        new_shape = [batch_size]
        for g, p in zip(grid_shape, self.patch_size):
            new_shape.extend([g, p])
        new_shape.append(in_channels)

        x = x.reshape(new_shape)

        # Permute to group grid and patch dimensions
        # (B, G0, P0, G1, P1, ..., C) -> (B, G0, G1, ..., P0, P1, ..., C)
        perm = [0]
        perm.extend(range(1, 2 * num_dims + 1, 2))  # Grid dims
        perm.extend(range(2, 2 * num_dims + 2, 2))  # Patch dims
        perm.append(2 * num_dims + 1)  # Channel dim

        x = jnp.transpose(x, perm)

        # Flatten patch dimensions and apply linear projection
        # (B, G0, G1, ..., P0*P1*...*C)
        patch_size_total = reduce(mul, self.patch_size, 1)
        x = x.reshape(batch_size, *grid_shape, patch_size_total * in_channels)

        # Linear projection to embed_dim
        x = nn.Dense(
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            name="proj",
        )(x)

        return x


class PatchMerging(nn.Module):
    """Patch Merging layer for hierarchical feature reduction.

    This layer merges adjacent patches to reduce the spatial resolution
    by a factor of 2 in each dimension while increasing the channel dimension.

    For 2D: groups 2x2 patches -> 4C channels -> project to 2C
    For 3D: groups 2x2x2 patches -> 8C channels -> project to 2C
    General: groups 2^n patches -> 2^n * C channels -> project to 2C

    Attributes:
        dim: Output dimension (2 * input dimension by default).
        norm_layer: Whether to apply layer normalization.
        dtype: Computation dtype.
    """

    dim: int | None = None
    norm_layer: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Apply patch merging.

        Args:
            x: Input tensor of shape (B, *spatial_dims, C) where each
               spatial dimension is divisible by 2.

        Returns:
            Merged tensor of shape (B, *spatial_dims//2, dim).

        Raises:
            ValueError: If spatial dimensions are not divisible by 2.
        """
        batch_size = x.shape[0]
        spatial_shape = x.shape[1:-1]
        channels = x.shape[-1]
        num_dims = len(spatial_shape)

        # Validate divisibility by 2
        for i, s in enumerate(spatial_shape):
            if s % 2 != 0:
                raise ValueError(
                    f"Spatial dimension {i} ({s}) must be divisible by 2 for patch merging"
                )

        # Output dimension
        out_dim = self.dim if self.dim is not None else 2 * channels

        # New spatial shape after merging
        new_spatial = tuple(s // 2 for s in spatial_shape)

        # Number of patches merged (2^num_dims)
        num_merged = 2**num_dims

        # Reshape to separate even/odd positions
        # For 2D: (B, H, W, C) -> (B, H//2, 2, W//2, 2, C)
        # For 3D: (B, D, H, W, C) -> (B, D//2, 2, H//2, 2, W//2, 2, C)
        new_shape = [batch_size]
        for s in spatial_shape:
            new_shape.extend([s // 2, 2])
        new_shape.append(channels)

        x = x.reshape(new_shape)

        # Permute to group merged patches
        # Move all the 2s to the end before channels
        perm = [0]
        for i in range(num_dims):
            perm.append(1 + 2 * i)  # Spatial dims (halved)
        for i in range(num_dims):
            perm.append(2 + 2 * i)  # The 2s
        perm.append(1 + 2 * num_dims)  # Channels

        x = jnp.transpose(x, perm)

        # Flatten the 2s and channels
        # (B, *new_spatial, 2, 2, ..., C) -> (B, *new_spatial, 2^n * C)
        x = x.reshape(batch_size, *new_spatial, num_merged * channels)

        # Apply layer normalization
        if self.norm_layer:
            x = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, name="norm")(x)

        # Linear projection to output dimension
        x = nn.Dense(
            out_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
            name="reduction",
        )(x)

        return x


class PatchExpanding(nn.Module):
    """Patch Expanding layer for upsampling.

    This is the inverse of PatchMerging, used in decoder architectures
    like Swin-UNet.

    Attributes:
        dim: Output dimension.
        scale_factor: Upsampling factor (default 2).
        norm_layer: Whether to apply layer normalization.
        dtype: Computation dtype.
    """

    dim: int
    scale_factor: int = 2
    norm_layer: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Apply patch expanding.

        Args:
            x: Input tensor of shape (B, *spatial_dims, C).

        Returns:
            Expanded tensor of shape (B, *spatial_dims*scale, dim).
        """
        batch_size = x.shape[0]
        spatial_shape = x.shape[1:-1]
        num_dims = len(spatial_shape)

        # Compute number of output channels before reshape
        scale_volume = self.scale_factor**num_dims
        expand_channels = self.dim * scale_volume

        # Linear projection to expand channels
        x = nn.Dense(
            expand_channels,
            use_bias=False,
            dtype=self.dtype,
            name="expand",
        )(x)

        # Reshape to separate scale factor dimensions
        # (B, *spatial, dim * scale^n) -> (B, *spatial, dim, scale, scale, ...)
        new_shape = [batch_size] + list(spatial_shape) + [self.dim]
        for _ in range(num_dims):
            new_shape.append(self.scale_factor)
        x = x.reshape(new_shape)

        # Permute to interleave scale factors with spatial dims
        # (B, S0, S1, ..., C, scale, scale, ...) -> (B, S0, scale, S1, scale, ..., C)
        perm = [0]
        for i in range(num_dims):
            perm.append(1 + i)  # Spatial dim
            perm.append(1 + num_dims + 1 + i)  # Scale factor
        perm.append(1 + num_dims)  # Channels

        x = jnp.transpose(x, perm)

        # Merge spatial and scale dimensions
        new_spatial = tuple(s * self.scale_factor for s in spatial_shape)
        x = x.reshape(batch_size, *new_spatial, self.dim)

        # Apply layer normalization
        if self.norm_layer:
            x = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, name="norm")(x)

        return x


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings for patch sequences.

    Attributes:
        num_positions: Number of positions (patches).
        embed_dim: Embedding dimension.
        dropout_rate: Dropout rate for embeddings.
        dtype: Computation dtype.
    """

    num_positions: int
    embed_dim: int
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: Array,
        deterministic: bool = True,
    ) -> Array:
        """Add positional embeddings to input.

        Args:
            x: Input tensor of shape (B, N, C) or (B, *spatial, C).
            deterministic: Whether to apply dropout.

        Returns:
            Input with positional embeddings added.
        """
        # Get position embedding parameter
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, self.num_positions, self.embed_dim),
        )

        # Reshape input if needed
        input_shape = x.shape
        if len(input_shape) > 3:
            # Flatten spatial dimensions
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1, self.embed_dim)

        # Add positional embeddings
        x = x + pos_embed

        # Apply dropout
        if not deterministic and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)

        # Reshape back if needed
        if len(input_shape) > 3:
            x = x.reshape(input_shape)

        return x
