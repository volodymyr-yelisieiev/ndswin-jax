"""Patch embedding and merging helpers for N-dimensional tensors."""

from __future__ import annotations

from itertools import product
from math import ceil, prod
from typing import Sequence, Tuple, Type

import flax.linen as nn
import jax.numpy as jnp

from ..utils.mlp import MLP


def pad_to_blocks(x: jnp.ndarray, blocks: Sequence[int]) -> Tuple[jnp.ndarray, Tuple[int, ...]]:
    """Pad ``x`` so each spatial axis is divisible by the corresponding block size."""

    if x.ndim < 3:
        raise ValueError("Expected at least batch, one spatial axis, and channels.")

    pad_widths = []
    pad_spec = [(0, 0)]
    for size, block in zip(x.shape[1:-1], blocks):
        pad = (block - (size % block)) % block
        pad_widths.append(pad)
        pad_spec.append((0, pad))
    pad_spec.append((0, 0))

    if any(pad_widths):
        x = jnp.pad(x, pad_spec, mode="constant")

    return x, tuple(pad_widths)


def unpad(x: jnp.ndarray, pad_widths: Sequence[int], base_grid: Sequence[int]) -> jnp.ndarray:
    """Remove padding that :func:`pad_to_blocks` previously appended."""

    if len(pad_widths) != len(base_grid):
        raise ValueError("`pad_widths` and `base_grid` must share the same length.")

    if any(pad_widths):
        slices = tuple(slice(0, size) for size in base_grid)
        x = x[(slice(None),) + slices + (slice(None),)]
    return x


class PatchEmbed(nn.Module):
    """ViT-style patch embedding for arbitrary dimensional inputs."""

    space: int
    in_resolution: Sequence[int]
    patch_size: Sequence[int]
    embed_dim: int
    in_channels: int = 3
    flatten: bool = True
    use_conv: bool = False
    mlp_ratio: float = 8.0
    mlp_depth: int = 2
    norm_layer: Type[nn.Module] = nn.LayerNorm

    def setup(self) -> None:
        if len(self.in_resolution) != self.space:
            raise ValueError("`in_resolution` dimensionality must match `space`.")
        if len(self.patch_size) != self.space:
            raise ValueError("`patch_size` dimensionality must match `space`.")

        self.grid_size = tuple(
            int(ceil(res / patch)) for res, patch in zip(self.in_resolution, self.patch_size)
        )

        if self.use_conv:
            self.patch = nn.Conv(
                features=self.embed_dim,
                kernel_size=self.patch_size,
                strides=self.patch_size,
                padding="VALID",
                feature_group_count=1,
            )
        else:
            hidden_width = int(self.embed_dim * self.mlp_ratio)
            patch_tokens = self.in_channels * prod(self.patch_size)
            features = [patch_tokens]
            features += [hidden_width] * max(self.mlp_depth - 1, 0)
            features += [self.embed_dim]
            self.patch = MLP(features=features, dropout_rate=0.0)

        self.norm = self.norm_layer()

    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        if x.ndim != self.space + 2:
            raise ValueError(
                f"Expected tensors shaped (batch, *{self.space} spatial dims, channels)"
            )

        if self.use_conv:
            x, _ = pad_to_blocks(x, self.patch_size)
            x = self.patch(x)
        else:
            x, _ = pad_to_blocks(x, self.patch_size)
            batch = x.shape[0]
            channel = x.shape[-1]
            patch_flat_size = prod(self.patch_size)
            flattened = x.reshape(
                batch,
                *self.grid_size,
                patch_flat_size * channel,
            )
            x = self.patch(flattened, deterministic=deterministic)

        if self.flatten:
            x = x.reshape(x.shape[0], -1, x.shape[-1])

        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """Swin-style patch merging block that halves selected spatial axes."""

    space: int
    dim: int
    in_resolution: Sequence[int]
    c_multiplier: int = 2
    norm_layer: Type[nn.Module] = nn.LayerNorm
    mask_spaces: Sequence[bool] | None = None

    def setup(self) -> None:
        if len(self.in_resolution) != self.space:
            raise ValueError("`in_resolution` dimensionality must match `space`.")

        mask_spaces = tuple(self.mask_spaces) if self.mask_spaces is not None else (True,) * self.space
        if len(mask_spaces) != self.space:
            raise ValueError("`mask_spaces` must provide one flag per spatial axis.")
        self.mask_config = mask_spaces

        self.grid_size = tuple(
            int(ceil(res / 2)) if mask else res
            for res, mask in zip(self.in_resolution, self.mask_config)
        )

        self.norm = self.norm_layer()
        self.reduction = nn.Dense(self.c_multiplier * self.dim, use_bias=False)
        self.padding_blocks = tuple(2 if mask else 1 for mask in self.mask_config)

    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        del deterministic
        if x.ndim != self.space + 2:
            raise ValueError(
                f"Expected tensors shaped (batch, *{self.space} spatial dims, channels)"
            )
        x, _ = pad_to_blocks(x, self.padding_blocks)

        subspaces = []
        for offsets in product(*[[0, 1] if mask else [0] for mask in self.mask_config]):
            slices = tuple(
                slice(offset, None, 2) if mask else slice(None)
                for mask, offset in zip(self.mask_config, offsets)
            )
            subspaces.append(x[(slice(None),) + slices + (slice(None),)])

        x = jnp.concatenate(subspaces, axis=-1)
        x = self.norm(x)
        x = self.reduction(x)

        target_grid = self.grid_size
        x = x[(slice(None),) + tuple(slice(0, g) for g in target_grid) + (slice(None),)]
        return x


__all__ = ["pad_to_blocks", "unpad", "PatchEmbed", "PatchMerging"]
