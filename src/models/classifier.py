"""High-level N-dimensional Swin Transformer classifier built with Flax."""

from __future__ import annotations

from math import ceil
from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp

from ..layers import PatchEmbed, PatchMerging, PositionalEmbedding, SwinLayer


class NDSwinClassifier(nn.Module):
    """Hierarchical Swin Transformer classifier for arbitrary dimensional inputs."""

    dim: int
    resolution: Sequence[int]
    space: int = 2
    in_channels: int = 3
    num_classes: int = 3
    patch_size: Sequence[int] = (4, 4)
    window_size: Sequence[int] = (7, 7)
    depth: Sequence[int] = (2, 2, 2)
    num_heads: Sequence[int] = (3, 6, 12)
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.1
    head_drop: float = 0.3
    use_conv: bool = False
    use_abs_pos: bool = False
    merge_mask: Optional[Sequence[bool] | bool] = None
    qkv_bias: bool = True
    mlp_drop: float = 0.0
    attn_drop: float = 0.0
    norm_layer: Callable[[], nn.Module] = nn.LayerNorm
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu

    def setup(self) -> None:
        if len(self.depth) != len(self.num_heads):
            raise ValueError("`depth` and `num_heads` must share the same length.")
        if len(self.patch_size) != self.space:
            raise ValueError("`patch_size` dimensionality must match `space`.")
        if len(self.window_size) != self.space:
            raise ValueError("`window_size` dimensionality must match `space`.")
        if len(self.resolution) != self.space:
            raise ValueError("`resolution` dimensionality must match `space`.")

        mask_spaces: Optional[Sequence[bool]]
        if self.merge_mask is None:
            mask_spaces = None
        elif isinstance(self.merge_mask, bool):
            mask_spaces = (self.merge_mask,) * self.space
        else:
            if len(self.merge_mask) != self.space:
                raise ValueError("`merge_mask` must provide one flag per spatial axis.")
            mask_spaces = tuple(self.merge_mask)

        self.num_layers = len(self.depth)
        grid_track = []

        self.patch_embed = PatchEmbed(
            space=self.space,
            in_resolution=self.resolution,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.dim,
            flatten=False,
            use_conv=self.use_conv,
            mlp_ratio=self.mlp_ratio,
            mlp_depth=1,
            norm_layer=self.norm_layer,
        )

        current_dim = self.dim
        current_grid = tuple(
            int(ceil(res / patch)) for res, patch in zip(self.resolution, self.patch_size)
        )
        grid_track.append(current_grid)

        if self.use_abs_pos:
            self.abs_pos = PositionalEmbedding(
                dim=current_dim,
                grid_size=current_grid,
                learnable=True,
                init_type="rand",
            )

        total_blocks = sum(self.depth)
        if self.drop_path_rate > 0.0:
            if total_blocks == 1:
                drop_schedule = [0.0]
            else:
                drop_schedule = [0.0 + i * (self.drop_path_rate - 0.0) / (total_blocks - 1) for i in range(total_blocks)]
        else:
            drop_schedule = [0.0] * total_blocks

        layers = []
        mergers = []
        offset = 0

        for stage_depth, heads in zip(self.depth, self.num_heads):
            stage_drop = tuple(float(rate) for rate in drop_schedule[offset : offset + stage_depth])
            offset += stage_depth

            layer = SwinLayer(
                space=self.space,
                dim=current_dim,
                depth=stage_depth,
                num_heads=heads,
                grid_size=current_grid,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_path=stage_drop if self.drop_path_rate > 0.0 else 0.0,
                mlp_drop=self.mlp_drop,
                attn_drop=self.attn_drop,
                norm_layer=self.norm_layer,
                activation=self.activation,
            )
            layers.append(layer)

            merge = PatchMerging(
                space=self.space,
                dim=current_dim,
                in_resolution=current_grid,
                c_multiplier=2,
                norm_layer=self.norm_layer,
                mask_spaces=mask_spaces,
            )
            mergers.append(merge)

            current_dim = current_dim * merge.c_multiplier
            if mask_spaces is None:
                merge_flags = (True,) * self.space
            else:
                merge_flags = mask_spaces
            current_grid = tuple(
                int(ceil(dim / 2)) if use else dim
                for dim, use in zip(current_grid, merge_flags)
            )
            grid_track.append(current_grid)

        self.grid_sizes = tuple(grid_track)
        self.swin_layers = tuple(layers)
        self.patch_merges = tuple(mergers)

        self.head_norm = self.norm_layer()
        self.head_dropout = nn.Dropout(self.head_drop) if self.head_drop > 0.0 else None
        self.head_dense = nn.Dense(self.num_classes)

    def forward_features(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        x = jnp.moveaxis(x, 1, -1)
        x = self.patch_embed(x, deterministic=deterministic)
        if self.use_abs_pos:
            x = self.abs_pos(x)

        for layer, merge in zip(self.swin_layers, self.patch_merges):
            x = layer(x, deterministic=deterministic)
            x = merge(x, deterministic=deterministic)
        return x

    def forward_head(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = self.head_norm(x)
        x = self.activation(x)
        x = jnp.mean(x, axis=1)
        if self.head_dropout is not None:
            x = self.head_dropout(x, deterministic=deterministic)
        x = self.head_dense(x)
        return x

    def __call__(self, x: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        features = self.forward_features(x, deterministic=deterministic)
        logits = self.forward_head(features, deterministic=deterministic)
        return logits


__all__ = ["NDSwinClassifier"]
