"""Absolute positional encodings."""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


def _sincos_embedding(grid_size: Sequence[int], dim: int) -> jnp.ndarray:
    space = len(grid_size)
    if dim % (2 * space) != 0:
        raise ValueError(
            "`dim` must be divisible by twice the number of spatial dimensions for sinusoidal embeddings."
        )

    per_axis = dim // space
    half = per_axis // 2
    if half == 0:
        raise ValueError("Positional dimension per axis must be at least 2.")

    coords = jnp.stack(jnp.meshgrid(*[jnp.arange(n) for n in grid_size], indexing="ij"), axis=-1)

    embeddings = []
    freq = 1.0 / (10000 ** (jnp.arange(half) / half))
    for axis in range(space):
        ang = coords[..., axis][..., None] * freq
        embeddings.append(jnp.sin(ang))
        embeddings.append(jnp.cos(ang))

    return jnp.concatenate(embeddings, axis=-1)


class PositionalEmbedding(nn.Module):
    """Absolute positional embeddings with sinusoidal or learnable parameters."""

    dim: int
    grid_size: Sequence[int]
    learnable: bool = False
    init_type: str = "sincos"

    def setup(self) -> None:
        grid = tuple(self.grid_size)
        if self.learnable:
            self.pos_embed = self.param(
                "pos_embed",
                lambda rng, shape: jnp.zeros(shape, dtype=jnp.float32),
                (1, *grid, self.dim),
            )
        else:
            if self.init_type == "sincos":
                init_value = _sincos_embedding(grid, self.dim)[None]
            elif self.init_type == "rand":
                init_value = jnp.zeros((1, *grid, self.dim), dtype=jnp.float32)
            else:
                raise ValueError(f"Unsupported init_type: {self.init_type}")
            self.pos_embed = self.variable("constants", "pos_embed", lambda rng: init_value)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        reshaped = False
        if x.ndim == 3:
            b, tokens, c = x.shape
            expected = int(jnp.prod(jnp.array(self.grid_size)))
            if tokens != expected:
                raise ValueError(
                    f"Token length {tokens} does not match positional grid product {expected}."
                )
            x = x.reshape(b, *self.grid_size, c)
            reshaped = True

        pos = self.pos_embed if self.learnable else self.pos_embed.value
        x = x + pos
        if reshaped:
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x


__all__ = ["PositionalEmbedding"]

