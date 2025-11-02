"""Stochastic depth utilities."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def stochastic_depth(x: Any, rate: float, *, deterministic: bool, rng: jax.Array) -> Any:
    """Apply per-sample stochastic depth (a.k.a. drop-path)."""

    if rate < 0.0 or rate > 1.0:
        raise ValueError("`rate` must lie in the [0, 1] interval.")

    if deterministic or rate == 0.0:
        return x

    keep_prob = 1.0 - rate
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = jax.random.bernoulli(rng, p=keep_prob, shape=shape)
    mask = mask.astype(x.dtype)
    if keep_prob > 0.0:
        mask = mask / keep_prob
    return x * mask
