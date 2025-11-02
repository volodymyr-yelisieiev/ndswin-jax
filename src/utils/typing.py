"""Shared typing aliases for type checkers and documentation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jnp.ndarray
PRNGKey = jax.Array

__all__ = ["Array", "PRNGKey", "jnp"]

