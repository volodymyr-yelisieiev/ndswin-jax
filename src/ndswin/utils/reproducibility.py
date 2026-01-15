"""Reproducibility utilities for NDSwin-JAX.

This module provides utilities for ensuring reproducible experiments
through proper random seed management and deterministic operations.
"""

import os
import random
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from ndswin.types import PRNGKey


def set_seed(seed: int, deterministic: bool = False) -> PRNGKey:
    """Set random seeds for reproducibility.

    This sets seeds for Python's random module, NumPy, and creates a JAX PRNG key.
    Note that full determinism in JAX requires additional configuration.

    Args:
        seed: Random seed value.
        deterministic: Whether to enable deterministic mode.
            This may impact performance.

    Returns:
        JAX PRNG key initialized with the seed.
    """
    # Python random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # JAX PRNG key
    key = jax.random.PRNGKey(seed)

    # Enable deterministic operations if requested
    if deterministic:
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
        )

    return key


def create_rng_keys(
    key: PRNGKey,
    names: list[str],
) -> dict[str, PRNGKey]:
    """Create named PRNG keys from a base key.

    This is useful for separating randomness between different parts of the model
    (e.g., dropout, initialization).

    Args:
        key: Base PRNG key.
        names: List of names for the generated keys.

    Returns:
        Dictionary mapping names to PRNG keys.
    """
    keys = jax.random.split(key, len(names) + 1)
    return {name: keys[i + 1] for i, name in enumerate(names)}


def split_key(key: PRNGKey, num: int = 2) -> tuple[PRNGKey, ...]:
    """Split a PRNG key into multiple keys.

    Args:
        key: PRNG key to split.
        num: Number of keys to generate.

    Returns:
        Tuple of PRNG keys.
    """
    return tuple(jax.random.split(key, num))


def get_deterministic_key(seed: int) -> PRNGKey:
    """Get a deterministic PRNG key from an integer seed.

    Args:
        seed: Integer seed.

    Returns:
        JAX PRNG key.
    """
    return jax.random.PRNGKey(seed)


def fold_in(key: PRNGKey, data: int) -> PRNGKey:
    """Fold an integer into a PRNG key.

    This is useful for creating unique keys per example in a batch.

    Args:
        key: Base PRNG key.
        data: Integer to fold in.

    Returns:
        New PRNG key.
    """
    return jax.random.fold_in(key, data)


class PRNGSequence:
    """A sequence of PRNG keys for use in loops.

    This class provides an iterator interface for generating PRNG keys
    in a loop without having to manually split and track keys.

    Example:
        >>> rng_seq = PRNGSequence(0)
        >>> for i in range(10):
        ...     key = next(rng_seq)
        ...     # Use key for random operations
    """

    def __init__(self, seed_or_key: int | PRNGKey) -> None:
        """Initialize PRNG sequence.

        Args:
            seed_or_key: Integer seed or existing PRNG key.
        """
        if isinstance(seed_or_key, int):
            self._key = jax.random.PRNGKey(seed_or_key)
        else:
            self._key = seed_or_key

    def __next__(self) -> PRNGKey:
        """Get next PRNG key in sequence.

        Returns:
            Next PRNG key.
        """
        keys = jax.random.split(self._key)
        self._key = keys[0]
        return keys[1]

    def __iter__(self) -> "PRNGSequence":
        """Return iterator.

        Returns:
            Self.
        """
        return self

    def take(self, num: int) -> list[PRNGKey]:
        """Take multiple PRNG keys.

        Args:
            num: Number of keys to take.

        Returns:
            List of PRNG keys.
        """
        return [next(self) for _ in range(num)]


class KeyGenerator:
    """Generator for PRNG keys with named subkeys.

    This class manages PRNG keys for different components of a model
    or training loop, ensuring proper key splitting and tracking.

    Example:
        >>> keygen = KeyGenerator(42)
        >>> params_key = keygen.get("params")
        >>> dropout_key = keygen.get("dropout")
    """

    def __init__(self, seed: int) -> None:
        """Initialize KeyGenerator.

        Args:
            seed: Random seed.
        """
        self._base_key = jax.random.PRNGKey(seed)
        self._counter = 0
        self._named_keys: dict[str, PRNGKey] = {}

    def get(self, name: str | None = None) -> PRNGKey:
        """Get a PRNG key.

        If a name is provided and was used before, returns the same key.
        Otherwise, generates a new key.

        Args:
            name: Optional name for the key.

        Returns:
            PRNG key.
        """
        if name is not None and name in self._named_keys:
            return self._named_keys[name]

        key = jax.random.fold_in(self._base_key, self._counter)
        self._counter += 1

        if name is not None:
            self._named_keys[name] = key

        return key

    def split(self, num: int = 2) -> tuple[PRNGKey, ...]:
        """Split the current key.

        Args:
            num: Number of keys to generate.

        Returns:
            Tuple of PRNG keys.
        """
        key = self.get()
        return tuple(jax.random.split(key, num))

    def reset(self, seed: int | None = None) -> None:
        """Reset the generator.

        Args:
            seed: New seed. If None, uses original seed.
        """
        if seed is not None:
            self._base_key = jax.random.PRNGKey(seed)
        self._counter = 0
        self._named_keys.clear()


def make_reproducible_init(init_fn: Callable) -> Callable:
    """Decorator to make initialization functions reproducible.

    This decorator ensures that an initialization function uses a
    consistent PRNG key for reproducibility.

    Args:
        init_fn: Initialization function that takes a PRNG key.

    Returns:
        Wrapped function with reproducible initialization.

    Example:
        >>> @make_reproducible_init
        ... def init_model(key, config):
        ...     return model.init(key, dummy_input)
    """

    def wrapped(seed: int, *args: Any, **kwargs: Any) -> Any:
        key = jax.random.PRNGKey(seed)
        return init_fn(key, *args, **kwargs)

    return wrapped


def check_reproducibility(
    fn: Callable,
    key: PRNGKey,
    *args: Any,
    num_runs: int = 3,
    **kwargs: Any,
) -> bool:
    """Check if a function produces reproducible results.

    This runs the function multiple times with the same key and checks
    if the outputs are identical.

    Args:
        fn: Function to check.
        key: PRNG key to use.
        *args: Positional arguments for fn.
        num_runs: Number of runs to compare.
        **kwargs: Keyword arguments for fn.

    Returns:
        True if function is reproducible, False otherwise.
    """
    results = []
    for _ in range(num_runs):
        result = fn(key, *args, **kwargs)
        results.append(result)

    # Check all results are equal
    first = results[0]
    for result in results[1:]:
        if isinstance(first, jnp.ndarray):
            if not jnp.allclose(first, result):
                return False
        elif isinstance(first, dict):
            for k in first:
                if not jnp.allclose(first[k], result[k]):
                    return False
        elif first != result:
            return False

    return True


def get_random_state() -> dict[str, object]:
    """Get current random state for all RNGs.

    Returns:
        Dictionary with random states.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def set_random_state(state: dict[str, Any]) -> None:
    """Set random state for all RNGs.

    Args:
        state: Dictionary with random states from get_random_state.
    """
    if "python" in state:
        random.setstate(cast(tuple, state["python"]))
    if "numpy" in state:
        np.random.set_state(cast(Any, state["numpy"]))
