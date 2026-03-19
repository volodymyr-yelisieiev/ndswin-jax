"""Tests for reproducibility utilities."""

import os
import random
import unittest.mock as mock

import jax
import jax.numpy as jnp
import numpy as np

from ndswin.utils.reproducibility import (
    KeyGenerator,
    PRNGSequence,
    check_reproducibility,
    create_rng_keys,
    fold_in,
    get_deterministic_key,
    get_random_state,
    make_reproducible_init,
    set_random_state,
    set_seed,
    split_key,
)


@mock.patch.dict(os.environ, {"XLA_FLAGS": ""})
def test_set_seed(monkeypatch):
    """Test seed execution boundaries."""
    key = set_seed(42, deterministic=True)
    assert isinstance(key, jax.Array)
    
    # Check if deterministic flag set exactly
    assert "xla_gpu_deterministic_ops=true" in os.environ["XLA_FLAGS"]


def test_create_rng_keys():
    """Test mapping keys to dictionary names."""
    base = get_deterministic_key(42)
    keys = create_rng_keys(base, ["dropout", "params"])
    
    assert "dropout" in keys
    assert "params" in keys
    assert isinstance(keys["dropout"], jax.Array)


def test_split_key():
    """Test standard splitting."""
    base = get_deterministic_key(42)
    k1, k2, k3 = split_key(base, 3)
    assert not jnp.allclose(k1, k2)


def test_fold_in():
    """Test fold in uniqueness."""
    base = get_deterministic_key(42)
    f1 = fold_in(base, 1)
    f2 = fold_in(base, 2)
    assert not jnp.allclose(f1, f2)


def test_prng_sequence():
    """Test PRNG sequence generator logic."""
    seq = PRNGSequence(42)
    k1 = next(seq)
    k2 = next(seq)
    assert not jnp.allclose(k1, k2)
    
    keys = seq.take(3)
    assert len(keys) == 3


def test_prng_sequence_with_key():
    """Test PRNG sequence initializing from an existing key."""
    key = get_deterministic_key(42)
    seq = PRNGSequence(key)
    k1 = next(seq)
    assert isinstance(k1, jax.Array)


def test_key_generator():
    """Test robust KeyGenerator operations."""
    gen = KeyGenerator(42)
    
    # Name fetching
    k1 = gen.get("params")
    k2 = gen.get("params")
    # Same name should return identically same key (cached)
    assert jnp.allclose(k1, k2)
    
    # Generic un-named fetch
    un1 = gen.get()
    un2 = gen.get()
    assert not jnp.allclose(un1, un2)
    
    # Splitting
    s1, s2 = gen.split(2)
    assert not jnp.allclose(s1, s2)
    
    # Reset
    gen.reset(24)
    # The name cache should be empty so we get a completely new one and not the one for 42
    k3 = gen.get("params")
    assert not jnp.allclose(k1, k3)


def test_make_reproducible_init():
    """Test reproducible initialization wrappers."""
    @make_reproducible_init
    def init_fn(key, config_val):
        return jax.random.normal(key, (2, 2)) * config_val
    
    out1 = init_fn(42, 1.0)
    out2 = init_fn(42, 1.0)
    assert jnp.allclose(out1, out2)


def test_check_reproducibility():
    """Test reproducibility execution checker."""
    def reproducible_fn(key, x):
        return jax.random.normal(key, (1,)) * x
    
    def non_reproducible_fn(key, x):
        return jax.random.normal(jax.random.PRNGKey(np.random.randint(100)), (1,)) * x
        
    key = get_deterministic_key(42)
    assert check_reproducibility(reproducible_fn, key, 2.0)
    
    np.random.seed(42)
    assert not check_reproducibility(non_reproducible_fn, key, 2.0)


def test_check_reproducibility_dict():
    """Test reproducibility execution checker across dict structures."""
    def reproducible_fn(key):
        return {"a": jax.random.normal(key, (1,)), "b": jnp.array([1, 2])}
        
    key = get_deterministic_key(42)
    assert check_reproducibility(reproducible_fn, key)


def test_check_reproducibility_scalar():
    """Test reproducibility execution checker across scalars."""
    def reproducible_fn(key):
        return 42
        
    key = get_deterministic_key(42)
    assert check_reproducibility(reproducible_fn, key)


def test_get_set_random_state():
    """Test capturing and setting intrinsic python random states."""
    random.seed(42)
    np.random.seed(42)
    
    state = get_random_state()
    
    val1 = random.random()
    val2 = np.random.rand()
    
    set_random_state(state)
    
    val3 = random.random()
    val4 = np.random.rand()
    
    assert val1 == val3
    assert val2 == val4
