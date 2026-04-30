"""Tests for attention modules."""

import jax
import jax.numpy as jnp

from ndswin.core.attention import ShiftedWindowAttention, WindowAttention


class TestWindowAttention:
    """Tests for WindowAttention module."""

    def test_basic_forward(self, rng):
        """Test basic forward pass."""
        attention = WindowAttention(
            num_heads=3,
            window_size=(4, 4),
        )

        # (num_windows * batch, window_size^2, dim)
        x = jnp.ones((4, 16, 48))

        variables = attention.init(rng, x, deterministic=True)
        output = attention.apply(variables, x, deterministic=True)

        assert output.shape == x.shape

    def test_with_mask(self, rng):
        """Test attention with mask."""
        attention = WindowAttention(
            num_heads=3,
            window_size=(4, 4),
        )

        x = jnp.ones((4, 16, 48))
        mask = jnp.zeros((4, 1, 16, 16))

        variables = attention.init(rng, x, deterministic=True)
        output = attention.apply(variables, x, mask=mask, deterministic=True)

        assert output.shape == x.shape

    def test_3d_window(self, rng):
        """Test 3D window attention."""
        attention = WindowAttention(
            num_heads=3,
            window_size=(2, 4, 4),
        )

        # (num_windows * batch, window_size prod, dim)
        x = jnp.ones((8, 32, 48))

        variables = attention.init(rng, x, deterministic=True)
        output = attention.apply(variables, x, deterministic=True)

        assert output.shape == x.shape

    def test_relative_position_bias(self, rng):
        """Test that relative position bias is applied."""
        attention = WindowAttention(
            num_heads=3,
            window_size=(4, 4),
            qkv_bias=True,
        )

        x = jnp.ones((4, 16, 48))

        variables = attention.init(rng, x, deterministic=True)

        # Check relative position bias table exists
        assert "relative_position_bias_table" in variables["params"]

        output = attention.apply(variables, x, deterministic=True)
        assert output.shape == x.shape


class TestShiftedWindowAttention:
    """Tests for ShiftedWindowAttention module."""

    def test_no_shift(self, rng):
        """Test without shift (regular window attention)."""
        attention = ShiftedWindowAttention(
            num_heads=3,
            window_size=(4, 4),
            shift_size=(0, 0),
        )

        # (B, H, W, C)
        x = jnp.ones((2, 8, 8, 48))

        variables = attention.init(rng, x, deterministic=True)
        output = attention.apply(variables, x, deterministic=True)

        assert output.shape == x.shape

    def test_with_shift(self, rng):
        """Test with shift (SW-MSA)."""
        attention = ShiftedWindowAttention(
            num_heads=3,
            window_size=(4, 4),
            shift_size=(2, 2),
        )

        x = jnp.ones((2, 8, 8, 48))

        variables = attention.init(rng, x, deterministic=True)
        output = attention.apply(variables, x, deterministic=True)

        assert output.shape == x.shape

    def test_3d_shifted_attention(self, rng):
        """Test 3D shifted window attention."""
        attention = ShiftedWindowAttention(
            num_heads=3,
            window_size=(2, 4, 4),
            shift_size=(1, 2, 2),
        )

        # (B, D, H, W, C)
        x = jnp.ones((1, 4, 8, 8, 48))

        variables = attention.init(rng, x, deterministic=True)
        output = attention.apply(variables, x, deterministic=True)

        assert output.shape == x.shape

    def test_equivariance(self, rng):
        """Test that output changes with input changes."""
        attention = ShiftedWindowAttention(
            num_heads=3,
            window_size=(4, 4),
            shift_size=(2, 2),
        )

        x1 = jax.random.normal(rng, (1, 8, 8, 48))
        x2 = jax.random.normal(jax.random.PRNGKey(1), (1, 8, 8, 48))

        variables = attention.init(rng, x1, deterministic=True)
        out1 = attention.apply(variables, x1, deterministic=True)
        out2 = attention.apply(variables, x2, deterministic=True)

        # Outputs should be different
        assert not jnp.allclose(out1, out2)
