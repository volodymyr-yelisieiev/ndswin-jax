"""Tests for window operations."""

import jax
import jax.numpy as jnp

from ndswin.core.window_ops import (
    create_attention_mask,
    cyclic_shift,
    pad_for_window,
    partition_windows,
    reverse_partition_windows,
)


class TestPartitionWindows:
    """Tests for window partitioning."""

    def test_2d_partition(self):
        """Test 2D window partitioning."""
        # Create input: (B, H, W, C)
        x = jnp.arange(64).reshape(1, 8, 8, 1).astype(jnp.float32)
        window_size = (4, 4)

        windows = partition_windows(x, window_size)

        # Should have 4 windows: (1*2*2, 4, 4, 1)
        assert windows.shape == (4, 4, 4, 1)

    def test_3d_partition(self):
        """Test 3D window partitioning."""
        # Create input: (B, D, H, W, C)
        x = jnp.ones((1, 4, 8, 8, 16))
        window_size = (2, 4, 4)

        windows = partition_windows(x, window_size)

        # 2*2*2 = 8 windows, each with (2, 4, 4, 16) shape
        assert windows.shape == (8, 2, 4, 4, 16)

    def test_partition_reversible(self):
        """Test that partition is reversible."""
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 8, 32))
        window_size = (4, 4)
        spatial_shape = (8, 8)

        windows = partition_windows(x, window_size)
        reversed_x = reverse_partition_windows(windows, spatial_shape, window_size)

        assert jnp.allclose(x, reversed_x)

    def test_3d_partition_reversible(self):
        """Test that 3D partition is reversible."""
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 8, 8, 48))
        window_size = (2, 4, 4)
        spatial_shape = (4, 8, 8)

        windows = partition_windows(x, window_size)
        reversed_x = reverse_partition_windows(windows, spatial_shape, window_size)

        assert jnp.allclose(x, reversed_x)


class TestCyclicShift:
    """Tests for cyclic shift operations."""

    def test_2d_shift(self):
        """Test 2D cyclic shift."""
        x = jnp.arange(16).reshape(1, 4, 4, 1).astype(jnp.float32)
        shift_size = (2, 2)

        shifted = cyclic_shift(x, shift_size)

        # Should have same shape
        assert shifted.shape == x.shape

        # Check that values are shifted
        assert not jnp.allclose(x, shifted)

    def test_shift_reversible(self):
        """Test that shift is reversible."""
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 8, 8, 32))
        shift_size = (4, 4)

        shifted = cyclic_shift(x, shift_size)
        # Shift by negative amount to reverse
        reverse_shift = tuple(-s for s in shift_size)
        unshifted = cyclic_shift(shifted, reverse_shift)

        assert jnp.allclose(x, unshifted)

    def test_3d_shift(self):
        """Test 3D cyclic shift."""
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 8, 8, 16))
        shift_size = (1, 2, 2)

        shifted = cyclic_shift(x, shift_size)
        reverse_shift = tuple(-s for s in shift_size)
        unshifted = cyclic_shift(shifted, reverse_shift)

        assert jnp.allclose(x, unshifted)


class TestAttentionMask:
    """Tests for attention mask creation."""

    def test_2d_mask_shape(self):
        """Test 2D attention mask shape."""
        spatial_shape = (8, 8)
        window_size = (4, 4)
        shift_size = (2, 2)

        mask = create_attention_mask(spatial_shape, window_size, shift_size)

        # Number of windows
        num_windows = (8 // 4) * (8 // 4)  # 4
        window_area = 4 * 4  # 16

        assert mask.shape == (num_windows, window_area, window_area)

    def test_no_shift_mask(self):
        """Test mask with no shift is all True (no masking needed)."""
        spatial_shape = (8, 8)
        window_size = (4, 4)
        shift_size = (0, 0)

        mask = create_attention_mask(spatial_shape, window_size, shift_size)

        # With no shift, mask should be all True (allow attention everywhere)
        assert jnp.all(mask)


class TestPadForWindow:
    """Tests for window padding."""

    def test_no_padding_needed(self):
        """Test when no padding is needed."""
        x = jnp.ones((1, 8, 8, 32))
        window_size = (4, 4)

        padded, original_shape = pad_for_window(x, window_size)

        assert padded.shape == x.shape
        assert original_shape == (8, 8)

    def test_padding_needed(self):
        """Test when padding is needed."""
        x = jnp.ones((1, 7, 7, 32))
        window_size = (4, 4)

        padded, original_shape = pad_for_window(x, window_size)

        # Should be padded to 8x8
        assert padded.shape == (1, 8, 8, 32)
        assert original_shape == (7, 7)

    def test_3d_padding(self):
        """Test 3D padding."""
        x = jnp.ones((1, 3, 7, 7, 32))
        window_size = (2, 4, 4)

        padded, original_shape = pad_for_window(x, window_size)

        # Should be padded to 4x8x8
        assert padded.shape == (1, 4, 8, 8, 32)
        assert original_shape == (3, 7, 7)
