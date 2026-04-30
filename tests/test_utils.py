"""Tests for core utility functions."""

import jax.numpy as jnp

from ndswin.core.utils import get_relative_position_index, to_ntuple


class TestToNTuple:
    """Tests for to_ntuple function."""

    def test_int_to_tuple(self):
        """Test converting int to tuple."""
        result = to_ntuple(3, 2)  # Convert 3 to a 2-tuple
        assert result == (3, 3)

    def test_tuple_passthrough(self):
        """Test tuple is passed through."""
        result = to_ntuple((1, 2), 2)  # Pass through tuple of length 2
        assert result == (1, 2)

    def test_list_to_tuple(self):
        """Test list is converted to tuple."""
        result = to_ntuple([1, 2, 3], 3)  # Convert list to tuple
        assert result == (1, 2, 3)

    def test_n_tuple_dimensions(self):
        """Test various dimensions."""
        for n in [1, 2, 3, 4, 5]:
            result = to_ntuple(7, n)  # Convert 7 to n-tuple
            assert len(result) == n
            assert all(x == 7 for x in result)


class TestRelativePositionIndex:
    """Tests for relative position index computation."""

    def test_2d_window(self):
        """Test 2D window position index."""
        window_size = (4, 4)
        index = get_relative_position_index(window_size)

        # Should have shape (16, 16) for 4x4 window
        assert index.shape == (16, 16)

        # Diagonal should all be the same (self-relative position is the same)
        diagonal = jnp.diag(index)
        assert jnp.all(diagonal == diagonal[0])

    def test_3d_window(self):
        """Test 3D window position index."""
        window_size = (2, 4, 4)
        index = get_relative_position_index(window_size)

        # Should have shape (32, 32) for 2x4x4 window
        expected_size = 2 * 4 * 4
        assert index.shape == (expected_size, expected_size)

    def test_values_in_range(self):
        """Test that index values are valid."""
        window_size = (4, 4)
        index = get_relative_position_index(window_size)

        # Max index should be based on table size
        num_relative = (2 * 4 - 1) * (2 * 4 - 1)
        assert index.max() < num_relative
        assert index.min() >= 0
