"""Pytest configuration and fixtures."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Set random seeds for reproducibility
np.random.seed(42)


@pytest.fixture
def rng():
    """Create a JAX random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_2d_input():
    """Create sample 2D input (image)."""
    return jnp.ones((2, 3, 32, 32))  # (B, C, H, W)


@pytest.fixture
def sample_3d_input():
    """Create sample 3D input (video/volume)."""
    return jnp.ones((2, 3, 8, 32, 32))  # (B, C, D, H, W)


@pytest.fixture
def sample_4d_input():
    """Create sample 4D input."""
    return jnp.ones((2, 3, 4, 8, 8, 8))  # (B, C, T, D, H, W)


@pytest.fixture
def config_2d():
    """Create 2D Swin config."""
    from ndswin.config import NDSwinConfig

    return NDSwinConfig(
        num_dims=2,
        patch_size=(4, 4),
        window_size=(4, 4),
        in_channels=3,
        embed_dim=48,
        depths=(2, 2),
        num_heads=(3, 6),
        num_classes=10,
        drop_path_rate=0.0,
        drop_rate=0.0,
    )


@pytest.fixture
def config_3d():
    """Create 3D Swin config."""
    from ndswin.config import NDSwinConfig

    return NDSwinConfig(
        num_dims=3,
        patch_size=(2, 4, 4),
        window_size=(2, 4, 4),
        in_channels=3,
        embed_dim=48,
        depths=(2, 2),
        num_heads=(3, 6),
        num_classes=10,
    )


@pytest.fixture
def training_config():
    """Create training config."""
    from ndswin.config import TrainingConfig

    return TrainingConfig(
        epochs=10,  # Must be > warmup_epochs (default is 5)
        batch_size=4,
        learning_rate=1e-4,
        num_classes=10,
    )


# Configure JAX for testing
def pytest_configure(config):
    """Configure pytest."""
    # Disable JIT for easier debugging in tests
    # jax.config.update("jax_disable_jit", True)
    pass
