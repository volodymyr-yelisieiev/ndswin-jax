"""Small-shape tests for N-dimensional NDSwin model paths."""

import jax
import jax.numpy as jnp
import pytest

from ndswin.config import NDSwinConfig
from ndswin.models.swin import NDSwinTransformer


@pytest.mark.parametrize(
    ("num_dims", "input_shape", "patch_size", "window_size", "depths", "num_heads"),
    [
        (1, (2, 1, 16), (2,), (4,), (1, 1), (1, 2)),
        (2, (2, 1, 16, 16), (2, 2), (4, 4), (1, 1), (1, 2)),
        (3, (1, 1, 8, 8, 8), (2, 2, 2), (2, 2, 2), (1, 1), (1, 2)),
        (4, (1, 1, 4, 4, 4, 4), (2, 2, 2, 2), (2, 2, 2, 2), (1,), (1,)),
    ],
)
def test_ndswin_forward_small_nd_configs(
    num_dims, input_shape, patch_size, window_size, depths, num_heads
):
    """Forward passes should work for small 1D, 2D, 3D, and 4D configs."""
    config = NDSwinConfig(
        num_dims=num_dims,
        in_channels=1,
        patch_size=patch_size,
        window_size=window_size,
        embed_dim=8,
        depths=depths,
        num_heads=num_heads,
        num_classes=3,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    )
    model = NDSwinTransformer(config)
    x = jnp.ones(input_shape, dtype=jnp.float32)

    variables = model.init(jax.random.PRNGKey(0), x, deterministic=True)
    logits = model.apply(variables, x, deterministic=True)

    assert logits.shape == (input_shape[0], 3)
    assert jnp.all(jnp.isfinite(logits))


def test_ndswin_validates_input_channels():
    """The model should fail early when input channels do not match config."""
    config = NDSwinConfig(
        num_dims=2,
        in_channels=1,
        patch_size=(2, 2),
        window_size=(2, 2),
        embed_dim=8,
        depths=(1,),
        num_heads=(1,),
        num_classes=2,
    )
    model = NDSwinTransformer(config)
    x = jnp.ones((1, 3, 8, 8), dtype=jnp.float32)

    with pytest.raises(ValueError, match="Expected 1 input channels"):
        model.init(jax.random.PRNGKey(0), x, deterministic=True)
