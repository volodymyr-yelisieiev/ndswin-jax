"""Tests for building blocks."""

import jax
import jax.numpy as jnp

from ndswin.core.blocks import BasicLayer, DropPath, MLPBlock, SwinTransformerBlock
from ndswin.core.patch_embed import PatchEmbed, PatchMerging


class TestMLPBlock:
    """Tests for MLPBlock."""

    def test_forward(self, rng):
        """Test MLP forward pass."""
        mlp = MLPBlock(
            hidden_dim=192,
            out_dim=48,
        )

        x = jnp.ones((4, 16, 48))
        variables = mlp.init(rng, x, deterministic=True)
        output = mlp.apply(variables, x, deterministic=True)

        assert output.shape == x.shape

    def test_expansion_ratio(self, rng):
        """Test MLP with expansion ratio."""
        mlp = MLPBlock(
            hidden_dim=192,  # 4x expansion
            out_dim=48,
        )

        x = jnp.ones((4, 16, 48))
        variables = mlp.init(rng, x, deterministic=True)

        # Check hidden dimension - MLPBlock uses fc1/fc2 as layer names
        fc1_kernel = variables["params"]["fc1"]["kernel"]
        assert fc1_kernel.shape[1] == 192


class TestDropPath:
    """Tests for DropPath."""

    def test_deterministic(self, rng):
        """Test DropPath in deterministic mode."""
        drop_path = DropPath(drop_prob=0.1)

        x = jnp.ones((4, 16, 48))
        variables = drop_path.init(rng, x, deterministic=True)
        output = drop_path.apply(variables, x, deterministic=True)

        # In deterministic mode, output should equal input
        assert jnp.allclose(output, x)

    def test_stochastic(self, rng):
        """Test DropPath in stochastic mode."""
        drop_path = DropPath(drop_prob=0.5)

        x = jnp.ones((4, 16, 48))
        variables = drop_path.init(rng, x, deterministic=False)

        # Run multiple times and check that some paths are dropped
        outputs = []
        for i in range(10):
            output = drop_path.apply(
                variables, x, deterministic=False, rngs={"dropout": jax.random.PRNGKey(i)}
            )
            outputs.append(output)

        # Outputs should vary
        assert not all(jnp.allclose(outputs[0], o) for o in outputs[1:])


class TestSwinTransformerBlock:
    """Tests for SwinTransformerBlock."""

    def test_forward_no_shift(self, rng):
        """Test block without shift."""
        block = SwinTransformerBlock(
            dim=48,
            num_heads=3,
            window_size=(4, 4),
            shift_size=(0, 0),
            mlp_ratio=4.0,
        )

        x = jnp.ones((2, 8, 8, 48))
        variables = block.init(rng, x, deterministic=True)
        output = block.apply(variables, x, deterministic=True)

        assert output.shape == x.shape

    def test_forward_with_shift(self, rng):
        """Test block with shift."""
        block = SwinTransformerBlock(
            dim=48,
            num_heads=3,
            window_size=(4, 4),
            shift_size=(2, 2),
            mlp_ratio=4.0,
        )

        x = jnp.ones((2, 8, 8, 48))
        variables = block.init(rng, x, deterministic=True)
        output = block.apply(variables, x, deterministic=True)

        assert output.shape == x.shape

    def test_3d_block(self, rng):
        """Test 3D transformer block."""
        block = SwinTransformerBlock(
            dim=48,
            num_heads=3,
            window_size=(2, 4, 4),
            shift_size=(1, 2, 2),
            mlp_ratio=4.0,
        )

        x = jnp.ones((1, 4, 8, 8, 48))
        variables = block.init(rng, x, deterministic=True)
        output = block.apply(variables, x, deterministic=True)

        assert output.shape == x.shape


class TestPatchEmbed:
    """Tests for PatchEmbed."""

    def test_2d_embed(self, rng):
        """Test 2D patch embedding."""
        embed = PatchEmbed(
            patch_size=(4, 4),
            embed_dim=48,
        )

        # (B, C, H, W)
        x = jnp.ones((2, 3, 32, 32))
        variables = embed.init(rng, x)
        output = embed.apply(variables, x)

        # (B, H/4, W/4, embed_dim)
        assert output.shape == (2, 8, 8, 48)

    def test_3d_embed(self, rng):
        """Test 3D patch embedding."""
        embed = PatchEmbed(
            patch_size=(2, 4, 4),
            embed_dim=48,
        )

        # (B, C, D, H, W)
        x = jnp.ones((1, 3, 8, 32, 32))
        variables = embed.init(rng, x)
        output = embed.apply(variables, x)

        # (B, D/2, H/4, W/4, embed_dim)
        assert output.shape == (1, 4, 8, 8, 48)


class TestPatchMerging:
    """Tests for PatchMerging."""

    def test_2d_merge(self, rng):
        """Test 2D patch merging."""
        merge = PatchMerging(dim=96)

        # (B, H, W, C)
        x = jnp.ones((2, 8, 8, 48))
        variables = merge.init(rng, x)
        output = merge.apply(variables, x)

        # (B, H/2, W/2, out_dim)
        assert output.shape == (2, 4, 4, 96)

    def test_3d_merge(self, rng):
        """Test 3D patch merging."""
        merge = PatchMerging(dim=96)

        # (B, D, H, W, C)
        x = jnp.ones((1, 4, 8, 8, 48))
        variables = merge.init(rng, x)
        output = merge.apply(variables, x)

        # (B, D/2, H/2, W/2, out_dim) or similar
        assert output.shape[-1] == 96
        # Spatial dimensions should be reduced
        assert output.shape[1] <= x.shape[1]


class TestBasicLayer:
    """Tests for BasicLayer (stage)."""

    def test_basic_layer(self, rng):
        """Test basic layer (stage)."""
        layer = BasicLayer(
            dim=48,
            depth=2,
            num_heads=3,
            window_size=(4, 4),
            mlp_ratio=4.0,
            downsample=True,
            out_dim=96,
        )

        x = jnp.ones((2, 8, 8, 48))
        variables = layer.init(rng, x, deterministic=True)
        output = layer.apply(variables, x, deterministic=True)

        # With downsample, spatial is halved
        assert output.shape == (2, 4, 4, 96)

    def test_no_downsample(self, rng):
        """Test layer without downsampling."""
        layer = BasicLayer(
            dim=48,
            depth=2,
            num_heads=3,
            window_size=(4, 4),
            mlp_ratio=4.0,
            downsample=False,
        )

        x = jnp.ones((2, 8, 8, 48))
        variables = layer.init(rng, x, deterministic=True)
        output = layer.apply(variables, x, deterministic=True)

        # Without downsample, shape is preserved
        assert output.shape == x.shape
