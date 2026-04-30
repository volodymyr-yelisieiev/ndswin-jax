"""Tests for complete models."""

import jax
import jax.numpy as jnp
import optax

from ndswin.models.classifier import SwinClassifier, create_classifier
from ndswin.models.swin import NDSwinTransformer


class TestNDSwinTransformer:
    """Tests for NDSwinTransformer."""

    def test_2d_forward(self, rng, config_2d):
        """Test 2D model forward pass."""
        model = NDSwinTransformer(config=config_2d)

        x = jnp.ones((2, 3, 32, 32))
        variables = model.init(rng, x, deterministic=True)
        output = model.apply(variables, x, deterministic=True)

        assert output.shape == (2, config_2d.num_classes)

    def test_3d_forward(self, rng, config_3d):
        """Test 3D model forward pass."""
        model = NDSwinTransformer(config=config_3d)

        x = jnp.ones((1, 3, 8, 32, 32))
        variables = model.init(rng, x, deterministic=True)
        output = model.apply(variables, x, deterministic=True)

        assert output.shape == (1, config_3d.num_classes)

    def test_return_features(self, rng, config_2d):
        """Test returning intermediate features."""
        model = NDSwinTransformer(config=config_2d)

        x = jnp.ones((2, 3, 32, 32))
        variables = model.init(rng, x, deterministic=True, return_features=True)
        result = model.apply(variables, x, deterministic=True, return_features=True)

        assert "logits" in result
        assert "features" in result
        assert result["logits"].shape == (2, config_2d.num_classes)
        assert len(result["features"]) == len(config_2d.depths)

    def test_training_mode(self, rng, config_2d):
        """Test training mode with dropout."""
        model = NDSwinTransformer(config=config_2d)

        x = jnp.ones((2, 3, 32, 32))
        variables = model.init(rng, x, deterministic=False)

        # Run in training mode (deterministic=False)
        dropout_rng = jax.random.PRNGKey(1)
        output = model.apply(variables, x, deterministic=False, rngs={"dropout": dropout_rng})

        assert output.shape == (2, config_2d.num_classes)

    def test_gradient_flow(self, rng, config_2d):
        """Test that gradients flow through the model."""
        model = NDSwinTransformer(config=config_2d)

        x = jax.random.normal(rng, (2, 3, 32, 32))
        variables = model.init(rng, x, deterministic=True)

        def loss_fn(params):
            output = model.apply({"params": params}, x, deterministic=True)
            return jnp.mean(output)

        grads = jax.grad(loss_fn)(variables["params"])

        # Check that gradients are not zero
        total_grad_norm = sum(jnp.sum(jnp.abs(g)) for g in jax.tree_util.tree_leaves(grads))
        assert total_grad_norm > 0


class TestSwinClassifier:
    """Tests for SwinClassifier wrapper."""

    def test_create_classifier(self, rng):
        """Test classifier creation."""
        classifier = create_classifier(
            model_name="swin_tiny",
            num_dims=2,
            num_classes=10,
        )

        x = jnp.ones((2, 3, 64, 64))
        variables = classifier.init(rng, x, deterministic=True)
        output = classifier.apply(variables, x, deterministic=True)

        assert output.shape == (2, 10)

    def test_classifier_wrapper(self, rng, config_2d):
        """Test SwinClassifier wrapper."""
        classifier = SwinClassifier(config=config_2d)

        x = jnp.ones((2, 3, 32, 32))
        variables = classifier.init(rng, x, deterministic=True)
        output = classifier.apply(variables, x, deterministic=True)

        assert output.shape == (2, config_2d.num_classes)


class TestModelJIT:
    """Tests for JIT compilation of models."""

    def test_jit_inference(self, rng, config_2d):
        """Test JIT-compiled inference."""
        model = NDSwinTransformer(config=config_2d)

        x = jnp.ones((2, 3, 32, 32))
        variables = model.init(rng, x, deterministic=True)

        @jax.jit
        def predict(params, x):
            return model.apply({"params": params}, x, deterministic=True)

        # First call (compilation)
        output1 = predict(variables["params"], x)

        # Second call (cached)
        output2 = predict(variables["params"], x)

        assert jnp.allclose(output1, output2)

    def test_jit_training_step(self, rng, config_2d):
        """Test JIT-compiled training step."""
        model = NDSwinTransformer(config=config_2d)

        x = jnp.ones((2, 3, 32, 32))
        y = jnp.array([0, 1])

        variables = model.init(rng, x, deterministic=False)
        optimizer = optax.adam(1e-4)
        opt_state = optimizer.init(variables["params"])

        @jax.jit
        def train_step(params, opt_state, x, y):
            def loss_fn(params):
                logits = model.apply({"params": params}, x, deterministic=True)
                loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
                return loss

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)
            return params_new, opt_state_new, loss

        # Run a few steps
        params = variables["params"]
        for _ in range(3):
            params, opt_state, loss = train_step(params, opt_state, x, y)

        assert jnp.isfinite(loss)


class TestParameterCount:
    """Tests for parameter counting."""

    def test_parameter_count_2d(self, rng, config_2d):
        """Test parameter count for 2D model."""
        model = NDSwinTransformer(config=config_2d)

        x = jnp.ones((1, 3, 32, 32))
        variables = model.init(rng, x, deterministic=True)

        # Count parameters
        num_params = sum(p.size for p in jax.tree_util.tree_leaves(variables["params"]))

        # Should be reasonable for a tiny model
        assert num_params > 0
        assert num_params < 100_000_000  # Less than 100M for tiny

    def test_parameter_shapes(self, rng, config_2d):
        """Test that parameter shapes are as expected."""
        model = NDSwinTransformer(config=config_2d)

        x = jnp.ones((1, 3, 32, 32))
        variables = model.init(rng, x, deterministic=True)

        # Check patch embedding
        assert "patch_embed" in variables["params"]

        # Check stages
        for i in range(len(config_2d.depths)):
            stage_key = f"layer{i}"
            assert stage_key in variables["params"], f"Missing {stage_key}"
