"""Integration tests for NDSwin-JAX."""

import jax
import jax.numpy as jnp
import optax
import pytest

from ndswin.config import NDSwinConfig, TrainingConfig
from ndswin.models.swin import NDSwinTransformer
from ndswin.training.data import SyntheticDataLoader


class TestEndToEndTraining:
    """End-to-end training tests."""

    @pytest.fixture
    def small_config(self):
        """Small model config for testing."""
        return NDSwinConfig(
            num_dims=2,
            patch_size=(4, 4),
            window_size=(4, 4),
            in_channels=3,
            embed_dim=24,
            depths=(1, 1),
            num_heads=(2, 4),
            num_classes=5,
            drop_path_rate=0.0,
            drop_rate=0.0,
        )

    @pytest.fixture
    def train_config(self):
        """Training config for testing."""
        return TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            num_classes=5,
            warmup_epochs=0,
        )

    def test_train_step(self, rng, small_config, train_config):
        """Test a single training step."""
        model = NDSwinTransformer(config=small_config)

        # Create data
        x = jax.random.normal(rng, (2, 3, 16, 16))
        y = jnp.array([0, 1])

        # Initialize
        variables = model.init(rng, x, deterministic=False)
        optimizer = optax.adamw(1e-3)
        _ = optimizer.init(variables["params"])

        # Training step
        def loss_fn(params):
            logits = model.apply({"params": params}, x, deterministic=True)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))

        loss, grads = jax.value_and_grad(loss_fn)(variables["params"])

        assert jnp.isfinite(loss)

    def test_training_loop(self, rng, small_config, train_config):
        """Test a short training loop."""
        model = NDSwinTransformer(config=small_config)

        # Create synthetic data
        train_loader = SyntheticDataLoader(
            num_samples=8,
            input_shape=(3, 16, 16),
            num_classes=5,
            batch_size=2,
            shuffle=True,
        )

        # Initialize model
        sample_x = jnp.ones((1, 3, 16, 16))
        variables = model.init(rng, sample_x, deterministic=True)

        # Create optimizer
        optimizer = optax.adamw(1e-3)
        opt_state = optimizer.init(variables["params"])
        params = variables["params"]

        # Define loss function factory to avoid closure issues
        def make_loss_fn(batch_images, batch_labels):
            def loss_fn(params):
                logits = model.apply({"params": params}, batch_images, deterministic=True)
                return jnp.mean(
                    optax.softmax_cross_entropy_with_integer_labels(logits, batch_labels)
                )

            return loss_fn

        # Run a few batches
        losses = []
        for batch in train_loader:
            loss_fn = make_loss_fn(batch["image"], batch["label"])
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss))

        # Check that training progressed
        assert all(jnp.isfinite(loss_val) for loss_val in losses)


class TestMultiDimensional:
    """Tests for multi-dimensional support."""

    def test_2d_pipeline(self, rng):
        """Test complete 2D pipeline."""
        config = NDSwinConfig(
            num_dims=2,
            patch_size=(4, 4),
            window_size=(4, 4),
            embed_dim=48,
            depths=(2, 2),
            num_heads=(3, 6),
            num_classes=10,
        )

        model = NDSwinTransformer(config=config)
        x = jax.random.normal(rng, (2, 3, 32, 32))

        variables = model.init(rng, x, deterministic=True)
        output = model.apply(variables, x, deterministic=True)

        assert output.shape == (2, 10)

    def test_3d_pipeline(self, rng):
        """Test complete 3D pipeline."""
        config = NDSwinConfig(
            num_dims=3,
            patch_size=(2, 4, 4),
            window_size=(2, 4, 4),
            embed_dim=48,
            depths=(2, 2),
            num_heads=(3, 6),
            num_classes=10,
        )

        model = NDSwinTransformer(config=config)
        x = jax.random.normal(rng, (1, 3, 4, 16, 16))

        variables = model.init(rng, x, deterministic=True)
        output = model.apply(variables, x, deterministic=True)

        assert output.shape == (1, 10)


class TestInference:
    """Tests for inference."""

    def test_predictor(self, rng):
        """Test ClassificationPredictor."""
        from ndswin.inference.predictor import ClassificationPredictor

        config = NDSwinConfig(
            num_dims=2,
            patch_size=(4, 4),
            window_size=(4, 4),
            in_channels=3,
            embed_dim=24,
            depths=(1,),
            num_heads=(2,),
            num_classes=5,
        )

        model = NDSwinTransformer(config=config)
        x = jnp.ones((1, 3, 16, 16))
        variables = model.init(rng, x, deterministic=True)

        predictor = ClassificationPredictor(
            model=model,
            params=variables["params"],
            class_names=["a", "b", "c", "d", "e"],
        )

        # Single prediction
        result = predictor.predict(jnp.ones((3, 16, 16)))

        assert "class_id" in result
        assert "confidence" in result
        assert "class_name" in result
