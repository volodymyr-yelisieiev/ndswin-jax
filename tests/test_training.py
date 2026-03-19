"""Tests for training infrastructure."""

import jax
import jax.numpy as jnp

from ndswin.training.augmentation import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
)
from ndswin.training.losses import (
    cross_entropy_loss,
    focal_loss,
)
from ndswin.training.metrics import (
    MetricTracker,
    accuracy,
    top_k_accuracy,
)
from ndswin.training.optimizer import (
    create_learning_rate_schedule,
    create_optimizer,
)
from ndswin.training.scheduler import (
    CosineAnnealingSchedule,
    LinearSchedule,
    WarmupSchedule,
)


class TestLosses:
    """Tests for loss functions."""

    def test_cross_entropy_basic(self):
        """Test basic cross-entropy loss."""
        logits = jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        labels = jnp.array([2, 1])

        loss = cross_entropy_loss(logits, labels)

        assert loss.shape == ()
        assert loss > 0

    def test_cross_entropy_one_hot(self):
        """Test cross-entropy with one-hot labels."""
        logits = jnp.array([[1.0, 2.0, 3.0]])
        labels = jnp.array([[0.0, 0.0, 1.0]])

        loss = cross_entropy_loss(logits, labels)

        assert loss.shape == ()

    def test_label_smoothing(self):
        """Test label smoothing."""
        logits = jnp.array([[1.0, 2.0, 3.0]])
        labels = jnp.array([2])

        loss_no_smooth = cross_entropy_loss(logits, labels, label_smoothing=0.0)
        loss_smooth = cross_entropy_loss(logits, labels, label_smoothing=0.1)

        # Smoothed loss should be different
        assert not jnp.allclose(loss_no_smooth, loss_smooth)

    def test_focal_loss(self):
        """Test focal loss."""
        logits = jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        labels = jnp.array([2, 1])

        loss = focal_loss(logits, labels, gamma=2.0)

        assert loss.shape == ()
        assert loss > 0


class TestMetrics:
    """Tests for metrics."""

    def test_accuracy(self):
        """Test accuracy computation."""
        logits = jnp.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
        labels = jnp.array([1, 0, 0])

        acc = accuracy(logits, labels)

        # 2 out of 3 correct
        assert jnp.isclose(acc, 2 / 3)

    def test_accuracy_perfect(self):
        """Test perfect accuracy."""
        logits = jnp.array([[0.1, 0.9], [0.9, 0.1]])
        labels = jnp.array([1, 0])

        acc = accuracy(logits, labels)

        assert jnp.isclose(acc, 1.0)

    def test_top_k_accuracy(self):
        """Test top-k accuracy."""
        logits = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        labels = jnp.array([3])  # Second highest

        acc_1 = top_k_accuracy(logits, labels, k=1)
        acc_2 = top_k_accuracy(logits, labels, k=2)

        assert jnp.isclose(acc_1, 0.0)
        assert jnp.isclose(acc_2, 1.0)

    def test_metric_tracker(self):
        """Test MetricTracker."""
        tracker = MetricTracker()

        tracker.update({"loss": 1.0, "accuracy": 0.5}, count=2)
        tracker.update({"loss": 2.0, "accuracy": 0.7}, count=2)

        metrics = tracker.compute()

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert jnp.isclose(metrics["loss"], 1.5)  # Average of 1.0 and 2.0
        assert jnp.isclose(metrics["accuracy"], 0.6)  # Average of 0.5 and 0.7


class TestAugmentation:
    """Tests for data augmentation."""

    def test_normalize(self):
        """Test normalization."""
        normalize = Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )

        x = jnp.ones((3, 32, 32))
        normalized = normalize(x)

        # (1 - 0.5) / 0.5 = 1
        assert jnp.allclose(normalized, 1.0)

    def test_random_flip(self):
        """Test random horizontal flip."""
        flip = RandomHorizontalFlip(p=1.0)  # Always flip

        x = jnp.arange(12).reshape(3, 2, 2).astype(jnp.float32)
        flipped = flip(x, key=jax.random.PRNGKey(0))

        # Check that last dimension is reversed
        assert jnp.allclose(flipped[:, :, 0], x[:, :, 1])
        assert jnp.allclose(flipped[:, :, 1], x[:, :, 0])

    def test_compose(self):
        """Test composing transforms."""
        transform = Compose(
            [
                RandomHorizontalFlip(p=0.5),
                Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )

        x = jnp.ones((1, 32, 32))
        transformed = transform(x, key=jax.random.PRNGKey(0))

        # Should be normalized
        assert transformed.shape == x.shape


class TestSchedulers:
    """Tests for learning rate schedulers."""

    def test_cosine_schedule(self):
        """Test cosine annealing schedule."""
        schedule = CosineAnnealingSchedule(
            base_lr=0.1,
            total_steps=100,
            min_lr=0.001,
        )

        # Start
        assert jnp.isclose(schedule(0), 0.1)

        # End
        assert jnp.isclose(schedule(100), 0.001, atol=1e-3)

        # Middle should be between
        mid_lr = schedule(50)
        assert 0.001 < mid_lr < 0.1

    def test_warmup_schedule(self):
        """Test warmup schedule."""
        schedule = WarmupSchedule(
            base_lr=0.1,
            warmup_steps=10,
        )

        # Start
        assert schedule(0) == 0.0

        # End of warmup
        assert jnp.isclose(schedule(10), 0.1)

        # After warmup
        assert jnp.isclose(schedule(20), 0.1)

    def test_linear_schedule(self):
        """Test linear schedule."""
        schedule = LinearSchedule(
            base_lr=0.1,
            total_steps=100,
            min_lr=0.01,
        )

        # Start
        assert jnp.isclose(schedule(0), 0.1)

        # End
        assert jnp.isclose(schedule(100), 0.01)


class TestOptimizer:
    """Tests for optimizer creation."""

    def test_create_schedule(self):
        """Test learning rate schedule creation."""
        schedule = create_learning_rate_schedule(
            base_lr=0.1,
            num_steps=1000,
            warmup_steps=100,
            schedule_type="cosine",
        )

        # Should be callable
        lr = schedule(500)
        assert 0 < lr <= 0.1

    def test_create_optimizer(self, training_config):
        """Test optimizer creation."""
        optimizer = create_optimizer(
            training_config,
            num_steps=1000,
        )

        # Should be an Optax optimizer
        assert hasattr(optimizer, "init")
        assert hasattr(optimizer, "update")
