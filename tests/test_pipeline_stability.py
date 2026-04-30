"""Tests for data pipeline stability, GPU safety, and naming consistency."""

import jax.numpy as jnp
import numpy as np
import pytest


class TestProcessImageScaling:
    """Test that _process_image correctly scales integer images to [0, 1]."""

    def _make_loader(self):
        """Create a minimal HuggingFaceDataLoader-like object for testing."""
        from ndswin.training.data import HuggingFaceDataLoader

        # We can't easily instantiate the full loader without a dataset,
        # so we test the method via a mock-like approach.
        class FakeLoader:
            image_size = None

            def _process_image(self, img):
                return HuggingFaceDataLoader._process_image(self, img)

        return FakeLoader()

    def test_uint8_scaled_to_01(self):
        loader = self._make_loader()
        img = np.array([[[255, 0, 128]]], dtype=np.uint8)  # (1, 1, 3)
        result = loader._process_image(img)
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        assert result.min() >= 0.0
        np.testing.assert_allclose(result.max(), 1.0, atol=1e-6)

    def test_uint16_scaled_to_01(self):
        loader = self._make_loader()
        img = np.array([[[65535, 0, 32768]]], dtype=np.uint16)
        result = loader._process_image(img)
        assert result.dtype == np.float32
        assert result.max() <= 1.0

    def test_float32_passthrough(self):
        loader = self._make_loader()
        img = np.array([[[0.5, 0.1, 0.9]]], dtype=np.float32)
        result = loader._process_image(img)
        np.testing.assert_allclose(result.flatten()[:3], [0.5, 0.1, 0.9], atol=1e-6)

    def test_2d_gets_channel_dim(self):
        loader = self._make_loader()
        img = np.zeros((32, 32), dtype=np.uint8)
        result = loader._process_image(img)
        assert result.ndim == 3
        assert result.shape[0] == 1  # Channel dim added

    def test_channels_moved_to_first(self):
        loader = self._make_loader()
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        result = loader._process_image(img)
        assert result.shape == (3, 32, 32)


class TestBatchPadding:
    """Test that trainer pads batches for multi-GPU sharding."""

    def test_pad_batch_divisible(self):
        from ndswin.training.trainer import Trainer

        # Minimal model mock — we only need _pad_batch which doesn't use model
        trainer = Trainer.__new__(Trainer)
        trainer.num_devices = 4

        batch = {
            "image": jnp.ones((8, 3, 32, 32)),
            "label": jnp.zeros((8,), dtype=jnp.int32),
        }
        padded, real_bs = trainer._pad_batch(batch)
        assert padded["image"].shape[0] == 8  # Already divisible, no padding
        assert real_bs == 8

    def test_pad_batch_not_divisible(self):
        from ndswin.training.trainer import Trainer

        trainer = Trainer.__new__(Trainer)
        trainer.num_devices = 4

        batch = {
            "image": jnp.ones((5, 3, 32, 32)),
            "label": jnp.zeros((5,), dtype=jnp.int32),
        }
        padded, real_bs = trainer._pad_batch(batch)
        assert padded["image"].shape[0] == 8  # Padded to next multiple of 4
        assert real_bs == 5

    def test_pad_batch_single_sample(self):
        from ndswin.training.trainer import Trainer

        trainer = Trainer.__new__(Trainer)
        trainer.num_devices = 4

        batch = {
            "image": jnp.ones((1, 3, 32, 32)),
            "label": jnp.zeros((1,), dtype=jnp.int32),
        }
        padded, real_bs = trainer._pad_batch(batch)
        assert padded["image"].shape[0] == 4
        assert real_bs == 1


class TestStampNaming:
    """Test that get_stamp produces clean, non-redundant names."""

    def test_no_double_hf_prefix(self):
        from ndswin.config import ExperimentConfig

        exp = ExperimentConfig(name="cifar10")
        stamp = exp.get_stamp()
        # Should NOT contain "hf" at all since name is "cifar10"
        assert "hf" not in stamp
        # Should start with cifar10
        assert stamp.startswith("cifar10_")

    def test_hf_prefix_stripped(self):
        from ndswin.config import ExperimentConfig

        exp = ExperimentConfig(name="hf_cifar10")
        stamp = exp.get_stamp()
        # Should still be clean and removed prefix
        assert stamp.startswith("cifar10_")
        assert not stamp.startswith("hf_")

    def test_stamp_has_timestamp(self):
        from ndswin.config import ExperimentConfig

        exp = ExperimentConfig(name="test")
        stamp = exp.get_stamp()
        # Should contain a timestamp-like pattern
        parts = stamp.split("_")
        assert len(parts) >= 3  # name_hash_YYYYMMDD_HHMMSS


class TestCifar100DataRange:
    """Test that CIFAR-100 data is in expected normalized range."""

    def test_cifar10_loader_normalized_range(self):
        """Test that native CIFAR loader outputs data roughly in [-2.5, 2.5] (ImageNet norm)."""
        from ndswin.training.data import HuggingFaceDataLoader

        try:
            loader = HuggingFaceDataLoader(
                hf_id="cifar10", split="train", batch_size=4, shuffle=False
            )
            batch = next(iter(loader))
            img = np.array(batch["image"])
            # After normalization with CIFAR mean/std, values should be roughly in [-3, 3]
            assert img.max() < 10.0, f"Image max {img.max()} suggests raw uint8 (not normalized)"
            assert img.min() > -10.0, f"Image min {img.min()} suggests raw uint8 (not normalized)"
            # Should NOT be in [0, 255] range
            assert img.max() < 5.0, f"Image max {img.max()} too large — normalization broken"
        except Exception as e:
            if "CIFAR" in str(e) or "download" in str(e).lower():
                pytest.skip(f"CIFAR-100 not available: {e}")
            raise
