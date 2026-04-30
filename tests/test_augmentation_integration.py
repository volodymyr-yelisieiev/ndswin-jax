"""Integration tests for augmentation pipeline wiring.

Verifies that data augmentation transforms are properly created and passed
to data loaders by create_data_loader(), and that Mixup/CutMix is wired
into the training pipeline.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ndswin.config import DataConfig
from ndswin.training.augmentation import (
    ColorJitter,
    Compose,
    Cutout,
    MixupOrCutmix,
    create_augmentation_pipeline,
)
from ndswin.training.data import CIFARDataLoader, SyntheticDataLoader, create_data_loader


class TestAugmentationPipelineWiring:
    """Tests that augmentation pipeline is correctly built from configs."""

    def test_pipeline_with_augmentation_enabled(self):
        """Pipeline with augmentation should contain transforms."""
        config = DataConfig(
            dataset="cifar10",
            image_size=(32, 32),
            in_channels=3,
            augmentation=True,
            random_crop=True,
            random_flip=True,
            normalize=True,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        pipeline = create_augmentation_pipeline(config, is_training=True)
        assert isinstance(pipeline, Compose)
        # random_crop + random_flip + normalize = 3
        assert len(pipeline.transforms) == 3

    def test_pipeline_skip_normalize(self):
        """Pipeline with skip_normalize should not include Normalize."""
        config = DataConfig(
            dataset="cifar10",
            image_size=(32, 32),
            in_channels=3,
            augmentation=True,
            random_crop=True,
            random_flip=True,
            normalize=True,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        pipeline = create_augmentation_pipeline(config, is_training=True, skip_normalize=True)
        assert isinstance(pipeline, Compose)
        # random_crop + random_flip only (no normalize)
        assert len(pipeline.transforms) == 2

    def test_pipeline_without_augmentation(self):
        """Pipeline without augmentation should only normalize."""
        config = DataConfig(
            dataset="cifar10",
            image_size=(32, 32),
            in_channels=3,
            augmentation=False,
            normalize=True,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        pipeline = create_augmentation_pipeline(config, is_training=True)
        assert isinstance(pipeline, Compose)
        # Only normalize
        assert len(pipeline.transforms) == 1

    def test_pipeline_for_eval(self):
        """Eval pipeline should not have random augmentations."""
        config = DataConfig(
            dataset="cifar10",
            image_size=(32, 32),
            in_channels=3,
            augmentation=True,
            random_crop=True,
            random_flip=True,
            normalize=True,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        pipeline = create_augmentation_pipeline(config, is_training=False)
        assert isinstance(pipeline, Compose)
        # Should only have normalize
        assert len(pipeline.transforms) == 1

    def test_pipeline_with_color_jitter_and_cutout(self):
        """Pipeline should include color jitter and cutout when configured."""
        config = DataConfig(
            dataset="cifar10",
            image_size=(32, 32),
            in_channels=3,
            augmentation=True,
            random_crop=True,
            random_flip=True,
            color_jitter=True,
            cutout_size=8,
            normalize=True,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        pipeline = create_augmentation_pipeline(config, is_training=True)
        assert isinstance(pipeline, Compose)
        assert len(pipeline.transforms) == 5
        assert any(isinstance(transform, ColorJitter) for transform in pipeline.transforms)
        assert any(isinstance(transform, Cutout) for transform in pipeline.transforms)

    def test_pipeline_skips_color_jitter_for_3d_inputs(self):
        """Color jitter should remain a 2D RGB-only augmentation."""
        config = DataConfig(
            dataset="volume_folder",
            image_size=(32, 32, 32),
            in_channels=1,
            augmentation=True,
            random_flip=True,
            color_jitter=True,
            normalize=True,
            mean=(0.5,),
            std=(0.5,),
        )
        pipeline = create_augmentation_pipeline(config, is_training=True)
        assert isinstance(pipeline, Compose)
        assert not any(isinstance(transform, ColorJitter) for transform in pipeline.transforms)


class TestMixupCutmixIntegration:
    """Tests that Mixup/CutMix is correctly created from config."""

    def test_mixup_only(self):
        """MixupOrCutmix with only mixup should work."""
        transform = MixupOrCutmix(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            p=1.0,  # Always mixup
            num_classes=100,
        )
        key = jax.random.PRNGKey(42)
        x = jnp.ones((4, 3, 32, 32))
        y = jnp.array([0, 1, 2, 3])
        mixed_x, mixed_y = transform(x, y, key)
        assert mixed_x.shape == x.shape
        assert mixed_y.shape == (4, 100)

    def test_cutmix_only(self):
        """MixupOrCutmix with only cutmix should work."""
        transform = MixupOrCutmix(
            mixup_alpha=1.0,
            cutmix_alpha=1.0,
            p=0.0,  # Always cutmix
            num_classes=100,
        )
        key = jax.random.PRNGKey(42)
        x = jnp.ones((4, 3, 32, 32))
        y = jnp.array([0, 1, 2, 3])
        mixed_x, mixed_y = transform(x, y, key)
        assert mixed_x.shape == x.shape
        assert mixed_y.shape == (4, 100)

    def test_mixup_cutmix_combined(self):
        """MixupOrCutmix with both should work."""
        transform = MixupOrCutmix(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            p=0.5,
            num_classes=100,
        )
        key = jax.random.PRNGKey(42)
        x = jnp.ones((4, 3, 32, 32))
        y = jnp.array([0, 1, 2, 3])
        mixed_x, mixed_y = transform(x, y, key)
        assert mixed_x.shape == x.shape
        assert mixed_y.shape == (4, 100)

    def test_mixup_from_data_config(self):
        """Creating MixupOrCutmix from DataConfig fields should work."""
        config = DataConfig(
            dataset="cifar10",
            image_size=(32, 32),
            in_channels=3,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
        )
        # Simulates what train.py does
        mixup_alpha = config.mixup_alpha
        cutmix_alpha = config.cutmix_alpha
        assert mixup_alpha > 0
        assert cutmix_alpha > 0
        transform = MixupOrCutmix(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            p=0.5,
            num_classes=100,
        )
        key = jax.random.PRNGKey(0)
        x = jnp.ones((2, 3, 32, 32))
        y = jnp.array([0, 1])
        mixed_x, mixed_y = transform(x, y, key)
        assert mixed_x.shape == x.shape
        assert mixed_y.shape == (2, 100)


class TestDataLoaderAugmentation:
    """Tests that create_data_loader properly wires transforms."""

    def test_synthetic_loader_no_transform(self):
        """Synthetic loader should work without transforms."""
        config = DataConfig(
            dataset="synthetic",
            image_size=(32, 32),
            in_channels=3,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        loader = create_data_loader(config, split="train", batch_size=4)
        assert isinstance(loader, SyntheticDataLoader)
        batch = next(iter(loader))
        assert batch["image"].shape[0] == 4

    def test_augmentation_pipeline_applied_to_data(self):
        """Verify augmentation pipeline produces valid output shapes."""
        config = DataConfig(
            dataset="cifar10",
            image_size=(32, 32),
            in_channels=3,
            augmentation=True,
            random_crop=True,
            random_flip=True,
            normalize=True,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        pipeline = create_augmentation_pipeline(config, is_training=True)

        # Simulate a single normalized image (C, H, W) -- transforms are per-sample
        x = np.random.randn(3, 32, 32).astype(np.float32)
        key = jax.random.PRNGKey(42)
        result = pipeline(x, key)

        # Output should maintain the same shape
        assert result.shape == (3, 32, 32)

    def test_cifar_loader_extracts_fast_path_color_jitter_and_cutout(self):
        """CIFAR loader should activate the fast augmentation path for tuned 2D recipes."""

        class MockCIFARLoader(CIFARDataLoader):
            def _load_data(self):
                return np.ones((8, 3, 32, 32), dtype=np.float32), np.zeros(8, dtype=np.int64)

            @property
            def dataset_info(self):
                from ndswin.training.data import DatasetInfo

                return DatasetInfo(
                    "mock", 10, 8, 0, 0, (3, 32, 32), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                )

        config = DataConfig(
            dataset="cifar10",
            image_size=(32, 32),
            in_channels=3,
            augmentation=True,
            random_crop=True,
            random_flip=True,
            color_jitter=True,
            cutout_size=8,
            normalize=True,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        pipeline = create_augmentation_pipeline(config, is_training=True, skip_normalize=True)
        loader = MockCIFARLoader(
            name="mock",
            data_dir="tmp",
            split="train",
            batch_size=4,
            shuffle=False,
            transform=pipeline,
        )

        assert loader._do_random_crop is True
        assert loader._random_flip_p == 0.5
        assert loader._color_jitter_brightness == 0.4
        assert loader._color_jitter_contrast == 0.4
        assert loader._cutout_size == 8
