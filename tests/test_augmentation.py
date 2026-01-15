"""Tests for data augmentation utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ndswin.training.augmentation import (
    ColorJitter,
    Compose,
    Cutmix,
    Cutout,
    Denormalize,
    Mixup,
    MixupOrCutmix,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
    create_augmentation_pipeline,
)


@pytest.fixture
def dummy_img_2d():
    return jnp.ones((3, 32, 32))


@pytest.fixture
def dummy_img_3d():
    return jnp.ones((1, 16, 16, 16))


@pytest.fixture
def prng_key():
    return jax.random.PRNGKey(42)


def test_random_horizontal_flip(dummy_img_2d, prng_key):
    """Test horizontal flip on 2D images."""
    transform = RandomHorizontalFlip(p=1.0)
    
    # Deterministic
    out_det = transform(dummy_img_2d)
    assert out_det.shape == dummy_img_2d.shape
    
    # Random
    out_rand = transform(dummy_img_2d, prng_key)
    assert out_rand.shape == dummy_img_2d.shape


def test_random_vertical_flip(dummy_img_2d, prng_key):
    """Test vertical flip on 2D images."""
    transform = RandomVerticalFlip(p=1.0)
    out = transform(dummy_img_2d, prng_key)
    assert out.shape == dummy_img_2d.shape


def test_random_crop(dummy_img_2d, prng_key):
    """Test random crop with padding."""
    transform = RandomCrop(size=(16, 16), padding=(4, 4))
    
    # Center crop (deterministic)
    out_center = transform(dummy_img_2d)
    assert out_center.shape == (3, 16, 16)
    
    # Random crop
    out_rand = transform(dummy_img_2d, prng_key)
    assert out_rand.shape == (3, 16, 16)


def test_normalize_denormalize(dummy_img_2d):
    """Test normalization and its inverse."""
    mean = (0.5, 0.5, 0.5)
    std = (0.2, 0.2, 0.2)
    
    norm = Normalize(mean=mean, std=std)
    denorm = Denormalize(mean=mean, std=std)
    
    x_norm = norm(dummy_img_2d)
    assert not jnp.allclose(x_norm, dummy_img_2d)
    
    x_denorm = denorm(x_norm)
    assert jnp.allclose(x_denorm, dummy_img_2d)


def test_random_rotation(dummy_img_2d, prng_key):
    """Test random rotation (identity placeholder)."""
    transform = RandomRotation(degrees=15.0)
    out = transform(dummy_img_2d, prng_key)
    assert out.shape == dummy_img_2d.shape


def test_color_jitter(dummy_img_2d, prng_key):
    """Test color jitter adjustments."""
    transform = ColorJitter(brightness=0.5, contrast=0.5)
    out = transform(dummy_img_2d, prng_key)
    assert out.shape == dummy_img_2d.shape


def test_mixup(prng_key):
    """Test mixup augmentation."""
    mixup = Mixup(alpha=1.0, num_classes=10)
    
    x = jnp.ones((4, 3, 32, 32))
    y = jnp.array([0, 1, 2, 3])
    
    x_mixed, y_mixed = mixup(x, y, prng_key)
    assert x_mixed.shape == x.shape
    assert y_mixed.shape == (4, 10)


def test_cutmix(prng_key):
    """Test cutmix augmentation."""
    cutmix = Cutmix(alpha=1.0, num_classes=10)
    
    x = jnp.ones((4, 3, 32, 32))
    y = jnp.array([0, 1, 2, 3])
    
    x_mixed, y_mixed = cutmix(x, y, prng_key)
    assert x_mixed.shape == x.shape
    assert y_mixed.shape == (4, 10)


def test_mixup_or_cutmix(prng_key):
    """Test combined mixup/cutmix wrapper."""
    transform = MixupOrCutmix(p=0.5, num_classes=10)
    
    x = jnp.ones((4, 3, 32, 32))
    y = jnp.array([0, 1, 2, 3])
    
    x_mixed, y_mixed = transform(x, y, prng_key)
    assert x_mixed.shape == x.shape
    assert y_mixed.shape == (4, 10)


def test_cutout(dummy_img_2d, prng_key):
    """Test cutout augmentation."""
    transform = Cutout(size=8, p=1.0)
    out = transform(dummy_img_2d, prng_key)
    assert out.shape == dummy_img_2d.shape


def test_compose(dummy_img_2d, prng_key):
    """Test composition of multiple transforms."""
    transform = Compose([
        RandomCrop(size=(16, 16)),
        RandomHorizontalFlip(p=1.0)
    ])
    
    out = transform(dummy_img_2d, prng_key)
    assert out.shape == (3, 16, 16)


def test_create_augmentation_pipeline():
    """Test pipeline factory."""
    class DummyConfig:
        def __init__(self):
            self.augmentation = True
            self.random_crop = True
            self.image_size = (16, 16)
            self.random_flip = True
            self.color_jitter = True
            self.cutout_size = 4
            self.normalize = True
            self.mean = (0.5,)
            self.std = (0.5,)
            
    config = DummyConfig()
    pipeline = create_augmentation_pipeline(config, is_training=True)
    
    assert isinstance(pipeline, Compose)
    assert len(pipeline.transforms) == 5
