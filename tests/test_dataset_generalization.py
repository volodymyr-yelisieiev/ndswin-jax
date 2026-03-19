"""Test generalization across different datasets and tasks (Classification vs Segmentation)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ndswin.config import NDSwinConfig, TrainingConfig
from ndswin.models import NDSwinTransformer
from ndswin.models.classifier import SwinForSegmentation
from ndswin.training.trainer import Trainer
from ndswin.training.data import SyntheticDataLoader, SyntheticSegmentationDataLoader
from typing import Iterator

@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)

def test_classification_generalization(rng):
    """Test that trainer works correctly for classification tasks (CIFAR-like)."""
    num_classes = 10
    model_config = NDSwinConfig(
        num_dims=2,
        patch_size=(4, 4),
        window_size=(4, 4),
        in_channels=3,
        embed_dim=24,
        depths=(2, 2),
        num_heads=(3, 6),
        num_classes=num_classes
    )
    
    num_devices = len(jax.devices())
    batch_size = max(1, num_devices) * 2  # Ensure divisible by num_devices

    train_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=1e-3,
        num_classes=num_classes,
        loss="cross_entropy"
    )
    
    model = NDSwinTransformer(config=model_config)
    
    train_loader = SyntheticDataLoader(
        num_samples=batch_size * 2,
        input_shape=(3, 32, 32),
        num_classes=num_classes,
        batch_size=batch_size
    )
    
    trainer = Trainer(
        model=model,
        config=train_config,
        task="classification",
        loss_name="cross_entropy"
    )
    
    # Run fit for 1 epoch
    history = trainer.fit(train_loader, num_epochs=1)
    
    # Ensure history contains expected keys for classification
    assert "train_loss" in history
    assert "train_accuracy" in history
    assert len(history["train_loss"]) == 1
    assert jnp.isfinite(history["train_loss"][0])

def test_segmentation_generalization(rng):
    """Test that trainer works correctly for segmentation tasks (Dice/BCE)."""
    num_classes = 2 # binary segmentation
    model_config = NDSwinConfig(
        num_dims=3,
        patch_size=(4, 4, 4),
        window_size=(4, 4, 4),
        in_channels=1,
        embed_dim=24,
        depths=(2, 2, 2),
        num_heads=(3, 6, 12),
        num_classes=num_classes
    )
    
    num_devices = len(jax.devices())
    batch_size = max(1, num_devices) * 2

    train_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=1e-3,
        num_classes=num_classes,
        loss="dice"
    )
    
    # Use SwinForSegmentation for segmentation tasks
    model = SwinForSegmentation(
        config=model_config,
        num_classes=num_classes
    )
    
    # Segmentation yields images (B, C, D, H, W) and labels (B, D, H, W)
    train_loader = SyntheticSegmentationDataLoader(
        num_samples=batch_size * 2,
        input_shape=(1, 16, 16, 16),
        num_classes=num_classes,
        batch_size=batch_size
    )
    
    trainer = Trainer(
        model=model,
        config=train_config,
        task="segmentation",
        loss_name="dice"
    )
    
    # Run fit for 1 epoch
    history = trainer.fit(train_loader, num_epochs=1)
    
    # Ensure history contains expected keys for segmentation
    assert "train_loss" in history
    assert "train_dice" in history
    assert "train_voxel_accuracy" in history
    assert len(history["train_loss"]) == 1
    assert jnp.isfinite(history["train_loss"][0])
    
def test_bce_segmentation_broadcasting(rng):
    """Test that BCE loss works with broadcasting for segmentation."""
    num_classes = 2
    model_config = NDSwinConfig(
        num_dims=3,
        patch_size=(4, 4, 4),
        window_size=(4, 4, 4),
        in_channels=1,
        embed_dim=24,
        depths=(2, 2, 2),
        num_heads=(3, 6, 12),
        num_classes=num_classes
    )
    
    num_devices = len(jax.devices())
    batch_size = max(1, num_devices) * 2

    train_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=1e-3,
        num_classes=num_classes,
        loss="bce"
    )
    
    # Use SwinForSegmentation for segmentation tasks
    model = SwinForSegmentation(
        config=model_config,
        num_classes=num_classes
    )
    
    train_loader = SyntheticSegmentationDataLoader(
        num_samples=batch_size * 2,
        input_shape=(1, 16, 16, 16),
        num_classes=num_classes,
        batch_size=batch_size
    )
    
    trainer = Trainer(
        model=model,
        config=train_config,
        task="segmentation",
        loss_name="bce"
    )
    
    # Run fit for 1 epoch
    history = trainer.fit(train_loader, num_epochs=1)
    
    # Ensure history contains expected keys for segmentation
    assert "train_loss" in history
    assert "train_dice" in history
    assert "train_voxel_accuracy" in history
    assert jnp.isfinite(history["train_loss"][0])
