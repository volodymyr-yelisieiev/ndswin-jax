# Quick Start Guide

This guide will help you get started with NDSwin-JAX in just a few minutes.

## Your First Model

```python
import jax
import jax.numpy as jnp
from ndswin import NDSwinConfig, NDSwinTransformer

# Create a configuration
config = NDSwinConfig(
    num_dims=2,
    patch_size=(4, 4),
    window_size=(4, 4),
    in_channels=3,
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    num_classes=10,
)

# Create the model
model = NDSwinTransformer(config=config)

# Initialize with random weights
rng = jax.random.PRNGKey(42)
dummy_input = jnp.ones((1, 3, 32, 32))  # (batch, channels, height, width)
variables = model.init(rng, dummy_input)

# Make predictions
output = model.apply(variables, dummy_input)
print(f"Output shape: {output.shape}")  # (1, 10)
```

## Using Preset Configurations

NDSwin-JAX provides preset configurations for common use cases:

```python
from ndswin import NDSwinConfig, NDSwinTransformer

# Tiny model for 2D images
config = NDSwinConfig.swin_tiny_2d()

# Small model for 2D images  
config = NDSwinConfig.swin_small_2d()

# For 3D data (medical imaging, video)
config = NDSwinConfig.swin_tiny_3d()

# For 4D data
config = NDSwinConfig.swin_tiny_4d()
```

## Training a Model

Here's a complete training example:

```python
import jax
import jax.numpy as jnp

from ndswin import NDSwinConfig, NDSwinTransformer, TrainingConfig
from ndswin.training import Trainer, SyntheticDataLoader

# Configuration
model_config = NDSwinConfig(
    num_dims=2,
    patch_size=(4, 4),
    window_size=(4, 4),
    in_channels=3,
    embed_dim=48,
    depths=(2, 2),
    num_heads=(3, 6),
    num_classes=10,
)

train_config = TrainingConfig(
    epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    num_classes=10,
)

# Create model
model = NDSwinTransformer(config=model_config)

# Create data loaders
train_loader = SyntheticDataLoader(
    num_samples=1000,
    input_shape=(3, 32, 32),
    num_classes=10,
    batch_size=32,
)

# Create trainer
trainer = Trainer(
    model=model,
    config=train_config,
    seed=42,
)

# Train
history = trainer.fit(train_loader)

print(f"Final loss: {history['train_loss'][-1]:.4f}")
print(f"Final accuracy: {history['train_accuracy'][-1]:.4f}")
```

## Making Predictions

```python
from ndswin.inference import ClassificationPredictor

# Create predictor
predictor = ClassificationPredictor(
    model=model,
    params=trainer.state.params,
    class_names=["class_0", "class_1", "..."],
)

# Single image prediction
image = jnp.zeros((3, 32, 32))  # Your image
result = predictor.predict(image)

print(f"Class: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Working with 3D Data

```python
from ndswin import NDSwinConfig, NDSwinTransformer

# Configure for 3D data (e.g., medical volumes)
config = NDSwinConfig(
    num_dims=3,
    patch_size=(4, 8, 8),
    window_size=(4, 8, 8),
    in_channels=1,  # Single channel
    embed_dim=48,
    depths=(2, 2, 4),
    num_heads=(3, 6, 12),
    num_classes=2,
)

model = NDSwinTransformer(config=config)

# Input shape: (batch, channels, depth, height, width)
x = jnp.ones((1, 1, 32, 64, 64))
rng = jax.random.PRNGKey(0)
variables = model.init(rng, x)
output = model.apply(variables, x)
```

## Feature Extraction

```python
# Get intermediate features
output = model.apply(
    variables, x, return_features=True
)

# output is a dict with 'logits' and 'features'
logits = output['logits']
features = output['features']

# features is a list of tensors from each stage
for i, feat in enumerate(features):
    print(f"Stage {i}: {feat.shape}")
```

## Saving and Loading Models

```python
from ndswin.models.pretrained import save_checkpoint, load_checkpoint

# Save
save_checkpoint(
    checkpoint_dir="checkpoints/",
    step=1000,
    variables=variables,
    config=config,
)

# Load
loaded = load_checkpoint("checkpoints/checkpoint_1000")
```

## Next Steps

- [Configuration Guide](configuration.md) - Detailed configuration options
- [Training Guide](training.md) - Documentation on how to train your own models
- [API Reference](api.md) - Full API documentation
