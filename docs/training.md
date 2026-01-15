# Training Guide

This guide covers training models with NDSwin-JAX using the configuration-driven training framework.

## Quick Start

The easiest way to train is using the Makefile with a configuration file:

```bash
# List available configurations
make list-configs

# Train with a specific config
make train CONFIG=configs/cifar100.json

# Train in a detached tmux session (recommended for long runs)
make train-tmux CONFIG=configs/cifar100.json

# Train with command-line overrides
make train CONFIG=configs/cifar100.json TRAIN_ARGS='--epochs 200 --batch-size 256'
```

## Configuration-Driven Training

NDSwin-JAX uses JSON configuration files for reproducible experiments. Configs are stored in the `configs/` directory.

### Available Configurations

| Config | Dataset | Dims | Description |
|--------|---------|------|-------------|
| `cifar10_baseline.json` | CIFAR-10 | 2D | Baseline CIFAR-10 training |
| `cifar100.json` | CIFAR-100 | 2D | CIFAR-100 training with stronger augmentation and regularization |

### Using the Training Script Directly

```bash
# Basic usage
python train/train.py --config configs/cifar100.json

# With overrides
python train/train.py --config configs/cifar100.json \
    --epochs 200 \
    --batch-size 256 \
    --lr 5e-4
```

### Output Structure

Training automatically creates timestamped outputs:

```
checkpoints/
└── cifar100_20260115_143022/
    ├── config.json          # Copy of config used
    ├── checkpoint_*.npz     # Model checkpoints
    └── metrics.json         # Final metrics and history

logs/
└── cifar100_20260115_143022.log
```

The folder name includes the config name and timestamp for easy identification of hyperparameters.

## Creating Custom Configurations

1. Copy an existing config as a template:
   ```bash
   cp configs/cifar100.json configs/my_experiment.json
   ```

2. Edit the configuration (see `configs/README.md` for full reference)

3. Run training:
   ```bash
   make train-tmux CONFIG=configs/my_experiment.json
   ```

## Multi-Dimensional Training

NDSwin-JAX supports 2D, 3D, 4D, and higher-dimensional data. Set `num_dims` and matching `patch_size`/`window_size` in your config:

### 3D Volumetric Data

```json
{
    "dataset": {
        "name": "synthetic",
        "input_shape": [1, 32, 32, 32]
    },
    "model": {
        "num_dims": 3,
        "patch_size": [4, 4, 4],
        "window_size": [4, 4, 4]
    }
}
```

### 4D Spatiotemporal Data

```json
{
    "dataset": {
        "name": "synthetic",
        "input_shape": [1, 8, 16, 16, 16]
    },
    "model": {
        "num_dims": 4,
        "patch_size": [2, 4, 4, 4],
        "window_size": [2, 4, 4, 4]
    }
}
```

## Programmatic Training

You can also train programmatically in Python:

```python
import jax
from ndswin import NDSwinConfig, NDSwinTransformer, TrainingConfig
from ndswin.training import Trainer, SyntheticDataLoader

# Create model
config = NDSwinConfig.swin_tiny_2d(num_classes=10)
model = NDSwinTransformer(config=config)

# Training config
train_config = TrainingConfig(
    epochs=100,
    batch_size=128,
    learning_rate=1e-3,
    weight_decay=0.05,
)

# Create trainer
trainer = Trainer(model=model, config=train_config, seed=42)

# Train
history = trainer.fit(train_loader, val_loader)
```

## Data Loading

### CIFAR-100

```python
from ndswin.training import CIFAR100DataLoader

train_loader = CIFAR100DataLoader(
    data_dir="data/",
    split="train",
    batch_size=128,
    shuffle=True,
    download=True,
)

val_loader = CIFAR100DataLoader(
    data_dir="data/",
    split="test",
    batch_size=128,
    shuffle=False,
)
```

### CIFAR-10

```python
from ndswin.training import CIFAR10DataLoader

train_loader = CIFAR10DataLoader(
    data_dir="data/",
    split="train",
    batch_size=128,
    shuffle=True,
    download=True,
)

val_loader = CIFAR10DataLoader(
    data_dir="data/",
    split="test",
    batch_size=128,
    shuffle=False,
)
```

### Custom Datasets

```python
from ndswin.training import DataLoader
import numpy as np

class CustomDataLoader(DataLoader):
    def __init__(self, data_path, batch_size, **kwargs):
        self.data = np.load(data_path)
        self.images = self.data['images']
        self.labels = self.data['labels']
        super().__init__(
            batch_size=batch_size,
            **kwargs
        )
    
    def __iter__(self):
        indices = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, len(self.images), self.batch_size):
            idx = indices[start:start + self.batch_size]
            yield {
                'image': self.images[idx],
                'label': self.labels[idx],
            }
```

## Data Augmentation

```python
from ndswin.training import (
    Compose,
    RandomHorizontalFlip,
    RandomCrop,
    Normalize,
)

# Create augmentation pipeline
transform = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomCrop(size=(32, 32), padding=4),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Apply to batch
augmented = transform(batch)
```

## Optimizers

```python
from ndswin.training import create_optimizer

# AdamW
optimizer = create_optimizer(
    config=train_config,
    num_steps=10000,
)
```

## Loss Functions

```python
from ndswin.training import (
    cross_entropy_loss,
    label_smoothing_cross_entropy,
    focal_loss,
)

# Standard cross-entropy
loss = cross_entropy_loss(logits, labels)

# With label smoothing
loss = label_smoothing_cross_entropy(logits, labels, smoothing=0.1)

# Focal loss for imbalanced data
loss = focal_loss(logits, labels, gamma=2.0)
```

## Training Metrics

```python
from ndswin.training import MetricTracker, accuracy, top_k_accuracy

tracker = MetricTracker()

for batch in train_loader:
    logits = model.apply(params, batch['image'])
    
    tracker.update({
        'loss': loss_val,
        'accuracy': accuracy(logits, batch['label']),
        'top5_accuracy': top_k_accuracy(logits, batch['label'], k=5),
    })

epoch_metrics = tracker.compute()
print(f"Train Loss: {epoch_metrics['loss']:.4f}")
print(f"Train Accuracy: {epoch_metrics['accuracy']:.4f}")
```

## Gradient Accumulation

For training with limited memory:

```python
# Accumulate gradients over multiple mini-batches
accumulation_steps = 4
accumulated_grads = None

for i, batch in enumerate(train_loader):
    grads = compute_grads(state, batch)
    
    if accumulated_grads is None:
        accumulated_grads = grads
    else:
        accumulated_grads = jax.tree_util.tree_map(
            lambda a, g: a + g, accumulated_grads, grads
        )
    
    if (i + 1) % accumulation_steps == 0:
        # Average and apply
        grads = jax.tree_util.tree_map(
            lambda g: g / accumulation_steps, accumulated_grads
        )
        state = state.apply_gradients(grads=grads)
        accumulated_grads = None
```

## Multi-GPU Training

```python
import jax
from jax import pmap

# Replicate model across devices
devices = jax.devices()
state = jax.device_put_replicated(state, devices)

# Define parallel training step
@pmap
def parallel_train_step(state, batch):
    return train_step(state, batch)

# Split batch across devices
batch = jax.tree_util.tree_map(
    lambda x: x.reshape((len(devices), -1) + x.shape[1:]), 
    batch
)

# Train
state, metrics = parallel_train_step(state, batch)
```

## Checkpointing

```python
from ndswin.models.pretrained import save_checkpoint, load_checkpoint

# Save during training
save_checkpoint(
    checkpoint_dir="checkpoints/",
    step=epoch,
    variables={"params": state.params},
    config=config,
)

# Load checkpoint
restored = load_checkpoint("checkpoints/checkpoint_50")
```

## Complete Training Script

To train on CIFAR-100:

```bash
python train/train_cifar100.py --epochs 100 --batch-size 128
```

See `train/train_cifar100.py` for a complete training example.

## Tips and Best Practices

1. **Start small**: Begin with a tiny model to verify the pipeline
2. **Monitor metrics**: Use logging to track loss and accuracy
3. **Validate regularly**: Check validation metrics every epoch
4. **Use warmup**: Learning rate warmup helps training stability
5. **Augmentation**: Strong augmentation prevents overfitting
6. **Label smoothing**: Improves generalization
7. **Weight decay**: Essential for transformer regularization
8. **Gradient clipping**: Prevents unstable training
9. **Save checkpoints**: Allow resuming and model selection
10. **Early stopping**: Prevent overfitting and save compute
