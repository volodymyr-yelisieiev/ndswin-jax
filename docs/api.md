# API Reference

## ndswin

Main package containing the N-Dimensional Swin Transformer implementation.

### Classes

#### `NDSwinConfig`

Configuration dataclass for the N-Dimensional Swin Transformer.

```python
from ndswin import NDSwinConfig

config = NDSwinConfig(
    num_dims=2,
    patch_size=(4, 4),
    window_size=(7, 7),
    in_channels=3,
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    num_classes=1000,
)
```

**Class Methods:**
- `swin_tiny_2d()` - Returns Swin-T configuration for 2D
- `swin_small_2d()` - Returns Swin-S configuration for 2D
- `swin_base_2d()` - Returns Swin-B configuration for 2D
- `swin_large_2d()` - Returns Swin-L configuration for 2D
- `swin_tiny_3d()` - Returns Swin-T configuration for 3D
- `swin_tiny_4d()` - Returns Swin-T configuration for 4D

---

#### `NDSwinTransformer`

Main N-Dimensional Swin Transformer model.

```python
from ndswin import NDSwinTransformer

model = NDSwinTransformer(config=config)
```

**Methods:**
- `__call__(x, deterministic=True, return_features=False)` - Forward pass

**Arguments:**
- `x`: Input tensor of shape (B, C, *spatial)
- `deterministic`: Whether to use deterministic operations (no dropout)
- `return_features`: If True, returns (logits, features) tuple

---

#### `SwinClassifier`

High-level classifier wrapper with additional options.

```python
from ndswin import SwinClassifier

classifier = SwinClassifier(config=config)
```

---

#### `TrainingConfig`

Configuration for training.

```python
from ndswin import TrainingConfig

config = TrainingConfig(
    num_epochs=100,
    batch_size=128,
    learning_rate=1e-3,
)
```

---

## ndswin.core

Core components for the Swin Transformer.

### Window Operations

```python
from ndswin.core import (
    partition_windows,
    reverse_partition_windows,
    cyclic_shift,
    create_attention_mask,
)
```

#### `partition_windows(x, window_size)`

Partition input tensor into windows.

**Arguments:**
- `x`: Input tensor (B, *spatial, C)
- `window_size`: Tuple of window sizes

**Returns:** Windows tensor (num_windows * B, window_size_prod, C)

---

#### `reverse_partition_windows(windows, window_size, spatial_shape)`

Reverse window partitioning.

---

#### `cyclic_shift(x, shift_size)`

Apply cyclic shift for shifted window attention.

---

### Attention

```python
from ndswin.core import WindowAttention, ShiftedWindowAttention
```

#### `WindowAttention`

Window-based multi-head self-attention with relative position bias.

```python
attention = WindowAttention(
    dim=96,
    num_heads=3,
    window_size=(7, 7),
    qkv_bias=True,
)
```

---

### Blocks

```python
from ndswin.core import SwinTransformerBlock, BasicLayer
```

#### `SwinTransformerBlock`

Single Swin Transformer block with W-MSA or SW-MSA.

---

#### `BasicLayer`

A stage containing multiple Swin blocks with optional downsampling.

---

## ndswin.training

Training infrastructure.

### Data Loading

```python
from ndswin.training import DataLoader, CIFAR10DataLoader, SyntheticDataLoader
```

#### `SyntheticDataLoader`

Generate synthetic data for testing.

```python
loader = SyntheticDataLoader(
    num_samples=1000,
    input_shape=(3, 32, 32),
    num_classes=10,
    batch_size=32,
)
```

---

### Losses

```python
from ndswin.training import cross_entropy_loss, focal_loss
```

#### `cross_entropy_loss(logits, labels, label_smoothing=0.0)`

Cross-entropy loss with optional label smoothing.

---

### Metrics

```python
from ndswin.training import accuracy, top_k_accuracy, MetricTracker
```

#### `MetricTracker`

Track and average metrics over batches.

```python
tracker = MetricTracker()
tracker.update({"loss": 0.5, "accuracy": 0.9})
metrics = tracker.compute()
```

---

### Trainer

```python
from ndswin.training import Trainer, TrainState, create_train_state
```

#### `Trainer`

Main trainer class.

```python
trainer = Trainer(
    model=model,
    config=train_config,
    seed=42,
)
history = trainer.fit(train_loader, val_loader)
```

---

## ndswin.inference

Inference utilities.

### Predictor

```python
from ndswin.inference import ClassificationPredictor, BatchProcessor
```

#### `ClassificationPredictor`

High-level predictor for classification.

```python
predictor = ClassificationPredictor(
    model=model,
    params=params,
    class_names=["cat", "dog", ...],
)
result = predictor.predict(image)
```

---

#### `BatchProcessor`

Efficient batch processing for inference.

```python
processor = BatchProcessor(predictor, batch_size=32)
results = processor.process(images)
```

---

### Export

```python
from ndswin.inference import export_to_onnx, export_to_saved_model
```

#### `export_to_onnx(model, params, path, input_shape)`

Export model to ONNX format.

---

## ndswin.utils

Utility functions.

### Device Detection

```python
from ndswin.utils import get_device_info, get_device_count
```

### Reproducibility

```python
from ndswin.utils import set_global_seed, PRNGSequence
```

### Visualization

```python
from ndswin.utils import visualize_attention, create_attention_video
```

---

## Type Definitions

```python
from ndswin import Array, PRNGKey, Shape
```

- `Array`: JAX array type
- `PRNGKey`: JAX random key
- `Shape`: Tuple of integers
