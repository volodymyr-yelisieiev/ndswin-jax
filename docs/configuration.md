# Configuration Guide

This guide covers all configuration options in NDSwin-JAX, including JSON config files for training.

## JSON Configuration Files

NDSwin-JAX uses JSON configuration files for reproducible training. Config files are stored in `configs/` and contain all settings for an experiment.

### Config File Structure

```json
{
    "name": "experiment_name",
    "description": "Human-readable description",
    
    "dataset": { ... },
    "model": { ... },
    "training": { ... },
    "augmentation": { ... },
    "checkpointing": { ... },
    "logging": { ... },
    "seed": 42
}
```

### Dataset Configuration

```json
{
    "dataset": {
        "name": "cifar100",           // cifar10, cifar100, synthetic
        "data_dir": "data",           // Data directory path
        "download": true,             // Auto-download if missing
        "num_classes": 100,           // Number of output classes
        "input_shape": [3, 32, 32],   // [C, *spatial_dims]
        "mean": [0.5, 0.5, 0.5],      // Normalization mean
        "std": [0.25, 0.25, 0.25],    // Normalization std
        
        // For synthetic data only:
        "num_samples_train": 1000,
        "num_samples_val": 200
    }
}
```

### Model Configuration (JSON)

```json
{
    "model": {
        "num_dims": 2,                // 2, 3, 4, etc.
        "patch_size": [4, 4],         // Must match num_dims
        "window_size": [4, 4],        // Must match num_dims
        "in_channels": 3,
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "mlp_ratio": 4.0,
        "drop_path_rate": 0.1,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0
    }
}
```

### Training Configuration (JSON)

```json
{
    "training": {
        "epochs": 100,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "min_learning_rate": 1e-6,
        "weight_decay": 0.05,
        "optimizer": "adamw",         // adamw, sgd
        "lr_schedule": "cosine",      // cosine, linear, step
        "warmup_epochs": 5,
        "label_smoothing": 0.1,
        "gradient_clip_norm": 1.0
    }
}
```

### Augmentation Configuration (JSON)

```json
{
    "augmentation": {
        "random_crop": true,
        "crop_padding": 4,
        "random_flip": true,
        "color_jitter": false,
        "color_jitter_strength": 0.4,
        "mixup_alpha": 0.0,           // 0 = disabled
        "cutmix_alpha": 0.0,          // 0 = disabled
        "cutout_size": 0              // 0 = disabled
    }
}
```

See `configs/README.md` for complete documentation of all options.

## Python Configuration Classes

### Model Configuration (NDSwinConfig)

The `NDSwinConfig` dataclass defines all model hyperparameters.

### Basic Parameters

```python
from ndswin import NDSwinConfig

config = NDSwinConfig(
    # Input dimensions
    image_size=(224, 224),    # Spatial dimensions
    patch_size=(4, 4),        # Patch size for embedding
    window_size=(7, 7),       # Window size for attention
    in_channels=3,            # Number of input channels
    
    # Model dimensions
    embed_dim=96,             # Base embedding dimension
    depths=(2, 2, 6, 2),      # Number of blocks per stage
    num_heads=(3, 6, 12, 24), # Number of attention heads per stage
    num_classes=1000,         # Number of output classes
)
```

### Dimension Flexibility

NDSwin-JAX supports arbitrary dimensions:

```python
# 2D images
config_2d = NDSwinConfig(
    image_size=(224, 224),
    patch_size=(4, 4),
    window_size=(7, 7),
    ...
)

# 3D volumes
config_3d = NDSwinConfig(
    image_size=(32, 64, 64),    # Depth, Height, Width
    patch_size=(4, 8, 8),
    window_size=(4, 8, 8),
    ...
)

# 4D data
config_4d = NDSwinConfig(
    image_size=(8, 16, 32, 32),  # T, D, H, W
    patch_size=(2, 4, 4, 4),
    window_size=(2, 4, 8, 8),
    ...
)
```

### Regularization Parameters

```python
config = NDSwinConfig(
    # ... basic params ...
    
    # Regularization
    drop_path_rate=0.1,      # Stochastic depth rate
    dropout_rate=0.0,        # Dropout rate in attention/MLP
    attention_dropout=0.0,   # Dropout in attention
    
    # Architecture options
    use_abs_pos=False,       # Use absolute position embeddings
    mlp_ratio=4.0,           # MLP expansion ratio
    qkv_bias=True,           # Bias in QKV projection
    norm_eps=1e-6,           # LayerNorm epsilon
)
```

### Full Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_size` | Tuple[int, ...] | (224, 224) | Input spatial dimensions |
| `patch_size` | Tuple[int, ...] | (4, 4) | Patch size for embedding |
| `window_size` | Tuple[int, ...] | (7, 7) | Window size for local attention |
| `in_channels` | int | 3 | Number of input channels |
| `embed_dim` | int | 96 | Base embedding dimension |
| `depths` | Tuple[int, ...] | (2, 2, 6, 2) | Number of blocks per stage |
| `num_heads` | Tuple[int, ...] | (3, 6, 12, 24) | Attention heads per stage |
| `num_classes` | int | 1000 | Number of output classes |
| `drop_path_rate` | float | 0.1 | Stochastic depth rate |
| `dropout_rate` | float | 0.0 | General dropout rate |
| `attention_dropout` | float | 0.0 | Attention dropout rate |
| `mlp_ratio` | float | 4.0 | MLP hidden dim ratio |
| `qkv_bias` | bool | True | Use bias in attention |
| `use_abs_pos` | bool | False | Absolute position embeddings |
| `norm_eps` | float | 1e-6 | LayerNorm epsilon |

## Training Configuration (TrainingConfig)

```python
from ndswin import TrainingConfig

config = TrainingConfig(
    # Basic training
    num_epochs=100,
    batch_size=128,
    learning_rate=1e-3,
    
    # Optimizer
    optimizer="adamw",
    weight_decay=0.05,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    
    # Learning rate schedule
    lr_schedule="cosine",
    warmup_epochs=5,
    warmup_steps=0,
    min_learning_rate=1e-6,
    
    # Regularization
    label_smoothing=0.1,
    max_grad_norm=1.0,
    
    # Data
    num_classes=1000,
)
```

### Training Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_epochs` | int | 100 | Number of training epochs |
| `batch_size` | int | 128 | Batch size per device |
| `learning_rate` | float | 1e-3 | Base learning rate |
| `optimizer` | str | "adamw" | Optimizer type |
| `weight_decay` | float | 0.05 | Weight decay |
| `lr_schedule` | str | "cosine" | LR schedule type |
| `warmup_epochs` | int | 5 | Warmup epochs |
| `label_smoothing` | float | 0.1 | Label smoothing |
| `max_grad_norm` | float | 1.0 | Gradient clipping |

## Data Configuration (DataConfig)

```python
from ndswin import DataConfig

config = DataConfig(
    dataset="cifar10",
    data_dir="data/",
    image_size=(32, 32),
    in_channels=3,
    num_workers=4,
    download=True,
)
```

## Using Preset Configurations

```python
# Built-in presets
config = NDSwinConfig.swin_tiny_2d()
config = NDSwinConfig.swin_small_2d()
config = NDSwinConfig.swin_base_2d()
config = NDSwinConfig.swin_large_2d()
config = NDSwinConfig.swin_tiny_3d()
config = NDSwinConfig.swin_tiny_4d()

# Modify preset
config = NDSwinConfig.swin_tiny_2d()
config = config.replace(
    num_classes=10,
    image_size=(64, 64),
)
```

## Experiment Configuration

For complete experiment setup:

```python
from ndswin.config import ExperimentConfig

experiment = ExperimentConfig(
    name="cifar10_swin_tiny",
    model=NDSwinConfig.swin_tiny_2d().replace(num_classes=10),
    training=TrainingConfig(num_epochs=100),
    data=DataConfig(dataset="cifar10"),
    seed=42,
    output_dir="experiments/cifar10",
)
```

## Tips for Configuration

### Memory Optimization

```python
# Reduce memory usage
config = NDSwinConfig(
    embed_dim=48,       # Smaller embedding
    depths=(2, 2, 2),   # Fewer layers
    num_heads=(3, 6, 12),
    drop_path_rate=0.0,  # Disable for inference
)
```

### Speed Optimization

```python
# Faster training
config = NDSwinConfig(
    window_size=(4, 4),  # Smaller windows
    depths=(2, 2),       # Fewer stages
)

train_config = TrainingConfig(
    batch_size=256,      # Larger batches
)
```

### Best Practices

1. **Window size** should divide image size evenly
2. **Patch size** should divide image size evenly
3. **Depths** and **num_heads** should have same length
4. **embed_dim** should be divisible by first num_heads value
5. For 3D+ data, use smaller window sizes to manage memory
