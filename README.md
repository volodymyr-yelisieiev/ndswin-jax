# NDSwin-JAX: N-Dimensional Swin Transformer

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.7.1-green.svg)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-0.10+-orange.svg)](https://github.com/google/flax)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A **robust n-dimensional Swin Transformer** implementation in JAX/Flax, supporting arbitrary dimensions (2D, 3D, 4D, and beyond).

## Features

- 🔢 **N-Dimensional Support**: Works with 2D images, 3D volumes, 4D spatiotemporal data, and higher
- ⚡ **JAX-Powered**: Leverages JIT compilation, automatic differentiation, and XLA optimization
- 🎯 **Reliable Architecture**: Comprehensive testing, type hints, and documentation
- 🔧 **Configurable**: Flexible architecture with customizable depths, heads, and dimensions
- 📊 **Training Tools**: Complete training pipeline with mixed precision, checkpointing, and logging
- 🚀 **Multi-Device**: Support for CPU, GPU (CUDA), and TPU

## Installation

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/ndswin-jax.git
cd ndswin-jax

# Create conda environment
conda env create -f environment.yml
conda activate ndswin-jax

# Install the package
pip install -e .
```

### Using pip

```bash
pip install ndswin-jax
```

## Quick Start

### 2D Image Classification

```python
import jax
import jax.numpy as jnp
from ndswin import NDSwinTransformer, NDSwinConfig

# Configure for 2D images
config = NDSwinConfig(
    num_dims=2,
    patch_size=(4, 4),
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    window_size=(7, 7),
    num_classes=1000,
)

# Create model
model = NDSwinTransformer(config)

# Initialize with random key
key = jax.random.PRNGKey(0)
x = jnp.ones((1, 3, 224, 224))  # (B, C, H, W)
variables = model.init(key, x)

# Forward pass
output = model.apply(variables, x)
print(f"Output shape: {output.shape}")  # (1, 1000)
```

### 3D Medical Volume Classification

```python
from ndswin import NDSwinTransformer, NDSwinConfig

# Configure for 3D volumes
config = NDSwinConfig(
    num_dims=3,
    patch_size=(4, 4, 4),
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    window_size=(7, 7, 7),
    num_classes=10,
)

model = NDSwinTransformer(config)
x = jnp.ones((1, 1, 64, 64, 64))  # (B, C, D, H, W)
```

### 4D Video Classification

```python
from ndswin import NDSwinTransformer, NDSwinConfig

# Configure for 4D spatiotemporal data
config = NDSwinConfig(
    num_dims=4,
    patch_size=(2, 4, 4, 4),
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    window_size=(4, 7, 7, 7),
    num_classes=400,
)

model = NDSwinTransformer(config)
x = jnp.ones((1, 3, 8, 64, 64, 64))  # (B, C, T, D, H, W)
```

## Training

### CIFAR-100 Training

```bash
python train/train_cifar100.py --epochs 100 --batch-size 128
```

### Custom Training

```python
from ndswin.training import Trainer, TrainingConfig

config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=32,
    epochs=100,
    warmup_epochs=5,
)

trainer = Trainer(model, config)
trainer.fit(train_loader, val_loader)
```

## Architecture

```
NDSwin (n-dimensional Swin Transformer)
├── Patch Embedding (n-dimensional)
├── Stage 1-4 (Hierarchical levels)
│   ├── Swin Blocks (W-MSA + SW-MSA alternating)
│   │   ├── Window Multi-Head Self-Attention
│   │   ├── MLP (Feed-forward)
│   │   └── LayerNorm + Residual
│   └── Patch Merging (downsampling)
└── Classification Head
    ├── Global Average Pooling
    ├── LayerNorm
    └── Linear Classifier
```

## Supported Dimensions

| Dimension | Input Shape | Use Case |
|-----------|-------------|----------|
| 2D | `(B, C, H, W)` | Image classification, segmentation |
| 3D | `(B, C, D, H, W)` | Medical volumes, video frames |
| 4D | `(B, C, T, D, H, W)` | Spatiotemporal data |
| 5D+ | Arbitrary | Custom scientific applications |

## Documentation

- [Installation](docs/installation.md)
- [Quick Start](docs/quickstart.md)
- [Training Guide](docs/training.md)
- [Inference Guide](docs/inference.md)
- [Configuration Guide](docs/configuration.md)
- [API Reference](docs/api.md)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ndswin_jax,
  title={NDSwin-JAX: N-Dimensional Swin Transformer in JAX},
  author={NDSwin-JAX Authors},
  year={2026},
  url={https://github.com/your-org/ndswin-jax}
}
```

## Acknowledgments

- [gerkone/ndswin](https://github.com/gerkone/ndswin): Reference PyTorch implementation for N-Dimensional Swin Transformers.
- Original Swin Transformer: [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- JAX team at Google
- Flax team at Google
