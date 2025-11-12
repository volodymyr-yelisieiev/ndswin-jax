# ndswin-jax: N-Dimensional Swin Transformer in JAX/Flax

A pure JAX/Flax implementation of the N-dimensional Swin Transformer, supporting arbitrary spatial dimensions (2D, 3D, and beyond).

## Features

- 🌟 **Fully generalized N-dimensional Swin Transformer** supporting 2D images, 3D voxels, and higher dimensions
- ⚡ **Pure JAX/Flax implementation** with no PyTorch or TensorFlow dependencies
- 🎯 **Swin Transformer V2** with continuous relative position bias for better transferability
- 🔧 **Modular design** with clean separation of concerns
- 📊 **Complete training pipeline** with checkpointing, logging, and evaluation

## Architecture

This implementation follows the Swin Transformer architecture from [Liu et al., 2021](https://arxiv.org/abs/2103.14030), generalized to arbitrary spatial dimensions as described in the [ndswin](https://github.com/gerkone/ndswin) PyTorch implementation.

Key components:
- **Window-based self-attention** with shifted windows for efficient computation
- **Hierarchical architecture** with patch merging for multi-scale representations
- **Continuous relative position bias** (Swin V2 style) for better extrapolation
- **N-dimensional operations** that work seamlessly across 2D, 3D, and higher dimensions

## Installation

### Requirements

- Python >= 3.8
- JAX >= 0.4.20 (with CUDA support for GPU training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/volodymyr-yelisieiev/ndswin-jax.git
cd ndswin-jax
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
# For CPU (testing/development)
pip install -r requirements.txt

# For GPU (production with CUDA 12)
pip install jax[cuda12] flax optax ml-collections datasets Pillow tqdm

# For GPU (production with CUDA 11)
pip install jax[cuda11_pip] flax optax ml-collections datasets Pillow tqdm
```

## Usage

### Training on CIFAR-100 (2D)

```bash
python train.py \
  --config configs.cifar100 \
  --workdir ./checkpoints/cifar100
```

### Training on ModelNet40 (3D)

```bash
python train.py \
  --config configs.modelnet40 \
  --workdir ./checkpoints/modelnet40
```

### Evaluation

```bash
python evaluate.py \
  --config configs.cifar100 \
  --workdir ./checkpoints/cifar100 \
  --split test
```

### Custom Configuration

You can override configuration values from the command line:

```bash
python train.py \
  --config configs.cifar100 \
  --workdir ./checkpoints/cifar100 \
  --config_override model.dim=128 \
  --config_override num_epochs=100
```

## Configuration

Training configurations are defined in Python files under `configs/`. Each config file exports a `get_config()` function that returns a configuration dictionary.

Example configuration structure:
```python
config = ConfigDict()

# Model architecture
config.model = ConfigDict()
config.model.space = 2              # Spatial dimensions (2 for images, 3 for voxels)
config.model.dim = 96               # Base dimension
config.model.depths = (2, 2, 6, 2)  # Depth of each stage
config.model.num_heads = (3, 6, 12, 24)  # Attention heads per stage
config.model.patch_size = (4, 4)    # Patch size
config.model.window_size = (7, 7)   # Window size

# Training settings
config.batch_size = 128
config.num_epochs = 300
config.learning_rate = 1e-3
# ... and more
```

## Testing

For quick testing without downloading datasets, use the synthetic data option:

```bash
# Create synthetic datasets
python create_synthetic_data.py

# Run 2-epoch tests
python train_synthetic.py --config configs.cifar100_test --workdir /tmp/test_cifar100 --dataset cifar100
python train_synthetic.py --config configs.modelnet40_test --workdir /tmp/test_modelnet40 --dataset modelnet40
```

## Model Architecture

The model follows a hierarchical architecture with multiple stages:

1. **Patch Embedding**: Splits input into non-overlapping patches
2. **Swin Transformer Stages**: Multiple stages with Swin Transformer blocks
   - Window-based self-attention with shifted windows
   - Feed-forward MLP with GELU activation
   - Stochastic depth for regularization
3. **Patch Merging**: Downsamples spatial dimensions between stages
4. **Classification Head**: Global average pooling + linear classifier

## Project Structure

```
ndswin-jax/
├── configs/                 # Training configurations
│   ├── cifar100.py
│   ├── modelnet40.py
│   └── ...
├── src/
│   ├── layers/             # Core model layers
│   │   ├── attention/      # Attention mechanisms
│   │   ├── patching.py     # Patch operations
│   │   ├── positional.py   # Positional embeddings
│   │   └── swin.py         # Swin blocks and layers
│   ├── models/             # High-level models
│   │   └── classifier.py   # Classifier model
│   └── utils/              # Utilities
├── training/               # Training utilities
│   ├── trainer.py          # Training loop
│   ├── state.py            # Model state management
│   └── ...
├── loader.py               # Data loading
├── train.py                # Training script
├── evaluate.py             # Evaluation script
└── requirements.txt        # Dependencies
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{liu2021swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

And the reference N-dimensional PyTorch implementation:
```bibtex
@software{ndswin,
  author = {Gerkone},
  title = {ndswin: N-Dimensional Swin Transformer},
  url = {https://github.com/gerkone/ndswin},
  year = {2023}
}
```

## License

This project follows the same open-source spirit as the reference implementation. Please refer to the LICENSE file for details.

## Acknowledgments

- Original Swin Transformer paper by Liu et al.
- [ndswin](https://github.com/gerkone/ndswin) PyTorch implementation by Gerkone
- JAX and Flax teams at Google for excellent frameworks
