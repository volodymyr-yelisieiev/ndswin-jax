# NDSwin-JAX

**N-Dimensional Swin Transformer implementation in JAX/Flax**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![JAX](https://img.shields.io/badge/JAX-0.4.7+-purple.svg)](https://github.com/google/jax)

A pure JAX/Flax implementation of the Swin Transformer, generalized to support **N-dimensional data** (2D images, 3D volumes, 4D spatio-temporal) within a single, unified codebase.

## Features

- **N-Dimensional**: Train on 2D images, 3D voxels, or higher-dimensional data with no code changes — just configure `num_dims`.
- **Bring Your Own Data**: Easily plug in ANY custom dataset of any dimension, either from the **Hugging Face Hub** or a **local directory**.
- **Autonomous Optimization Pipeline**: Run `make optimize` to seamlessly fetch data, run hyperparameter sweeps, and train the best model hands-free.
- **Multi-GPU Data Parallelism**: Automatically uses all available GPUs with JAX mesh sharding.
- **Hyperparameter Sweeps**: Built-in random search with early stopping, auto-training of best config.
- **Job Queue**: Run sequential sweep→train pipelines for multiple datasets automatically.
- **Point Cloud → Voxel**: Automatic voxelization of point cloud datasets.

## Quick Start

### 1. Setup

```bash
git clone https://github.com/your-org/ndswin-jax.git
cd ndswin-jax

# Option A: Conda (recommended for GPU)
conda env create -f environment.yml
conda activate ndswin-jax

# Option B: pip
pip install -e ".[dev,gpu]"
```

### 2. Autonomous Optimization Pipeline (The Holy Grail)

Find the best model for any dataset using a single command:

```bash
# Will fetch data, run a hyperparameter sweep, find the best config, and train it fully.
# The custom CONFIG, SWEEP, and DATA variables let you apply this to ANY problem.
make optimize CONFIG=configs/your_config.json \
              SWEEP=configs/sweeps/your_sweep.yaml \
              HF_DATASET=your_hf_dataset
```

### 3. Manual Training and Sweeping

Train a model on a predefined configuration:

```bash
make train CONFIG=configs/cifar10.json
```

Or run a manual hyperparameter sweep:

```bash
# Run sweep (20 trials) for your custom config
make sweep SWEEP=configs/sweeps/my_custom_sweep.yaml SWEEP_TRIALS=20
```

### 4. Run the Full 2D→3D Pipeline

```bash
# Fetch 3D dataset
make fetch-data HF_DATASET=jxie/modelnet40 DATASET_DIR=data/modelnet10

# Run sequential queue: CIFAR-100 sweep+train → ModelNet40 sweep+train
make queue-tmux
```

## Configuration

All experiments are driven by JSON config files. Key fields:

```json
{
  "name": "my_experiment",
  "model": {
    "num_dims": 2,
    "patch_size": [4, 4],
    "embed_dim": 48,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": [4, 4],
    "in_channels": 3,
    "num_classes": 100
  },
  "training": {
    "epochs": 100,
    "batch_size": 128,
    "learning_rate": 5e-4,
    "early_stopping": true
  },
  "data": {
    "dataset": "cifar10",
    "data_dir": "data/cifar10",
    "image_size": [32, 32],
    "in_channels": 3,
    "task": "classification"
  }
}
```

> **Note**: `image_size` should contain **spatial dimensions only** (not channels). Channels go in `in_channels`.

### Using Hugging Face Datasets

Set `dataset` to `"hf:<hf_id>"` or add `"hf_id"`:

```json
{
  "data": {
    "dataset": "hf:cifar10",
    "image_size": [32, 32],
    "in_channels": 3,
    "task": "classification"
  }
}
```

### Using Local Folder Datasets

For 3D voxel data organized as `data_dir/{split}/class_NNN/*.npz`:

```json
{
  "model": { "num_dims": 3 },
  "data": {
    "dataset": "volume_folder",
    "data_dir": "data/modelnet10",
    "image_size": [32, 32, 32],
    "in_channels": 1,
    "task": "classification"
  }
}
```

## Makefile Reference

All targets accept configurable overrides:

| Target | Description | Key Variables |
|--------|-------------|---------------|
| `train` | Train a model | `CONFIG` |
| `train-tmux` | Train in tmux | `CONFIG` |
| `sweep` | Run hyperparameter sweep | `SWEEP`, `SWEEP_TRIALS` |
| `auto-sweep` | Sweep + train best | `SWEEP`, `TRAIN_EPOCHS` |
| `auto-sweep-tmux` | Auto-sweep in tmux | `SWEEP`, `TRAIN_EPOCHS` |
| `queue` | Run job queue | `QUEUE_FILE` |
| `queue-tmux` | Queue in tmux | `QUEUE_FILE` |
| `fetch-data` | Fetch HF dataset | `HF_DATASET`, `DATASET_DIR` |
| `test` | Run tests | — |
| `test-fast` | Tests (stop on first failure) | — |
| `clean` | Clean build caches | — |
| `clean-runs` | Clean logs + outputs | — |
| `clean-all` | Clean everything | — |
| `status` | Show running tmux sessions | — |
| `list-configs` | List all configs | — |

### Examples

The framework provides configurations like `cifar10.json` or `modelnet10.json` strictly as demonstrative examples. You should use them as templates to create configs for your own tasks.

```bash
# Full autonomous optimization for a custom 3D dataset
make optimize CONFIG=configs/my_3d_task.json SWEEP=configs/sweeps/my_sweep.yaml

# Train with your custom configuration
make train CONFIG=configs/my_custom_task.json

# Fetch any HF dataset to local storage
make fetch-data HF_DATASET=username/custom_dataset DATASET_DIR=data/custom_dataset

# Clean up outputs and logs
make clean-all
```

## Project Structure

```
ndswin-jax/
├── configs/                    # Experiment configs
│   ├── cifar10.json          # 2D classification
│   ├── modelnet10.json        # 3D classification
│   ├── sweeps/                # Hyperparameter sweep configs
│   └── queues/                # Job queue configs
├── src/ndswin/                # Core library
│   ├── model.py               # NDSwin model
│   ├── config.py              # Configuration dataclasses
│   └── training/              # Training utilities
├── train/                     # Training scripts
│   ├── train.py               # Main training script
│   ├── run_sweep.py           # Hyperparameter sweep
│   ├── auto_sweep_and_train.py # Sweep → train pipeline
│   ├── queue_runner.py        # Sequential job queue
│   └── fetch_hf_dataset.py   # Dataset downloader
├── tests/                     # Test suite
├── Makefile                   # All commands
├── pyproject.toml             # Package config
└── environment.yml            # Conda environment
```

## Testing

```bash
# Run full test suite (203 tests)
make test

# Quick check
make test-fast

# Full CI check (lint + type-check + test)
make check
```

## Citation

If you find this repository helpful, please consider citing it.
