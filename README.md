# NDSwin-JAX

**N-Dimensional Swin Transformer implementation in JAX/Flax**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![JAX](https://img.shields.io/badge/JAX-0.7.1-purple.svg)](https://github.com/google/jax)
![Flax](https://img.shields.io/badge/Flax-0.10%2B-8A2BE2)
![Conda](https://img.shields.io/badge/environment-conda-44A833)

A pure JAX/Flax implementation of the Swin Transformer, generalized to support **N-dimensional data** (2D images, 3D volumes, 4D spatio-temporal) within a single, unified codebase.

## Features

- **N-Dimensional**: Train on 2D images, 3D voxels, or higher-dimensional data with no code changes — just configure `num_dims`.
- **Bring Your Own Data**: Easily plug in ANY custom dataset of any dimension, either from the **Hugging Face Hub** or a **local directory**.
- **Autonomous Optimization Pipeline**: Run `ndswin auto-sweep` directly, or use `make optimize` as a thin shortcut.
- **Multi-GPU Data Parallelism**: Automatically uses all available GPUs with JAX mesh sharding.
- **Hyperparameter Sweeps**: Built-in random search with early stopping, auto-training of best config.
- **Job Queue**: Run sequential sweep→train pipelines for multiple datasets automatically.
- **Point Cloud → Voxel**: Automatic voxelization of point cloud datasets.

## Published Results and Weights

The final restored-best checkpoints from the preserved practical-work artifact set are available on the Hugging Face Hub:

| Dataset | Final val top-1 | Test top-1 | Test top-5 | Weights |
|---|---:|---:|---:|---|
| CIFAR-10 | 91.82% | 89.95% | 99.36% | [ndswin-cifar10-final](https://huggingface.co/volodymyr-yelisieiev/ndswin-cifar10-final) |
| CIFAR-100 | 69.48% | 69.63% | 88.19% | [ndswin-cifar100-final](https://huggingface.co/volodymyr-yelisieiev/ndswin-cifar100-final) |
| ModelNet40 voxels | 74.21% | 69.65% | 91.49% | [ndswin-modelnet40-final](https://huggingface.co/volodymyr-yelisieiev/ndswin-modelnet40-final) |

All checkpoints were selected by validation accuracy. Test metrics were recomputed afterward from the restored checkpoint and are included in each model repository as `test_metrics.json`.

## Practical Work Reproduction

The Practical Work report is tied to the committed artifact bundle under
`report/data/sources/`. Those files are curated copies of the sweep summaries,
final metrics, test metrics, logs, selected configs, and ModelNet40 manifest used
for the published tables and figures. Regenerating the report data from this
bundle does not require local `outputs/` or `logs/` directories:

```bash
make env
conda activate ./Environment/ndswin-jax

make test-fast
make lint
make type-check

make -C report data
make -C report pdf
make -C report submission
```

For a lightweight pipeline sanity check instead of rerunning the full benchmark
campaign, use `make validate`. Full metric reproduction requires rerunning the
long sweep/retrain workloads on suitable CUDA hardware; the preserved report
artifacts provide the exact evidence used for the submitted PDF.

## Quick Start

### 1. Setup

```bash
git clone https://github.com/volodymyr-yelisieiev/ndswin-jax.git
cd ndswin-jax

# Official workflow: project-local environment
make env
conda activate ./Environment/ndswin-jax

# If you keep the prefix somewhere else, point Make at it explicitly:
make validate CONDA_PREFIX_DIR=/absolute/path/to/your/env
```

`make env` creates or updates `Environment/ndswin-jax`. If Conda/Mamba is unavailable, it bootstraps a local Miniforge under `.tools/` and installs the project with `dev`, `training`, and `gpu` extras. The GPU extra uses the current JAX CUDA 12 install style (`jax[cuda12]`), which supports the GTX 1080 Ti generation used for the preserved benchmark hardware.

The Makefile resolves `python`, `pip`, `pytest`, `ruff`, `mypy`, and `sphinx-build` from `Environment/ndswin-jax/bin` first, then falls back to the active shell `PATH` when that prefix is absent.

### 2. High-Level Optimization Shortcut

Find the best model for any dataset using a single command:

```bash
# `make optimize` is a thin shortcut over the package CLI.
# It fetches data, runs a sweep, selects the best trial by the configured metric,
# and trains the best config.
make optimize CONFIG=configs/your_config.json \
              SWEEP=configs/sweeps/your_sweep.yaml \
              HF_DATASET=your_hf_dataset

# Sweep summary files persist top-level selection metadata plus per-trial results.
```

### 3. Package CLI (`ndswin`)

The repository now ships a real package CLI so every workflow is available behind a single entrypoint:

```bash
ndswin train --config configs/cifar10.json
ndswin sweep --sweep configs/sweeps/cifar10_hyperparam_sweep.yaml --dry-run --trials 2
ndswin auto-sweep --sweep configs/sweeps/cifar10_hyperparam_sweep.yaml --train-epochs 50
ndswin queue --queue configs/queues/benchmark_2d_then_3d.yaml --dry-run
ndswin fetch-data --hf-id cifar10 --outdir data/cifar10 --limit 100
ndswin show-best
ndswin export-weights --run-dir outputs/cifar10/<run-stamp>
ndswin validate --config configs/cifar10.json --train-epochs 2
```

The `Makefile` delegates to the same package CLI, so `make train`, `make sweep`, and `make validate` stay compatible while avoiding direct `python train/...py` entrypoints. tmux-based targets remain optional wrappers, not the primary interface.

### 4. Manual Training and Sweeping

Train a model on a predefined configuration:

```bash
make train CONFIG=configs/cifar10.json
```

Or run a manual hyperparameter sweep:

```bash
# Run sweep (20 trials) for your custom config
make sweep SWEEP=configs/sweeps/my_custom_sweep.yaml SWEEP_TRIALS=20
```

### 5. Run a 2D→3D Queue

```bash
# Fetch the 40-class public example under the correct name
make fetch-data HF_DATASET=jxie/modelnet40 DATASET_DIR=data/modelnet40

# Run a sequential queue. `queue-tmux` is optional if you want a detached shell.
make queue QUEUE_FILE=configs/queues/benchmark_2d_then_3d.yaml

# Canonical detached practical-work benchmark:
make benchmark-tmux
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

For local N-D arrays organized as `data_dir/{split}/class_NNN/*.npy` or `.npz`:

```json
{
  "model": { "num_dims": 3 },
  "data": {
    "dataset": "array_folder",
    "data_dir": "data/modelnet40",
    "image_size": [32, 32, 32],
    "in_channels": 1,
    "task": "classification"
  }
}
```

`volume_folder` remains supported as a compatibility alias for 3D voxel experiments. Samples may be saved as `(*spatial)` for single-channel data or `(C, *spatial)` for explicit channel-first data; `.npz` files should preferably use the `image` key.

`ndswin fetch-data` now writes `dataset_manifest.json` under the exported dataset root. Training, sweep, and queue commands use that manifest or the on-disk class folders to reject class-count mismatches before the model starts.

## CLI and Makefile Reference

### `ndswin` subcommands

| Command | Description | Important Options |
|---|---|---|
| `ndswin train` | Train from a config JSON | `--config`, `--epochs`, `--batch-size`, `--lr`, `--data-dir`, `-o/--override` |
| `ndswin sweep` | Run a random-search sweep | `--sweep`, `--base-config`, `--trials`, `--dry-run`, `--outdir` |
| `ndswin auto-sweep` | Sweep, select best trial, then train it | `--sweep`, `--trials` (override only), `--train-epochs`, `--outdir` |
| `ndswin queue` | Execute fetch/train/sweep/auto-sweep jobs from a queue file | `--queue`, `--dry-run`, `--jobs`, `--retry`, `--skip-completed`, `--results-path` |
| `ndswin fetch-data` | Export a Hugging Face dataset into repo layout | `--hf-id`, `--outdir`, `--limit` |
| `ndswin show-best` | Print the best trial for each sweep summary | `--summary`, `--outputs-dir` |
| `ndswin export-weights` | Create a local HF-ready bundle without pushing | `--run-dir`, `--metrics`, `--checkpoint`, `--outdir`, `--hf-repo-id` |
| `ndswin validate` | Run tests plus a smoke training job | `--config`, `--train-epochs`, `--skip-tests`, `--skip-train`, `-o/--override` |

## Makefile Reference

All targets accept configurable overrides:

| Target | Description | Key Variables |
|--------|-------------|---------------|
| `train` | Train a model | `CONFIG` |
| `train-tmux` | Optional tmux wrapper for train | `CONFIG` |
| `sweep` | Run hyperparameter sweep | `SWEEP`, `SWEEP_TRIALS` |
| `auto-sweep` | Sweep + train best | `SWEEP`, `TRAIN_EPOCHS` |
| `auto-sweep-tmux` | Optional tmux wrapper for auto-sweep | `SWEEP`, `TRAIN_EPOCHS` |
| `queue` | Run job queue | `QUEUE_FILE` |
| `queue-tmux` | Optional tmux wrapper for queue | `QUEUE_FILE` |
| `benchmark` | Run the canonical 2D then 3D benchmark queue | `BENCHMARK_QUEUE`, `EXTRA_ARGS` |
| `benchmark-tmux` | Detached wrapper for `benchmark` | `BENCHMARK_QUEUE`, `EXTRA_ARGS` |
| `fetch-data` | Fetch HF dataset | `HF_DATASET`, `DATASET_DIR` |
| `test` | Run tests | — |
| `test-fast` | Tests (stop on first failure) | — |
| `clean` | Clean build caches | — |
| `clean-runs` | Clean logs + outputs | — |
| `clean-all` | Clean everything | — |
| `status` | Show running tmux sessions | — |
| `list-configs` | List all configs | — |

### Environment resolution

The repository standardizes on a project-local Conda prefix:

```bash
make env
conda activate ./Environment/ndswin-jax
```

By default, `make` resolves tools from `CONDA_PREFIX_DIR=Environment/ndswin-jax`. Override that variable only if you intentionally store the environment in another prefix:

```bash
make train CONFIG=configs/cifar10.json CONDA_PREFIX_DIR=/opt/conda/envs/ndswin-jax
```

### Examples

The framework provides configurations like `cifar10.json`, `modelnet10.json`, and `modelnet40.json` strictly as demonstrative examples. You should use them as templates to create configs for your own tasks.

```bash
# Full autonomous optimization for a custom 3D dataset
make optimize CONFIG=configs/my_3d_task.json SWEEP=configs/sweeps/my_sweep.yaml

# Train with your custom configuration
make train CONFIG=configs/my_custom_task.json

# Fetch any HF dataset to local storage
make fetch-data HF_DATASET=username/custom_dataset DATASET_DIR=data/custom_dataset

# Back up, then clean generated run outputs and logs
make backup
make clean-runs FORCE=1
```

## Project Structure

```
ndswin-jax/
├── configs/                    # Experiment configs
│   ├── cifar10.json          # 2D classification
│   ├── cifar100_tuned.json   # CIFAR-100 tuned 2D classification
│   ├── modelnet10.json        # True ModelNet10 template
│   ├── modelnet40.json        # ModelNet40 example
│   ├── sweeps/                # Hyperparameter sweep configs
│   └── queues/                # Job queue configs
├── src/ndswin/                # Core library + package CLI
│   ├── cli.py                 # Unified package CLI (`ndswin ...`)
│   ├── config.py              # Configuration dataclasses
│   └── training/              # Training utilities
├── train/                     # Thin compatibility wrappers over the package CLI
│   ├── train.py               # Wrapper for `ndswin train`
│   ├── run_sweep.py           # Wrapper for `ndswin sweep`
│   ├── auto_sweep_and_train.py # Wrapper for `ndswin auto-sweep`
│   ├── queue_runner.py        # Wrapper for `ndswin queue`
│   └── fetch_hf_dataset.py    # Wrapper for `ndswin fetch-data`
├── tests/                     # Test suite
├── Makefile                   # All commands
├── pyproject.toml             # Package config
└── environment.yml            # Conda environment
```

## Testing

```bash
# Run full test suite
make test

# Quick check
make test-fast

# Full CI check (lint + type-check + test)
make check
```

## Citation

If you find this repository helpful, please consider citing it.
