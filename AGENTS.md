# NDSwin-JAX — Agent Instructions

> This document tells AI coding agents how to use this project correctly.

## ★ Golden Rule

**Always use `make` commands. Never run Python scripts directly.**

## Official Environment Workflow

Use a **project-local Conda prefix** as the canonical environment layout:

```bash
conda env create --prefix ./Environment/ndswin-jax -f environment.yml
conda activate ./Environment/ndswin-jax
```

The Makefile resolves `python`, `pip`, `pytest`, `ruff`, `mypy`, and `sphinx-build` from `CONDA_PREFIX_DIR` and defaults that variable to `Environment/ndswin-jax`. Override `CONDA_PREFIX_DIR=/other/prefix` only when you intentionally keep the environment elsewhere.

## Recommended Workflow

```bash
# 1. Verify the pipeline is healthy
make validate

# 2. Run the full autonomous optimization pipeline
make optimize

# 3. Check the best result
make show-best

# 4. Visualize training curves
make tensorboard
```

## Quick Reference

| Task | Command |
|---|---|
| Full autonomous optimization | `make optimize CONFIG=... SWEEP=... HF_DATASET=...` |
| Quick optimization (fewer trials) | `make optimize SWEEP_TRIALS=5 TRAIN_EPOCHS=50` |
| Train a single config | `make train CONFIG=configs/your_config.json` |
| Run hyperparameter sweep | `make sweep SWEEP=configs/sweeps/your_sweep.yaml` |
| Sweep + auto-train best | `make auto-sweep SWEEP=configs/sweeps/your_sweep.yaml` |
| Run a job queue | `make queue QUEUE_FILE=configs/queues/my_queue.yaml` |
| Fetch dataset | `make fetch-data HF_DATASET=username/custom_dataset DATASET_DIR=...` |
| Check pipeline health | `make validate` |
| Show best sweep result | `make show-best` |
| Launch TensorBoard | `make tensorboard` |
| View available configs | `make list-configs` |
| View active sessions | `make status` |
| Create backup | `make backup` |
| Clean logs only | `make clean-logs FORCE=1` |
| Clean logs + outputs | `make clean-runs FORCE=1` |
| Stop all tmux sessions | `make stop-all` |
| Run tests | `make test-fast` |

## ✅ DO

- **Generalize Your Commands**: Do not blindly copy-paste `cifar10.json` examples. This framework is highly generic and designed for **ANY** custom dataset of ANY dimension. Build your own JSON configs and YAML sweeps tailored to the specific task you are solving and pass them to the Makefile.
- **Use `make optimize`** natively to fetch data, sweep, and train on the best configuration autonomouosly: `make optimize CONFIG=configs/your_task.json SWEEP=configs/sweeps/your_task_sweep.yaml HF_DATASET=your_hf_dataset`.
- **Use `make backup`** before any destructive cleanup.
- **Use `make validate`** to verify the pipeline works before long runs.
- **Use `make status`** to check if training sessions are already running.
- **Use `make stop-all`** to clean up stale tmux sessions before starting new runs.
- **Pre-fetch data** with `make fetch-data` before offline sweeps.
- **Override variables** on the command line to adjust behavior dynamically: `make optimize SWEEP_TRIALS=10 TRAIN_EPOCHS=200`.

## ❌ DON'T

- **Don't run `python train/train.py` directly** — use `make train`. The Makefile sets `HF_HOME`, `PYTHONPATH`, and uses the correct conda env automatically.
- **Don't delete `logs/` or `outputs/` manually** — use `make backup` then `make clean-runs FORCE=1`.
- **Don't modify sweep configs while a sweep is running** — check `make status` first.
- **Don't skip `make fetch-data`** before running in offline mode — the sweep uses `HF_HUB_OFFLINE=1`.
- **Don't use `pip install`** directly — use `make install` or `make install-dev`.

## Configuration Files

| File | Purpose |
|---|---|
| `configs/cifar10.json` | Example template for base configuration (model + training + data + augmentation) |
| `configs/sweeps/cifar10_hyperparam_sweep.yaml` | Example template for hyperparameter search space |
| `configs/queues/queue_2d_3d.yaml` | Example template for multi-job queue execution |

> Replace these examplary template configs with your own custom configurations when working on your unique tasks!

## Troubleshooting

| Problem | Solution |
|---|---|
| XLA deadlock during sweep | Fixed: each trial runs in isolated subprocess. Just re-run. |
| HuggingFace HTTP spam in logs | Run `make fetch-data` first, then sweep uses `HF_HUB_OFFLINE=1`. |
| Stale tmux sessions | `make stop-all` |
| GPU OOM | Reduce `EXTRA_ARGS="--batch-size 64"` |
| Tests failing | `make test-fast` to see first failure |

## Project Structure

```
ndswin-jax/
├── configs/           # JSON configs and YAML sweep/queue definitions
├── train/             # Training scripts (train.py, run_sweep.py, queue_runner.py)
├── src/ndswin/        # Core library (models, training, utils)
├── tests/             # Unit and integration tests
├── outputs/           # Training outputs, checkpoints, TensorBoard logs
├── logs/              # Run logs
├── backups/           # Timestamped backup archives
└── Makefile           # All commands (run `make help`)
```
