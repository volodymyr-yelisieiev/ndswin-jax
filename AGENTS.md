# NDSwin-JAX — Agent Instructions

> This file is the working contract for AI coding agents in this repository.
> Keep it aligned with the codebase, not with stale assumptions from older docs.

## 1. Mission

Help humans make safe, accurate changes to **NDSwin-JAX**, a JAX/Flax project for **N-dimensional Swin Transformer** training, sweeps, inference, and queue-driven experiment orchestration.

Priorities, in order:

1. **Be truthful about the current repo state.** Verify behavior from the code before documenting or promising it.
2. **Use the supported entrypoints.** In this repo, that means `make` targets first, and the package CLI (`ndswin`) underneath.
3. **Protect experiment artifacts.** Avoid destructive cleanup unless explicitly requested.
4. **Preserve generality.** The project is designed for arbitrary 2D/3D/N-D datasets, not just CIFAR examples.
5. **Leave the repo easier for the next agent.** Update docs/instructions when workflows change.

---

## 2. Golden Rules

### Rule A — Prefer `make`

For normal project workflows, **always start with a `make` target**.

Use:

- `make train`
- `make sweep`
- `make auto-sweep`
- `make queue`
- `make fetch-data`
- `make validate`
- `make test-fast`
- `make lint`
- `make format`
- `make type-check`

Avoid running project scripts such as:

- `python train/train.py`
- `python train/run_sweep.py`
- `python train/queue_runner.py`

Those files exist mainly as **thin compatibility wrappers** around the package CLI. The real implementation lives in `src/ndswin/cli.py`.

### Rule B — Treat `src/ndswin/cli.py` as the source of truth

If README text, old instructions, or comments conflict with runtime behavior, trust the current implementation in `src/ndswin/cli.py` and `Makefile`.

### Rule C — Do not assume the example configs are special

Files like `configs/cifar10.json`, `configs/modelnet10.json`, and the example sweep/queue files are **templates and smoke-test fixtures**, not the only intended workflows.

### Rule D — Never delete outputs casually

Do **not** manually remove `logs/`, `outputs/`, `configs/auto_best/`, or `.hf_cache/`.
Use the Make targets:

- `make backup`
- `make clean-logs FORCE=1`
- `make clean-runs FORCE=1`
- `make clean-all FORCE=1`

---

## 3. Current Repo Reality You Should Know

These points reflect the repository as it exists now:

- The canonical command surface is the **package CLI** `ndswin`, exposed by `pyproject.toml` and implemented in `src/ndswin/cli.py`.
- The `Makefile` delegates to `python -m ndswin.cli` with `PYTHONPATH=$(pwd)/src`.
- The main user-facing CLI subcommands are:
  - `train`
  - `sweep`
  - `auto-sweep`
  - `queue`
  - `fetch-data`
  - `show-best`
  - `validate`
- The repo still contains `train/*.py` entrypoints, but they are wrappers maintained for compatibility, not the preferred interface.
- Queue files may be **YAML or JSON** and currently support job types:
  - `train`
  - `sweep`
  - `auto-sweep`
- `make optimize` is the highest-level workflow: it tries to fetch data, runs `auto-sweep`, then runs `show-best`.
- `make validate` runs the CLI validation command, which by default runs **pytest first** and then a short smoke-training run.
- Hugging Face caches are intentionally localized via environment variables such as `HF_HOME=.hf_cache` and `HF_DATASETS_CACHE=.hf_cache`.
- Outputs are organized by dataset/stamp under `outputs/`, and logs are organized under `logs/`.

### Important nuance about the environment

The repository documents a **project-local Conda prefix** at `Environment/ndswin-jax`, and that remains the recommended layout.

However, when reasoning about actual behavior, note this distinction:

- The **documented workflow** is:
  ```bash
  conda env create --prefix ./Environment/ndswin-jax -f environment.yml
  conda activate ./Environment/ndswin-jax
  ```
- The **current Makefile behavior** assumes the required executables (`python`, `pytest`, `ruff`, etc.) are already available when `make` runs.

So in practice, agents should **activate the environment first** and then use `make`. Do not claim the Makefile fully bootstraps tool resolution for an inactive environment unless you verify that code path exists.

---

## 4. Recommended Agent Workflow

When asked to change code or docs, use this sequence:

```bash
# 1. Inspect the available commands
make help

# 2. Check the relevant implementation files
#    (read files directly; do not guess)

# 3. Make the smallest correct change

# 4. Run focused checks first
make test-fast
make lint
make type-check

# 5. If the change affects the full pipeline, run:
make validate
```

If full validation is too expensive, document exactly what you ran and why you stopped there.

---

## 5. Canonical Commands

### Environment setup

```bash
conda env create --prefix ./Environment/ndswin-jax -f environment.yml
conda activate ./Environment/ndswin-jax
```

### Main workflows

```bash
make optimize CONFIG=configs/your_config.json \
              SWEEP=configs/sweeps/your_sweep.yaml \
              HF_DATASET=your_hf_dataset

make train CONFIG=configs/your_config.json
make sweep SWEEP=configs/sweeps/your_sweep.yaml SWEEP_TRIALS=10
make auto-sweep SWEEP=configs/sweeps/your_sweep.yaml TRAIN_EPOCHS=50
make queue QUEUE_FILE=configs/queues/your_queue.yaml
make fetch-data HF_DATASET=username/dataset DATASET_DIR=data/your_dataset
make show-best
make tensorboard
make status
make stop-all
```

### Development workflows

```bash
make test
make test-fast
make test-cov
make lint
make format
make type-check
make docs
make build
make check
```

### Safe cleanup

```bash
make backup
make clean-logs FORCE=1
make clean-runs FORCE=1
make clean-all FORCE=1
```

---

## 6. How to Work with Configs, Sweeps, and Queues

### Configs

- Configs are JSON files.
- The core examples currently live in:
  - `configs/cifar10.json`
  - `configs/modelnet10.json`
- Prefer creating or editing task-specific configs instead of hard-coding assumptions around CIFAR.
- Preserve the repo's general N-D design:
  - `model.num_dims` controls the dimensionality
  - `data.image_size` should represent spatial dimensions
  - `data.in_channels` should represent channels separately

### Sweeps

- Sweep files may be YAML or JSON.
- The main examples live in `configs/sweeps/`.
- `make sweep` writes results under `outputs/sweeps/...` by default.
- `make auto-sweep` selects the best trial and trains it.
- `make optimize` wraps fetching + auto-sweep + show-best.

### Queues

- Queue files may be YAML or JSON.
- The current queue runner supports `train`, `sweep`, and `auto-sweep` jobs.
- Use queues when the human wants multi-stage or multi-dataset execution.
- Prefer `--dry-run` behavior from the CLI if you need to inspect generated commands safely.

---

## 7. Codebase Map

### High-level structure

- `src/ndswin/cli.py` — primary package CLI and orchestration logic.
- `src/ndswin/config.py` — configuration dataclasses and parsing.
- `src/ndswin/core/` — Swin building blocks, attention, window ops, patch embedding.
- `src/ndswin/models/` — model definitions and classifier/pretrained logic.
- `src/ndswin/training/` — trainer, data loading, augmentation, optimizer, losses, metrics, scheduler, checkpointing.
- `src/ndswin/inference/` — predictor/export/batch inference utilities.
- `src/ndswin/utils/` — logging, device, reproducibility, visualization, GPU helpers.
- `train/` — compatibility wrappers, not the preferred long-term integration point.
- `tests/` — unit/integration/smoke coverage, including CLI and Makefile behavior.
- `configs/` — example base configs, sweeps, and queues.

### Where to edit what

- **Command behavior changed?** Edit `src/ndswin/cli.py` first.
- **Training semantics changed?** Edit `src/ndswin/training/`.
- **Model architecture changed?** Edit `src/ndswin/core/` and/or `src/ndswin/models/`.
- **Config schema changed?** Edit `src/ndswin/config.py`, example configs, and affected tests/docs together.
- **Workflow docs changed?** Update `README.md` and this `AGENTS.md` together when appropriate.

---

## 8. Editing Conventions for Agents

### DO

- Verify claims from code before adding them to docs.
- Keep changes minimal and scoped.
- Update tests/docs when behavior changes.
- Preserve backward compatibility when wrappers or old entrypoints are still being supported.
- Prefer extending generic code paths rather than special-casing one dataset.
- Keep experiment paths, cache paths, and logging behavior consistent with existing patterns.

### DO NOT

- Do not hard-code CIFAR-only assumptions into reusable code.
- Do not bypass the package CLI when changing orchestration logic.
- Do not introduce a second competing workflow if `make` + `ndswin` already covers the task.
- Do not silently change output directory conventions.
- Do not rewrite large working subsystems unless the user asked for that level of refactor.

---

## 9. Testing Expectations

For agent-made changes, choose the smallest check set that meaningfully covers the edit.

### If you changed docs only

Usually run at least one or two lightweight sanity checks such as:

```bash
make help
make list-configs
```

If the docs mention tested behavior that has dedicated tests, run the most relevant test target that the environment can support.

### If you changed CLI, Makefile-facing behavior, or workflow docs tied to behavior

Prefer:

```bash
make test-fast
```

and, if needed:

```bash
make validate
```

### If you changed formatting or typing-sensitive Python code

Prefer:

```bash
make lint
make type-check
```

Always report exactly what you ran and whether any failure was due to environment limitations versus actual code problems.

---

## 10. Known Footguns

- `make validate` is not just a static check; it can launch a smoke training run.
- `make optimize` may fetch data and run long experiments. Do not launch it casually.
- `make clean-logs`, `make clean-runs`, and `make clean-all` are destructive without backups.
- `tmux`-based targets create detached sessions and log files; check `make status` before starting more sessions.
- The repository contains older wrapper scripts in `train/`; do not mistake them for the primary implementation layer.
- When updating documentation, avoid repeating claims about environment auto-resolution unless you confirmed the Makefile actually implements them.

---

## 11. How to Keep This File Healthy

Update this file when any of the following changes:

- `Makefile` targets or variables
- package CLI subcommands or flags
- output/logging/cache conventions
- queue/sweep/config formats
- environment setup expectations
- testing/lint/type-check workflows

When updating this file:

1. Read the relevant implementation files first.
2. Prefer precise language over aspirational language.
3. Remove stale guidance instead of piling new text on top of it.
4. If behavior is only partially implemented, say so plainly.

---

## 12. Short Version

If you remember only a few things, remember these:

- **Use `make`, not raw project scripts.** Don't call `python train/*.py` directly.
- **Treat `src/ndswin/cli.py` + `Makefile` as the truth.** Don't trust stale docs.
- **Keep the project generic across N-dimensional datasets.**
- **Don't delete experiment artifacts manually** — use `make backup` / `make clean-*`.
- **Activate the project environment before running `make`.**
- **Update this file when workflows change.**
