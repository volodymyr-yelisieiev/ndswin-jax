.PHONY: help install install-dev test lint format type-check docs clean clean-runs clean-all build check all dev train train-bg train-tmux list-configs sweep sweep-tmux auto-sweep auto-sweep-tmux queue queue-tmux fetch-data status archive-results optimize tensorboard stop-all show-best validate backup clean-logs print-env-resolution

# =============================================================================
# Configurable Variables (override with make VAR=value)
# =============================================================================
CONFIG        ?= configs/cifar10.json
SWEEP         ?= configs/sweeps/cifar10_hyperparam_sweep.yaml
QUEUE_FILE    ?= configs/queues/queue_2d_3d.yaml
TRAIN_EPOCHS  ?= 100
SWEEP_TRIALS  ?=
SWEEP_OUTDIR  ?= outputs/sweeps/$(shell basename $(SWEEP) .yaml)
HF_DATASET    ?= cifar10
DATASET_DIR   ?= data/$(subst /,_,$(HF_DATASET))
FETCH_LIMIT   ?=
TRIAL_TIMEOUT ?= 7200

# Toolchain resolution: prefer a project-local conda prefix when present.
# Override with: make CONDA_PREFIX_DIR=/your/prefix
CONDA_PREFIX_DIR ?= Environment/ndswin-jax
CONDA_BIN_DIR    := $(if $(wildcard $(CONDA_PREFIX_DIR)/bin/python),$(CONDA_PREFIX_DIR)/bin,)
PYTHON_BIN       := $(if $(CONDA_BIN_DIR),$(CONDA_BIN_DIR)/python,python)
PIPG             := $(if $(CONDA_BIN_DIR),$(CONDA_BIN_DIR)/pip,pip)
PYTEST_BIN       := $(if $(CONDA_BIN_DIR),$(CONDA_BIN_DIR)/pytest,pytest)
RUFF_BIN         := $(if $(CONDA_BIN_DIR),$(CONDA_BIN_DIR)/ruff,ruff)
MYPY_BIN         := $(if $(CONDA_BIN_DIR),$(CONDA_BIN_DIR)/mypy,mypy)
SPHINX_BIN       := $(if $(CONDA_BIN_DIR),$(CONDA_BIN_DIR)/sphinx-build,sphinx-build)
RESOLVED_TOOLCHAIN := $(if $(CONDA_BIN_DIR),conda-prefix,system-path)

# Expose the canonical names used throughout the Makefile
PYTHON  := $(PYTHON_BIN)
PIP     := $(PIPG)
PYTEST  := $(PYTEST_BIN)
RUFF    := $(RUFF_BIN)
MYPY    := $(MYPY_BIN)
SPHINX  := $(SPHINX_BIN)

# Derived
TRIALS_ARG    := $(if $(strip $(SWEEP_TRIALS)),--trials $(SWEEP_TRIALS),)
LIMIT_ARG     := $(if $(strip $(FETCH_LIMIT)),--limit $(FETCH_LIMIT),)
EXTRA_ARGS    ?=
FORCE         ?=
HF_ENV        := HF_HOME=.hf_cache HF_DATASETS_CACHE=.hf_cache
NDSWIN        := PYTHONPATH=$(shell pwd)/src $(PYTHON) -m ndswin.cli

# =============================================================================
# Help
# =============================================================================
help:
	@echo "NDSwin-JAX — N-Dimensional Swin Transformer"
	@echo ""
	@echo "Usage: make <target> [VAR=value ...]"
	@echo ""
	@echo "★ Quick Start:"
	@echo "  conda env create --prefix ./$(CONDA_PREFIX_DIR) -f environment.yml"
	@echo "  conda activate ./$(CONDA_PREFIX_DIR)"
	@echo "  optimize           One-command: fetch data → sweep → train best (THE recommended workflow)"
	@echo "  validate           Smoke test: 2-epoch train + tests to verify pipeline health"
	@echo ""
	@echo "Environment:"
	@echo "  CONDA_PREFIX_DIR=$(CONDA_PREFIX_DIR) (override if your project-local Conda prefix lives elsewhere)"
	@echo ""
	@echo "Training:"
	@echo "  train              Train (CONFIG=path/to/config.json)"
	@echo "  train-bg           Train in background (nohup)"
	@echo "  train-tmux         Train in detached tmux session"
	@echo ""
	@echo "Sweeps:"
	@echo "  sweep              Run hyperparameter sweep (SWEEP=path/to/sweep.yaml)"
	@echo "  sweep-tmux         Run sweep in tmux"
	@echo "  auto-sweep         Sweep + train best config"
	@echo "  auto-sweep-tmux    Auto-sweep in tmux"
	@echo ""
	@echo "Queue:"
	@echo "  queue              Run job queue (QUEUE_FILE=path/to/queue.yaml)"
	@echo "  queue-tmux         Run queue in tmux"
	@echo ""
	@echo "Data:"
	@echo "  fetch-data         Fetch HF dataset (HF_DATASET=cifar10 DATASET_DIR=data/cifar10)"
	@echo "  list-configs       List available configs"
	@echo ""
	@echo "Monitoring:"
	@echo "  tensorboard        Launch TensorBoard on outputs/"
	@echo "  show-best          Show best trial from latest sweep results"
	@echo "  status             Show running tmux sessions"
	@echo ""
	@echo "Development:"
	@echo "  test               Run tests"
	@echo "  test-fast          Run tests (stop on first failure)"
	@echo "  test-cov           Run tests with coverage"
	@echo "  lint / format      Lint / format code"
	@echo "  clean              Clean build caches"
	@echo "  clean-logs         Clean only logs (keep outputs)"
	@echo "  clean-runs         Clean logs + outputs"
	@echo "  clean-all          Clean everything (logs, outputs, data caches)"
	@echo "  stop-all           Kill all tmux sessions"
	@echo ""
	@echo "Safety:"
	@echo "  backup             Create timestamped backup of logs + outputs"
	@echo "  archive-results    (alias for backup)"
	@echo ""
	@echo "Examples:"
	@echo "  make optimize                                          # Full autonomous pipeline"
	@echo "  make optimize SWEEP_TRIALS=5 TRAIN_EPOCHS=50          # Quick optimization"
	@echo "  make train CONFIG=configs/cifar10.json"
	@echo "  make sweep SWEEP=configs/sweeps/cifar10_hyperparam_sweep.yaml SWEEP_TRIALS=10"
	@echo "  make auto-sweep SWEEP=configs/sweeps/modelnet10_hyperparam_sweep.yaml TRAIN_EPOCHS=50"
	@echo "  make fetch-data HF_DATASET=jxie/modelnet40 DATASET_DIR=data/modelnet10"
	@echo "  make queue-tmux QUEUE_FILE=configs/queues/queue_2d_3d.yaml"
	@echo "  make tensorboard"

print-env-resolution:
	@echo "resolver=$(RESOLVED_TOOLCHAIN)"
	@echo "conda_prefix_dir=$(CONDA_PREFIX_DIR)"
	@echo "conda_bin_dir=$(CONDA_BIN_DIR)"
	@echo "python_bin=$(PYTHON_BIN)"
	@echo "pip=$(PIP)"
	@echo "ruff=$(RUFF)"
	@echo "pytest=$(PYTEST)"
	@echo "mypy=$(MYPY)"
	@echo "sphinx=$(SPHINX)"

# =============================================================================
# ★ One-Command Optimization (the recommended workflow)
# =============================================================================
optimize:
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo "  NDSwin-JAX — Autonomous Optimization Pipeline"
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "  Config:  $(CONFIG)"
	@echo "  Sweep:   $(SWEEP)"
	@echo "  Trials:  $(or $(SWEEP_TRIALS),default from sweep config)"
	@echo "  Epochs:  $(TRAIN_EPOCHS)"
	@echo ""
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Step 1/4: Fetching data..."
	@$(MAKE) fetch-data --no-print-directory 2>/dev/null || echo "  (data already available or fetch skipped)"
	@echo ""
	@echo "Step 2/4: Running hyperparameter sweep..."
	$(HF_ENV) HF_HUB_OFFLINE=1 $(NDSWIN) auto-sweep \
		--sweep $(SWEEP) $(TRIALS_ARG) --train-epochs $(TRAIN_EPOCHS) $(EXTRA_ARGS)
	@echo ""
	@echo "Step 3/4: Showing best result..."
	@$(MAKE) show-best --no-print-directory 2>/dev/null || true
	@echo ""
	@echo "Step 4/4: Done!"
	@echo ""
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo "  ✓ Optimization complete. Best config saved to configs/auto_best/"
	@echo "  Run 'make tensorboard' to visualize training curves."
	@echo "  Run 'make show-best' to see the best hyperparameters."
	@echo "══════════════════════════════════════════════════════════════════════"

# =============================================================================
# Monitoring
# =============================================================================
tensorboard:
	@echo "Launching TensorBoard on outputs/ ..."
	@echo "Open http://localhost:6006 in your browser"
	$(PYTHON) -m tensorboard.main --logdir outputs/ --bind_all

show-best:
	$(NDSWIN) show-best

stop-all:
	@echo "Stopping all NDSwin tmux sessions..."
	@tmux ls 2>/dev/null | grep -E "^(train_|sweep_|autosweep_|queue_)" | cut -d: -f1 | while read s; do \
		echo "  Killing: $$s"; \
		tmux kill-session -t "$$s" 2>/dev/null; \
	done || echo "  No active sessions found."
	@echo "Done."

# =============================================================================
# Validation (smoke test)
# =============================================================================
validate:
	$(HF_ENV) $(NDSWIN) validate --config $(CONFIG) --train-epochs 2 --no-log-file $(EXTRA_ARGS)

# =============================================================================
# Training
# =============================================================================
train:
	@if [ ! -f "$(CONFIG)" ]; then \
		echo "Error: Config not found: $(CONFIG)"; \
		echo "Available:"; ls -1 configs/*.json 2>/dev/null; exit 1; \
	fi
	@echo "Training with: $(CONFIG)"
	$(HF_ENV) $(NDSWIN) train --config $(CONFIG) $(EXTRA_ARGS)

train-bg:
	@if [ ! -f "$(CONFIG)" ]; then echo "Error: Config not found: $(CONFIG)"; exit 1; fi
	@CONFIG_NAME=$$(basename $(CONFIG) .json); \
	STAMP="$${CONFIG_NAME}_$$(date +%Y%m%d_%H%M%S)"; \
	LOG="logs/$${STAMP}.log"; mkdir -p logs; \
	echo "Starting background training: $(CONFIG)"; \
	echo "Log: $${LOG}"; \
	nohup $(HF_ENV) $(NDSWIN) train --config $(CONFIG) $(EXTRA_ARGS) > "$${LOG}" 2>&1 & \
	echo "PID: $$!"

train-tmux:
	@if [ ! -f "$(CONFIG)" ]; then echo "Error: Config not found: $(CONFIG)"; exit 1; fi
	@CONFIG_NAME=$$(basename $(CONFIG) .json); \
	STAMP="$${CONFIG_NAME}_$$(date +%Y%m%d_%H%M%S)"; \
	SESSION="train_$${STAMP}"; \
	LOG="logs/$${STAMP}.log"; mkdir -p logs; \
	echo "Starting tmux session: $${SESSION}"; \
	tmux new -d -s "$${SESSION}" "cd $(shell pwd) && $(HF_ENV) $(NDSWIN) train --config $(CONFIG) $(EXTRA_ARGS) 2>&1 | tee '$${LOG}'; exec bash"; \
	echo "Attach: tmux attach -t $${SESSION}"; \
	echo "Log:    tail -f $${LOG}"

# =============================================================================
# Sweeps
# =============================================================================
sweep:
	@if [ ! -f "$(SWEEP)" ]; then echo "Error: Sweep not found: $(SWEEP)"; exit 1; fi
	@mkdir -p $(SWEEP_OUTDIR)
	@echo "Sweep: $(SWEEP) → $(SWEEP_OUTDIR)"
	$(HF_ENV) $(NDSWIN) sweep --sweep $(SWEEP) $(TRIALS_ARG) --outdir $(SWEEP_OUTDIR) $(EXTRA_ARGS)

sweep-tmux:
	@if [ ! -f "$(SWEEP)" ]; then echo "Error: Sweep not found: $(SWEEP)"; exit 1; fi
	@STAMP=$$(date +%Y%m%d_%H%M%S); \
	SESSION="sweep_$${STAMP}"; LOG="logs/sweep_$${STAMP}.log"; mkdir -p logs; \
	echo "Starting sweep in tmux: $${SESSION}"; \
	tmux new -d -s "$${SESSION}" "cd $(shell pwd) && $(HF_ENV) $(NDSWIN) sweep --sweep $(SWEEP) $(TRIALS_ARG) --outdir $(SWEEP_OUTDIR) $(EXTRA_ARGS) 2>&1 | tee '$${LOG}'; exec bash"; \
	echo "Attach: tmux attach -t $${SESSION}"; echo "Log: $${LOG}"

auto-sweep:
	@if [ ! -f "$(SWEEP)" ]; then echo "Error: Sweep not found: $(SWEEP)"; exit 1; fi
	@echo "Auto-sweep: $(SWEEP), train_epochs=$(TRAIN_EPOCHS)"
	$(HF_ENV) $(NDSWIN) auto-sweep --sweep $(SWEEP) $(TRIALS_ARG) --train-epochs $(TRAIN_EPOCHS) $(EXTRA_ARGS)

auto-sweep-tmux:
	@if [ ! -f "$(SWEEP)" ]; then echo "Error: Sweep not found: $(SWEEP)"; exit 1; fi
	@STAMP=$$(date +%Y%m%d_%H%M%S); \
	SESSION="autosweep_$${STAMP}"; LOG="logs/autosweep_$${STAMP}.log"; mkdir -p logs; \
	echo "Starting auto-sweep in tmux: $${SESSION}"; \
	tmux new -d -s "$${SESSION}" "cd $(shell pwd) && $(HF_ENV) $(NDSWIN) auto-sweep --sweep $(SWEEP) $(TRIALS_ARG) --train-epochs $(TRAIN_EPOCHS) $(EXTRA_ARGS) 2>&1 | tee '$${LOG}'; exec bash"; \
	echo "Attach: tmux attach -t $${SESSION}"; echo "Log: $${LOG}"

# =============================================================================
# Queue
# =============================================================================
queue:
	@if [ ! -f "$(QUEUE_FILE)" ]; then echo "Error: Queue not found: $(QUEUE_FILE)"; exit 1; fi
	@echo "Running queue: $(QUEUE_FILE)"
	$(HF_ENV) $(NDSWIN) queue --queue $(QUEUE_FILE) $(EXTRA_ARGS)

queue-tmux:
	@if [ ! -f "$(QUEUE_FILE)" ]; then echo "Error: Queue not found: $(QUEUE_FILE)"; exit 1; fi
	@STAMP=$$(date +%Y%m%d_%H%M%S); \
	SESSION="queue_$${STAMP}"; LOG="logs/queue_$${STAMP}.log"; mkdir -p logs; \
	echo "Starting queue in tmux: $${SESSION}"; \
	tmux new -d -s "$${SESSION}" "cd $(shell pwd) && $(HF_ENV) $(NDSWIN) queue --queue $(QUEUE_FILE) $(EXTRA_ARGS) 2>&1 | tee '$${LOG}'; exec bash"; \
	echo "Attach: tmux attach -t $${SESSION}"; echo "Log: $${LOG}"

# =============================================================================
# Data
# =============================================================================
fetch-data:
	@echo "Fetching: $(HF_DATASET) → $(DATASET_DIR)"
	$(HF_ENV) $(NDSWIN) fetch-data --hf-id "$(HF_DATASET)" --outdir "$(DATASET_DIR)" $(LIMIT_ARG)

list-configs:
	@echo "Configs:"; echo ""
	@for f in configs/*.json; do \
		[ -f "$$f" ] || continue; \
		name=$$(grep -o '"name"[[:space:]]*:[[:space:]]*"[^"]*"' "$$f" | head -1 | cut -d'"' -f4); \
		desc=$$(grep -o '"description"[[:space:]]*:[[:space:]]*"[^"]*"' "$$f" | head -1 | cut -d'"' -f4); \
		printf "  %-35s %s\n" "$$f" "$$name — $$desc"; \
	done
	@echo ""; echo "Sweeps:"; echo ""
	@for f in configs/sweeps/*.yaml; do \
		[ -f "$$f" ] || continue; \
		printf "  %s\n" "$$f"; \
	done
	@echo ""; echo "Queues:"; echo ""
	@for f in configs/queues/*.yaml; do \
		[ -f "$$f" ] || continue; \
		printf "  %s\n" "$$f"; \
	done

status:
	@echo "Active tmux sessions:"; echo ""
	@tmux ls 2>/dev/null || echo "  (none)"

# =============================================================================
# Development
# =============================================================================
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,docs]"

test:
	$(PYTEST) tests/ -v

test-fast:
	$(PYTEST) tests/ -v -x --tb=short

test-cov:
	$(PYTEST) tests/ -v --cov=ndswin --cov-report=html --cov-report=term-missing

lint:
	$(RUFF) check src/ndswin tests train
	$(RUFF) format --check src/ndswin tests train

format:
	$(RUFF) format src/ndswin tests train
	$(RUFF) check --fix src/ndswin tests train

type-check:
	$(MYPY) src/ndswin --ignore-missing-imports

docs:
	$(SPHINX) -b html docs docs/_build/html

# =============================================================================
# Backup & Cleanup
# =============================================================================
backup: archive-results

archive-results:
	@STAMP=$$(date +%Y%m%d_%H%M%S); \
	BACKUP="backups/results_$${STAMP}.tar.gz"; \
	mkdir -p backups; \
	if [ -d outputs ] || [ -d logs ] || [ -d configs/auto_best ]; then \
		tar czf "$${BACKUP}" $$([ -d outputs ] && echo outputs) $$([ -d logs ] && echo logs) $$([ -d configs/auto_best ] && echo configs/auto_best) 2>/dev/null; \
		echo "Archived to: $${BACKUP} ($$(du -sh $${BACKUP} | cut -f1))"; \
	else \
		echo "Nothing to archive (no outputs/, logs/, or configs/auto_best/ found)"; \
	fi

clean:
	rm -rf build/ dist/ *.egg-info/ src/*.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-logs:
	@if [ -z "$(FORCE)" ]; then \
		echo "⚠️  This will delete all log files:"; \
		[ -d logs ] && echo "  - logs/ ($$(du -sh logs 2>/dev/null | cut -f1))"; \
		echo ""; \
		printf "Are you sure? [y/N] "; \
		read ans; \
		case "$$ans" in [yY]*) ;; *) echo "Aborted."; exit 1;; esac; \
	fi
	rm -rf logs/
	@echo "Cleaned logs/"

clean-runs:
	@if [ -z "$(FORCE)" ]; then \
		echo "⚠️  This will PERMANENTLY delete all training results:"; \
		[ -d logs ] && echo "  - logs/ ($$(du -sh logs 2>/dev/null | cut -f1))"; \
		[ -d outputs ] && echo "  - outputs/ ($$(du -sh outputs 2>/dev/null | cut -f1))"; \
		[ -d configs/auto_best ] && echo "  - configs/auto_best/"; \
		echo ""; \
		echo "Run 'make backup' first to create a backup."; \
		printf "Are you sure? [y/N] "; \
		read ans; \
		case "$$ans" in [yY]*) ;; *) echo "Aborted."; exit 1;; esac; \
	fi
	rm -rf logs/ outputs/ configs/auto_best/
	@echo "Cleaned logs/, outputs/, configs/auto_best/"

clean-all: clean
	@if [ -z "$(FORCE)" ]; then \
		echo "⚠️  This will delete ALL caches, logs, outputs, and data caches."; \
		printf "Are you sure? [y/N] "; \
		read ans; \
		case "$$ans" in [yY]*) ;; *) echo "Aborted."; exit 1;; esac; \
	fi
	$(MAKE) clean-runs FORCE=1
	rm -rf .hf_cache/
	@echo "Cleaned everything."

build: clean
	$(PYTHON) -m build

check: lint type-check test
	@echo "All checks passed!"

all: format lint type-check test docs build
	@echo "Full build complete!"

dev: install-dev
	@echo "Development environment ready!"
