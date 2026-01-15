.PHONY: help install install-dev test lint format type-check docs clean clean-runs clean-all build check all dev train train-bg train-tmux list-configs sweep sweep-tmux auto-sweep auto-sweep-tmux queue queue-tmux fetch-data status

# Use conda env tools if available, otherwise system tools
CONDA_ENV := $(shell pwd)/envs/ndswin-jax
ifneq ($(wildcard $(CONDA_ENV)/bin/python),)
    PYTHON := $(CONDA_ENV)/bin/python -u
    PIP := $(CONDA_ENV)/bin/pip
    RUFF := $(CONDA_ENV)/bin/ruff
    MYPY := $(CONDA_ENV)/bin/mypy
    PYTEST := $(CONDA_ENV)/bin/pytest
    SPHINX := $(CONDA_ENV)/bin/sphinx-build
else
    PYTHON := python -u
    PIP := pip
    RUFF := ruff
    MYPY := mypy
    PYTEST := pytest
    SPHINX := sphinx-build
endif

# =============================================================================
# Configurable Variables (override with make VAR=value)
# =============================================================================
CONFIG        ?= configs/cifar100.json
SWEEP         ?= configs/sweeps/cifar100_hyperparam_sweep.yaml
QUEUE_FILE    ?= configs/queues/queue_2d_3d.yaml
TRAIN_EPOCHS  ?= 100
SWEEP_TRIALS  ?=
SWEEP_OUTDIR  ?= outputs/sweeps/$(shell basename $(SWEEP) .yaml)
HF_DATASET    ?= cifar100
DATASET_DIR   ?= data/$(subst /,_,$(HF_DATASET))
FETCH_LIMIT   ?=

# Derived
TRIALS_ARG    := $(if $(strip $(SWEEP_TRIALS)),--trials $(SWEEP_TRIALS),)
LIMIT_ARG     := $(if $(strip $(FETCH_LIMIT)),--limit $(FETCH_LIMIT),)
EXTRA_ARGS    ?=
HF_ENV        := HF_HOME=.hf_cache HF_DATASETS_CACHE=.hf_cache

# =============================================================================
# Help
# =============================================================================
help:
	@echo "NDSwin-JAX — N-Dimensional Swin Transformer"
	@echo ""
	@echo "Usage: make <target> [VAR=value ...]"
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
	@echo "  fetch-data         Fetch HF dataset (HF_DATASET=cifar100 DATASET_DIR=data/cifar100)"
	@echo "  list-configs       List available configs"
	@echo ""
	@echo "Development:"
	@echo "  test               Run tests"
	@echo "  test-fast          Run tests (stop on first failure)"
	@echo "  test-cov           Run tests with coverage"
	@echo "  lint / format      Lint / format code"
	@echo "  clean              Clean build caches"
	@echo "  clean-runs         Clean logs + outputs"
	@echo "  clean-all          Clean everything (logs, outputs, data caches)"
	@echo "  status             Show running tmux sessions"
	@echo ""
	@echo "Examples:"
	@echo "  make train CONFIG=configs/cifar100.json"
	@echo "  make sweep SWEEP=configs/sweeps/cifar100_hyperparam_sweep.yaml SWEEP_TRIALS=10"
	@echo "  make auto-sweep SWEEP=configs/sweeps/modelnet10_hyperparam_sweep.yaml TRAIN_EPOCHS=50"
	@echo "  make fetch-data HF_DATASET=jxie/modelnet40 DATASET_DIR=data/modelnet10"
	@echo "  make queue-tmux QUEUE_FILE=configs/queues/queue_2d_3d.yaml"

# =============================================================================
# Training
# =============================================================================
train:
	@if [ ! -f "$(CONFIG)" ]; then \
		echo "Error: Config not found: $(CONFIG)"; \
		echo "Available:"; ls -1 configs/*.json 2>/dev/null; exit 1; \
	fi
	@echo "Training with: $(CONFIG)"
	$(HF_ENV) $(PYTHON) train/train.py --config $(CONFIG) $(EXTRA_ARGS)

train-bg:
	@if [ ! -f "$(CONFIG)" ]; then echo "Error: Config not found: $(CONFIG)"; exit 1; fi
	@CONFIG_NAME=$$(basename $(CONFIG) .json); \
	STAMP="$${CONFIG_NAME}_$$(date +%Y%m%d_%H%M%S)"; \
	LOG="logs/$${STAMP}.log"; mkdir -p logs; \
	echo "Starting background training: $(CONFIG)"; \
	echo "Log: $${LOG}"; \
	nohup $(HF_ENV) $(PYTHON) train/train.py --config $(CONFIG) $(EXTRA_ARGS) > "$${LOG}" 2>&1 & \
	echo "PID: $$!"

train-tmux:
	@if [ ! -f "$(CONFIG)" ]; then echo "Error: Config not found: $(CONFIG)"; exit 1; fi
	@CONFIG_NAME=$$(basename $(CONFIG) .json); \
	STAMP="$${CONFIG_NAME}_$$(date +%Y%m%d_%H%M%S)"; \
	SESSION="train_$${STAMP}"; \
	LOG="logs/$${STAMP}.log"; mkdir -p logs; \
	echo "Starting tmux session: $${SESSION}"; \
	tmux new -d -s "$${SESSION}" "cd $(shell pwd) && $(HF_ENV) $(PYTHON) train/train.py --config $(CONFIG) $(EXTRA_ARGS) 2>&1 | tee '$${LOG}'; exec bash"; \
	echo "Attach: tmux attach -t $${SESSION}"; \
	echo "Log:    tail -f $${LOG}"

# =============================================================================
# Sweeps
# =============================================================================
sweep:
	@if [ ! -f "$(SWEEP)" ]; then echo "Error: Sweep not found: $(SWEEP)"; exit 1; fi
	@mkdir -p $(SWEEP_OUTDIR)
	@echo "Sweep: $(SWEEP) → $(SWEEP_OUTDIR)"
	$(HF_ENV) $(PYTHON) train/run_sweep.py --sweep $(SWEEP) $(TRIALS_ARG) --outdir $(SWEEP_OUTDIR) $(EXTRA_ARGS)

sweep-tmux:
	@if [ ! -f "$(SWEEP)" ]; then echo "Error: Sweep not found: $(SWEEP)"; exit 1; fi
	@STAMP=$$(date +%Y%m%d_%H%M%S); \
	SESSION="sweep_$${STAMP}"; LOG="logs/sweep_$${STAMP}.log"; mkdir -p logs; \
	echo "Starting sweep in tmux: $${SESSION}"; \
	tmux new -d -s "$${SESSION}" "cd $(shell pwd) && $(HF_ENV) $(PYTHON) train/run_sweep.py --sweep $(SWEEP) $(TRIALS_ARG) --outdir $(SWEEP_OUTDIR) $(EXTRA_ARGS) 2>&1 | tee '$${LOG}'; exec bash"; \
	echo "Attach: tmux attach -t $${SESSION}"; echo "Log: $${LOG}"

auto-sweep:
	@if [ ! -f "$(SWEEP)" ]; then echo "Error: Sweep not found: $(SWEEP)"; exit 1; fi
	@echo "Auto-sweep: $(SWEEP), train_epochs=$(TRAIN_EPOCHS)"
	$(HF_ENV) $(PYTHON) train/auto_sweep_and_train.py --sweep $(SWEEP) $(TRIALS_ARG) --train-epochs $(TRAIN_EPOCHS) $(EXTRA_ARGS)

auto-sweep-tmux:
	@if [ ! -f "$(SWEEP)" ]; then echo "Error: Sweep not found: $(SWEEP)"; exit 1; fi
	@STAMP=$$(date +%Y%m%d_%H%M%S); \
	SESSION="autosweep_$${STAMP}"; LOG="logs/autosweep_$${STAMP}.log"; mkdir -p logs; \
	echo "Starting auto-sweep in tmux: $${SESSION}"; \
	tmux new -d -s "$${SESSION}" "cd $(shell pwd) && $(HF_ENV) $(PYTHON) train/auto_sweep_and_train.py --sweep $(SWEEP) $(TRIALS_ARG) --train-epochs $(TRAIN_EPOCHS) $(EXTRA_ARGS) 2>&1 | tee '$${LOG}'; exec bash"; \
	echo "Attach: tmux attach -t $${SESSION}"; echo "Log: $${LOG}"

# =============================================================================
# Queue
# =============================================================================
queue:
	@if [ ! -f "$(QUEUE_FILE)" ]; then echo "Error: Queue not found: $(QUEUE_FILE)"; exit 1; fi
	@echo "Running queue: $(QUEUE_FILE)"
	$(HF_ENV) $(PYTHON) train/queue_runner.py --queue $(QUEUE_FILE) $(EXTRA_ARGS)

queue-tmux:
	@if [ ! -f "$(QUEUE_FILE)" ]; then echo "Error: Queue not found: $(QUEUE_FILE)"; exit 1; fi
	@STAMP=$$(date +%Y%m%d_%H%M%S); \
	SESSION="queue_$${STAMP}"; LOG="logs/queue_$${STAMP}.log"; mkdir -p logs; \
	echo "Starting queue in tmux: $${SESSION}"; \
	tmux new -d -s "$${SESSION}" "cd $(shell pwd) && $(HF_ENV) $(PYTHON) train/queue_runner.py --queue $(QUEUE_FILE) $(EXTRA_ARGS) 2>&1 | tee '$${LOG}'; exec bash"; \
	echo "Attach: tmux attach -t $${SESSION}"; echo "Log: $${LOG}"

# =============================================================================
# Data
# =============================================================================
fetch-data:
	@echo "Fetching: $(HF_DATASET) → $(DATASET_DIR)"
	$(HF_ENV) $(PYTHON) train/fetch_hf_dataset.py --hf-id "$(HF_DATASET)" --outdir "$(DATASET_DIR)" $(LIMIT_ARG)

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
# Cleanup
# =============================================================================
clean:
	rm -rf build/ dist/ *.egg-info/ src/*.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-runs:
	rm -rf logs/ outputs/ configs/auto_best/

clean-all: clean clean-runs
	rm -rf .hf_cache/

build: clean
	$(PYTHON) -m build

check: lint type-check test
	@echo "All checks passed!"

all: format lint type-check test docs build
	@echo "Full build complete!"

dev: install-dev
	@echo "Development environment ready!"
