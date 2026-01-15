.PHONY: help install install-dev test lint format type-check docs clean build check all dev train train-bg train-tmux list-configs

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

# Training configuration
CONFIG ?= configs/cifar100.json
TRAIN_ARGS ?=

help:
	@echo "NDSwin-JAX Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Training Targets:"
	@echo "  train             Train model with config (CONFIG=path/to/config.json)"
	@echo "  train-bg          Train in background with nohup"
	@echo "  train-tmux        Train in detached tmux session"
	@echo "  list-configs      List available configuration files"
	@echo ""
	@echo "Development Targets:"
	@echo "  install           Install package"
	@echo "  install-dev       Install package with development dependencies"
	@echo "  test              Run tests"
	@echo "  test-cov          Run tests with coverage"
	@echo "  lint              Run linting"
	@echo "  format            Format code"
	@echo "  type-check        Run type checking"
	@echo "  docs              Build documentation"
	@echo "  clean             Clean build artifacts"
	@echo "  build             Build package"
	@echo "  check             Run all checks (lint, type-check, test)"
	@echo "  all               Full build (format, lint, type-check, test, docs, build)"
	@echo ""
	@echo "Training Examples:"
	@echo "  make train CONFIG=configs/cifar100.json"
	@echo "  make train-tmux CONFIG=configs/cifar100.json"
	@echo "  make train CONFIG=configs/cifar100.json TRAIN_ARGS='--epochs 50'"

# =============================================================================
# Training Targets
# =============================================================================

train:
	@if [ ! -f "$(CONFIG)" ]; then \
		echo "Error: Config file not found: $(CONFIG)"; \
		echo "Available configs:"; \
		ls -1 configs/*.json 2>/dev/null || echo "  No configs found in configs/"; \
		exit 1; \
	fi
	@echo "Training with config: $(CONFIG)"
	$(PYTHON) train/train.py --config $(CONFIG) $(TRAIN_ARGS)

train-bg:
	@if [ ! -f "$(CONFIG)" ]; then \
		echo "Error: Config file not found: $(CONFIG)"; \
		exit 1; \
	fi
	@CONFIG_NAME=$$(basename $(CONFIG) .json); \
	TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	STAMP="$${CONFIG_NAME}_$${TIMESTAMP}"; \
	LOG_FILE="logs/$${STAMP}.log"; \
	mkdir -p logs; \
	echo "Starting background training with config: $(CONFIG)"; \
	echo "Log file: $${LOG_FILE}"; \
	nohup $(PYTHON) train/train.py --config $(CONFIG) --stamp "$${STAMP}" --no-log-file $(TRAIN_ARGS) > "$${LOG_FILE}" 2>&1 & \
	echo "PID: $$!"; \
	echo "$$!" > ".train_pid_$${CONFIG_NAME}"; \
	echo "To monitor: tail -f $${LOG_FILE}"

train-tmux:
	@if [ ! -f "$(CONFIG)" ]; then \
		echo "Error: Config file not found: $(CONFIG)"; \
		exit 1; \
	fi
	@CONFIG_NAME=$$(basename $(CONFIG) .json); \
	TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	STAMP="$${CONFIG_NAME}_$${TIMESTAMP}"; \
	SESSION_NAME="train_$${STAMP}"; \
	LOG_FILE="logs/$${STAMP}.log"; \
	mkdir -p logs; \
	echo "Starting tmux training session: $${SESSION_NAME}"; \
	echo "Config: $(CONFIG)"; \
	echo "Log file: $${LOG_FILE}"; \
	tmux new -d -s "$${SESSION_NAME}" "cd $(shell pwd) && $(PYTHON) train/train.py --config $(CONFIG) --stamp \"$${STAMP}\" --no-log-file $(TRAIN_ARGS) 2>&1 | tee '$${LOG_FILE}'; exec bash"; \
	echo ""; \
	echo "Commands:"; \
	echo "  Attach:  tmux attach -t $${SESSION_NAME}"; \
	echo "  Monitor: tail -f $${LOG_FILE}"; \
	echo "  Kill:    tmux kill-session -t $${SESSION_NAME}"

# -----------------------------------------------------------------------------
# Sweep targets (generalized)
# -----------------------------------------------------------------------------
SWEEP ?= configs/sweeps/cifar100_hyperparam_sweep.yaml
SWEEP_TRIALS ?= $(TRIALS)
# Only include the --trials argument if SWEEP_TRIALS is set (avoids passing an empty flag)
TRIALS_ARG := $(if $(strip $(SWEEP_TRIALS)),--trials $(SWEEP_TRIALS),)
SWEEP_OUTDIR ?= outputs/sweeps/$(shell basename $(SWEEP) .yaml)
SWEEP_ARGS ?=

sweep:
	@if [ ! -f "$(SWEEP)" ]; then \
		echo "Error: Sweep file not found: $(SWEEP)"; \
		exit 1; \
	fi
	@mkdir -p $(SWEEP_OUTDIR)
	@echo "Running sweep: $(SWEEP), trials=$(SWEEP_TRIALS), outdir=$(SWEEP_OUTDIR)"
	HF_HOME=.hf_cache HF_DATASETS_CACHE=.hf_cache $(PYTHON) train/run_sweep.py --sweep $(SWEEP) $(TRIALS_ARG) --outdir $(SWEEP_OUTDIR) $(SWEEP_ARGS)

sweep-tmux:
	@if [ ! -f "$(SWEEP)" ]; then \
		echo "Error: Sweep file not found: $(SWEEP)"; \
		exit 1; \
	fi
	@STAMP=$$(date +%Y%m%d_%H%M%S); \
	SESSION_NAME="sweep_$${STAMP}"; \
	LOG_FILE="logs/sweep_$${STAMP}.log"; \
	mkdir -p logs; \
	echo "Starting sweep in tmux session: $${SESSION_NAME}"; \
	tmux new -d -s "$${SESSION_NAME}" "cd $(shell pwd) && HF_HOME=.hf_cache HF_DATASETS_CACHE=.hf_cache $(PYTHON) train/run_sweep.py --sweep $(SWEEP) $(TRIALS_ARG) --outdir $(SWEEP_OUTDIR) $(SWEEP_ARGS) 2>&1 | tee '$${LOG_FILE}'; exec bash"; \
	echo "To attach: tmux attach -t $${SESSION_NAME}"; \
	echo "Log: $${LOG_FILE}"

sweep-cifar100:
	@$(MAKE) sweep SWEEP=configs/sweeps/cifar100_hyperparam_sweep.yaml

sweep-medseg:
	@$(MAKE) sweep SWEEP=configs/sweeps/medseg_msd_hyperparam_sweep.yaml

# Generic dataset fetch and sweep targets
HF_DATASET ?= cifar100
DATASET_NAME ?= $(subst /,_,$(HF_DATASET))

fetch-dataset:
	@echo "Fetching HF dataset: $(HF_DATASET) -> data/$(DATASET_NAME)"
	@HF_HOME=.hf_cache HF_DATASETS_CACHE=.hf_cache envs/ndswin-jax/bin/python train/fetch_hf_dataset.py --hf-id "$(HF_DATASET)" --outdir "data/$(DATASET_NAME)" --limit 0

# Run a sweep against a base config; override SWEEP and BASE_CONFIG as needed
sweep-dataset:
	@if [ -z "$(SWEEP)" ]; then echo "Error: set SWEEP to the sweep YAML file"; exit 1; fi
	@STAMP=$$(date +%Y%m%d_%H%M%S); \
	OUT=outputs/sweeps/$(DATASET_NAME)_sweep_$${STAMP}; \
	mkdir -p "$${OUT}"; \
	echo "Running sweep: $(SWEEP) using BASE_CONFIG=$(BASE_CONFIG) outdir=$${OUT}"; \
	HF_DATASETS_CACHE=.hf_cache envs/ndswin-jax/bin/python train/run_sweep.py --sweep $(SWEEP) --base-config $(BASE_CONFIG) --outdir $${OUT} $(SWEEP_ARGS)

list-configs:
	@echo "Available configuration files:"
	@echo ""
	@for config in configs/*.json; do \
		if [ -f "$$config" ]; then \
			name=$$(grep -o '"name"[[:space:]]*:[[:space:]]*"[^"]*"' "$$config" | head -1 | cut -d'"' -f4); \
			desc=$$(grep -o '"description"[[:space:]]*:[[:space:]]*"[^"]*"' "$$config" | head -1 | cut -d'"' -f4); \
			printf "  %-35s %s\n" "$$config" "$$name"; \
			if [ -n "$$desc" ]; then \
				printf "  %-35s   -> %s\n" "" "$$desc"; \
			fi; \
		fi; \
	done
	@echo ""
	@echo "Usage: make train CONFIG=configs/<config_file>.json"

# =============================================================================
# Installation Targets
# =============================================================================

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,docs]"

test:
	$(PYTEST) tests/ -v

test-cov:
	$(PYTEST) tests/ -v --cov=ndswin --cov-report=html --cov-report=term-missing

test-fast:
	$(PYTEST) tests/ -v -x --tb=short

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

docs-serve:
	cd docs/_build/html && python -m http.server 8000

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	$(PYTHON) -m build

publish-test:
	twine upload --repository testpypi dist/*

publish:
	twine upload dist/*

# Development shortcuts
dev: install-dev
	@echo "Development environment ready!"

check: lint type-check test
	@echo "All checks passed!"

all: format lint type-check test docs build
	@echo "Full build complete!"
