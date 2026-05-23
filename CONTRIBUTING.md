# Contributing

This project keeps the package CLI and Makefile as the primary workflow surface.
Use the existing configs and tests as the source of truth for current behavior.

## Local Setup

```bash
make env
conda activate ./Environment/ndswin-jax
```

If the environment lives somewhere else, pass `CONDA_PREFIX_DIR` explicitly to
Make targets.

## Development Workflow

```bash
make test-fast
make check
```

For training, sweep, queue, or dataset changes, prefer the `ndswin` CLI and the
Makefile targets documented in `README.md`.

## Change Guidelines

- Keep example configs reusable for custom datasets.
- Do not commit local data, training outputs, checkpoints, or generated caches.
- Update tests and docs when CLI contracts, configs, model behavior, or result
  export formats change.
- Keep package metadata, README examples, and Makefile commands aligned.
