"""Tests for environment and GPU dependency configuration."""

from pathlib import Path


def test_gpu_extra_uses_current_jax_cuda12_name():
    """The GPU extra should use the current JAX CUDA 12 extra, not old cuda12_pip."""
    pyproject = Path("pyproject.toml").read_text()

    assert "jax[cuda12]" in pyproject
    assert "cuda12_pip" not in pyproject


def test_environment_installs_dev_training_gpu_extras():
    """The project env should include the extras needed for tests and benchmarks."""
    environment = Path("environment.yml").read_text()

    assert "-e .[dev,training,gpu]" in environment
    assert "python=3.11" in environment
