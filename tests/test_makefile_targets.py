"""Tests for new Makefile targets and basic project structure."""

import os
import subprocess
from pathlib import Path


def test_makefile_help():
    """Verify 'make help' runs without error and lists key targets."""
    result = subprocess.run(["make", "help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "optimize" in result.stdout
    assert "tensorboard" in result.stdout
    assert "validate" in result.stdout
    assert "show-best" in result.stdout
    assert "stop-all" in result.stdout


def test_makefile_list_configs():
    """Verify 'make list-configs' runs and lists at least one config."""
    result = subprocess.run(["make", "list-configs"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "cifar10" in result.stdout.lower()


def test_agents_md_exists():
    """Verify AGENTS.md exists and contains key sections."""
    agents_path = Path("AGENTS.md")
    assert agents_path.exists(), "AGENTS.md not found in project root"
    content = agents_path.read_text()
    assert "make optimize" in content.lower() or "make optimize" in content
    assert "DON'T" in content or "DON\\'T" in content or "don't" in content.lower()
    assert "DO" in content


def test_makefile_resolves_project_local_conda_prefix(tmp_path):
    """Verify Makefile prefers CONDA_PREFIX_DIR/bin tools when present."""
    prefix_dir = tmp_path / "Environment" / "ndswin-jax"
    bin_dir = prefix_dir / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").write_text("#!/bin/sh\nexit 0\n")

    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    result = subprocess.run(
        ["make", "print-env-resolution", f"CONDA_PREFIX_DIR={prefix_dir}"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    stdout = result.stdout
    assert "resolver=conda-prefix" in stdout
    assert f"conda_prefix_dir={prefix_dir}" in stdout
    assert f"python_bin={bin_dir / 'python'}" in stdout
    assert f"pip={bin_dir / 'pip'}" in stdout
    assert f"pytest={bin_dir / 'pytest'}" in stdout
    assert f"ruff={bin_dir / 'ruff'}" in stdout


def test_makefile_falls_back_to_path_without_conda_prefix():
    """Verify Makefile falls back to PATH tools when no prefix python exists."""
    result = subprocess.run(
        ["make", "print-env-resolution", "CONDA_PREFIX_DIR=missing-prefix"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    stdout = result.stdout
    assert "resolver=system-path" in stdout
    assert "python_bin=python" in stdout
    assert "pip=pip" in stdout
    assert "pytest=pytest" in stdout
    assert "ruff=ruff" in stdout
