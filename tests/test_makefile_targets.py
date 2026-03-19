"""Tests for new Makefile targets and basic project structure."""

import subprocess
import sys
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
