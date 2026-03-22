"""Tests for training checkpoint utilities."""

import os
import tempfile
from pathlib import Path

import jax.numpy as jnp
import pytest

from ndswin.training.checkpoint import (
    CheckpointInfo,
    CheckpointManager,
    _flatten_dict,
    _set_nested,
)


def test_flatten_and_set_nested():
    """Test dictionary flattening and nesting utilities."""
    d = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
    flat = _flatten_dict(d)

    assert set(flat) == {
        (("a", "b"), 1),
        (("a", "c", "d"), 2),
        (("e",), 3),
    }

    nested = {}
    for keys, val in flat:
        _set_nested(nested, list(keys), val)

    assert nested == d


def test_checkpoint_info():
    """Test CheckpointInfo dataclass formatting."""
    info = CheckpointInfo(
        path="/tmp/ckpt_0.npz",
        step=10,
        epoch=1,
        metrics={"loss": 0.5},
        timestamp="2023-01-01T12:00:00",
    )
    assert info.step == 10
    assert info.epoch == 1
    assert "loss" in info.metrics


def test_checkpoint_manager_save_load():
    """Test basic saving and loading with CheckpointManager."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = CheckpointManager(directory=tmp_dir, max_to_keep=2)

        params = {"layer1": {"w": jnp.ones((2, 2)), "b": jnp.zeros((2,))}}
        batch_stats = {"layer1": {"mean": jnp.zeros((2,))}}
        metrics = {"loss": 0.1, "acc": 0.9}

        # Save checkpoint
        path1 = manager.save(
            params=params,
            step=10,
            epoch=1,
            metrics=metrics,
            batch_stats=batch_stats,
        )

        assert os.path.exists(path1)
        assert len(manager._checkpoints) == 1

        # Load checkpoint
        loaded = manager.load()
        assert loaded["step"] == 10
        assert loaded["epoch"] == 1
        assert "loss" in loaded["metrics"]
        assert loaded["metrics"]["loss"] == 0.1

        # Check params and batch stats
        assert jnp.allclose(loaded["params"]["layer1"]["w"], params["layer1"]["w"])
        assert jnp.allclose(loaded["batch_stats"]["layer1"]["mean"], batch_stats["layer1"]["mean"])


def test_checkpoint_manager_cleanup():
    """Test CheckpointManager cleans up old checkpoints."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = CheckpointManager(directory=tmp_dir, max_to_keep=2)
        params = {"w": jnp.ones(1)}

        paths = []
        for i in range(5):
            p = manager.save(params, step=i)
            paths.append(p)

        assert len(manager._checkpoints) == 2

        # Early checkpoints should be deleted
        for i in range(3):
            assert not os.path.exists(paths[i])

        # Latest 2 should exist
        for i in range(3, 5):
            assert os.path.exists(paths[i])


def test_checkpoint_manager_scan_existing():
    """Test that CheckpointManager finds existing checkpoints on init."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create dummy checkpoint files
        Path(tmp_dir, "ckpt_00000001.npz").touch()
        Path(tmp_dir, "ckpt_00000002.npz").touch()
        Path(tmp_dir, "other_file.txt").touch()

        manager = CheckpointManager(directory=tmp_dir)
        assert len(manager._checkpoints) == 2

        # Ensure it sorted by step
        steps = [c.step for c in manager._checkpoints]
        assert steps == [1, 2]


def test_checkpoint_manager_load_specific_step():
    """Test loading a specific step."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = CheckpointManager(directory=tmp_dir, max_to_keep=5)
        params = {"w": jnp.ones(1)}

        manager.save(params, step=10)
        manager.save(params, step=20)

        # Load specific step
        loaded = manager.load(step=10)
        assert loaded["step"] == 10

        with pytest.raises(ValueError, match="No checkpoint found"):
            manager.load(step=30)

        # Empty dir load
        empty_manager = CheckpointManager(directory=os.path.join(tmp_dir, "empty"))
        with pytest.raises(ValueError, match="No checkpoints found"):
            empty_manager.load()
