"""Checkpoint management for training.

This module provides checkpoint utilities specifically for training.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from ndswin.types import PyTree


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""

    path: str
    step: int
    epoch: int
    metrics: dict[str, float] | None = None
    timestamp: str | None = None


class CheckpointManager:
    """Manages training checkpoints with absolute path handling."""

    def __init__(
        self,
        directory: str,
        max_to_keep: int = 5,
        keep_best: int = 3,
        prefix: str = "ckpt",
    ) -> None:
        self.directory = Path(directory).resolve()
        self.max_to_keep = max_to_keep
        self.keep_best = keep_best
        self.prefix = prefix
        self.directory.mkdir(parents=True, exist_ok=True)
        self._checkpoints: list[CheckpointInfo] = []
        self._best_checkpoints: dict[str, list[CheckpointInfo]] = {}
        self._scan_existing()

    def _scan_existing(self) -> None:
        """Scan directory for existing checkpoints."""
        for path in self.directory.glob(f"{self.prefix}_*.npz"):
            try:
                parts = path.stem.split("_")
                step = int(parts[-1])
                self._checkpoints.append(CheckpointInfo(path=str(path), step=step, epoch=0))
            except (ValueError, IndexError):
                pass
        self._checkpoints.sort(key=lambda x: x.step)

    def save(
        self,
        params: PyTree,
        step: int,
        epoch: int = 0,
        metrics: dict[str, float] | None = None,
        batch_stats: PyTree | None = None,
        optimizer_state: PyTree | None = None,
    ) -> str:
        """Save a checkpoint."""
        import datetime

        filename = f"{self.prefix}_{step:08d}.npz"
        path = self.directory / filename

        flat_params = {
            f"params/{'/'.join(str(k) for k in keys)}": np.array(val)
            for keys, val in _flatten_dict(params)
        }

        if batch_stats is not None:
            flat_stats = {
                f"batch_stats/{'/'.join(str(k) for k in keys)}": np.array(val)
                for keys, val in _flatten_dict(batch_stats)
            }
            flat_params.update(flat_stats)

        metadata = {"step": np.array(step), "epoch": np.array(epoch)}
        if metrics is not None:
            for name, value in metrics.items():
                metadata[f"metrics/{name}"] = np.array(value)
        flat_params.update(metadata)

        np.savez(path, **flat_params)
        info = CheckpointInfo(
            path=str(path),
            step=step,
            epoch=epoch,
            metrics=metrics,
            timestamp=datetime.datetime.now().isoformat(),
        )
        self._checkpoints.append(info)
        self._cleanup()
        return str(path)

    def load(self, path: str | None = None, step: int | None = None) -> dict[str, Any]:
        """Load a checkpoint."""
        if path is None:
            if step is not None:
                for ckpt in self._checkpoints:
                    if ckpt.step == step:
                        path = ckpt.path
                        break
                if path is None:
                    raise ValueError(f"No checkpoint found at step {step}")
            else:
                if not self._checkpoints:
                    raise ValueError("No checkpoints found")
                path = self._checkpoints[-1].path

        data = dict(np.load(path, allow_pickle=True))
        params: dict[str, Any] = {}
        batch_stats: dict[str, Any] = {}
        metrics: dict[str, float] = {}

        for key, value in data.items():
            if key.startswith("params/"):
                _set_nested(params, key[7:].split("/"), jnp.array(value))
            elif key.startswith("batch_stats/"):
                _set_nested(batch_stats, key[12:].split("/"), jnp.array(value))
            elif key.startswith("metrics/"):
                metrics[key[8:]] = float(value)

        result: dict[str, Any] = {
            "params": params,
            "step": int(data.get("step", 0)),
            "epoch": int(data.get("epoch", 0)),
        }
        if batch_stats:
            result["batch_stats"] = batch_stats
        if metrics:
            result["metrics"] = metrics
        return result

    def _cleanup(self) -> None:
        """Remove old checkpoints."""
        if len(self._checkpoints) <= self.max_to_keep:
            return

        to_keep = {ckpt.path for ckpt in self._checkpoints[-self.max_to_keep :]}

        for ckpt in self._checkpoints[: -self.max_to_keep]:
            if ckpt.path not in to_keep:
                if os.path.exists(ckpt.path):
                    os.remove(ckpt.path)

        self._checkpoints = self._checkpoints[-self.max_to_keep :]


def _flatten_dict(d: dict, parent_key: tuple = ()) -> list[tuple[tuple, Any]]:
    items = []
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key))
        else:
            items.append((new_key, v))
    return items


def _set_nested(d: dict, keys: list[str], value: Any) -> None:
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value
