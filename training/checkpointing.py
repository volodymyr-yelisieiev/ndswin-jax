"""Checkpoint management for model training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from flax.training import train_state
import orbax.checkpoint as ocp


class CheckpointManager:
    """Manages saving and loading of model checkpoints.
    
    This class provides a high-level interface for checkpoint operations,
    including saving, loading, and managing checkpoint directories.
    
    Attributes:
        workdir: Base directory for checkpoints and logs
        keep: Maximum number of checkpoints to keep
    """

    def __init__(self, workdir: Path, keep: int = 3):
        """Initialize checkpoint manager.
        
        Args:
            workdir: Base directory for all training artifacts
            keep: Maximum number of checkpoints to retain
        """
        workdir_path = Path(workdir).expanduser()

        # Ensure Orbax sees an absolute checkpoint path even if the CLI passed a relative workdir.
        if not workdir_path.is_absolute():
            workdir_path = Path.cwd() / workdir_path

        workdir_path.mkdir(parents=True, exist_ok=True)
        self.workdir = workdir_path.resolve()
        self.checkpoint_dir = self.workdir / "checkpoints"
        self.best_checkpoint_dir = self.workdir / "best_checkpoint"
        self.keep = keep
        
        # Create directories if they don't exist
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.best_checkpoint_dir.mkdir(exist_ok=True)
        
        # Create Orbax checkpoint managers using new API
        self.ckpt_manager = ocp.CheckpointManager(
            self.checkpoint_dir,
            item_names=('state',),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=keep,
                create=True,
            ),
        )
        self.best_ckpt_manager = ocp.CheckpointManager(
            self.best_checkpoint_dir,
            item_names=('state',),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=1,
                create=True,
            ),
        )

    def save(self, state: train_state.TrainState, step: Optional[int] = None, is_best: bool = False) -> Path:
        """Save a checkpoint.
        
        Args:
            state: Training state to save
            step: Training step number (uses state.step if None)
            is_best: If True, save to best_checkpoint directory
            
        Returns:
            Path to the saved checkpoint
        """
        if step is None:
            step = int(state.step)
        
        if is_best:
            # Save best model using new Orbax API
            self.best_ckpt_manager.save(step, args=ocp.args.Composite(state=ocp.args.StandardSave(state)))
            self.best_ckpt_manager.wait_until_finished()
        else:
            # Save regular checkpoint using new Orbax API
            self.ckpt_manager.save(step, args=ocp.args.Composite(state=ocp.args.StandardSave(state)))
            self.ckpt_manager.wait_until_finished()
        
        return self.checkpoint_dir / f"checkpoint_{step}"

    def restore(self, state: train_state.TrainState, step: Optional[int] = None) -> train_state.TrainState:
        """Restore a checkpoint.
        
        Args:
            state: Template training state (for structure)
            step: Specific step to restore (restores latest if None)
            
        Returns:
            Restored training state
        """
        if step is None:
            step = self.ckpt_manager.latest_step()
        
        if step is None:
            # No checkpoint found, return original state
            return state
        
        # Restore using new Orbax API
        restored_dict = self.ckpt_manager.restore(step, args=ocp.args.Composite(state=ocp.args.StandardRestore(state)))
        return restored_dict.state

    def latest_step(self) -> Optional[int]:
        """Get the step number of the latest checkpoint.
        
        Returns:
            Latest checkpoint step, or None if no checkpoints exist
        """
        return self.ckpt_manager.latest_step()

    def save_config(self, config: Any) -> Path:
        """Save configuration to JSON file.
        
        Args:
            config: Configuration object (must be JSON-serializable)
            
        Returns:
            Path to the saved config file
        """
        config_path = self.workdir / "config.json"
        
        # Convert to dict if needed
        config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        
        return config_path

    def save_metrics(self, metrics: dict, append: bool = True) -> Path:
        """Save training metrics to JSONL file.
        
        Args:
            metrics: Dictionary of metrics to save
            append: If True, append to existing file; otherwise overwrite
            
        Returns:
            Path to the metrics file
        """
        metrics_path = self.workdir / "metrics.jsonl"
        mode = "a" if append else "w"
        
        with metrics_path.open(mode, encoding="utf-8") as f:
            json.dump(metrics, f)
            f.write("\n")
        
        return metrics_path

    def list_checkpoints(self) -> list[int]:
        """List all checkpoint steps in the checkpoint directory.
        
        Returns:
            List of checkpoint step numbers, sorted
        """
        return sorted(self.ckpt_manager.all_steps())

    def clear_checkpoints(self) -> None:
        """Remove all checkpoints from the checkpoint directory."""
        for step in self.list_checkpoints():
            self.ckpt_manager.delete(step)


__all__ = ["CheckpointManager"]
