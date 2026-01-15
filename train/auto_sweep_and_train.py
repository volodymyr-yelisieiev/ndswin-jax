#!/usr/bin/env python3
"""Convenience script to run a hyperparameter sweep and immediately train with the best parameters.

This script invokes `run_sweep.py` for N trials, parses the output summary JSON to find
the best configuration, and then executes `train.py` using that configuration.

Usage:
    python train/auto_sweep_and_train.py --sweep configs/sweeps/cifar100_hyperparam_sweep.yaml --trials 10
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_best_trial_from_summary(summary_path: Path) -> dict:
    """Read the summary JSON and find the trial with the best validation score."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Sweep summary not found at {summary_path}")

    with open(summary_path) as f:
        summary = json.load(f)

    if not summary:
        raise ValueError(f"Sweep summary is empty: {summary_path}")

    valid_trials = [t for t in summary if t.get("status") != "error" and t.get("status") != "dry"]
    if not valid_trials:
        raise ValueError(f"No successful trials found in sweep summary: {summary_path}")

    # Determine sorting metric based on task type
    # For segmentation, higher val_dice is better
    # For classification, higher val_accuracy is better
    best_trial = None

    if "val_dice" in valid_trials[0]:
        metric = "val_dice"
        best_trial = max(valid_trials, key=lambda t: t.get(metric, 0.0))
    elif "val_accuracy" in valid_trials[0]:
        metric = "val_accuracy"
        best_trial = max(valid_trials, key=lambda t: t.get(metric, 0.0))
    else:
        # Fallback to general loss if accuracy/dice not present (lower is better)
        metric = "loss"
        best_trial = min(valid_trials, key=lambda t: t.get("loss", float('inf')))
        
    print(f"\nBest trial found: Trial {best_trial.get('trial')} with {metric} = {best_trial.get(metric):.4f}")
    return best_trial


def main() -> None:
    parser = argparse.ArgumentParser("Run a sweep and immediately train on the best found config")
    parser.add_argument("--sweep", type=str, required=True, help="Path to sweep config (YAML/JSON)")
    parser.add_argument("--base-config", type=str, default=None, help="Optional base experiment JSON file")
    parser.add_argument("--trials", type=int, default=10, help="Number of sweep trials to run")
    parser.add_argument("--train-epochs", type=int, default=100, help="Number of epochs for the final training run")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory for the sweep")
    
    args = parser.parse_args()

    stamp = os.getenv("STAMP", "<stamped>")

    print("=" * 80)
    print("AUTOMATED SWEEP AND TRAIN WORKFLOW")
    print("=" * 80)
    print(f"1. Sweeping config: {args.sweep}")
    print(f"2. Number of trials: {args.trials}")
    print(f"3. Final training epochs: {args.train_epochs}")
    print("-" * 80)

    # 1. Run the sweep
    python_exe = sys.executable
    sweep_cmd = [
        python_exe, "train/run_sweep.py",
        "--sweep", args.sweep,
        "--trials", str(args.trials)
    ]
    if args.base_config:
        sweep_cmd.extend(["--base-config", args.base_config])
    if args.outdir:
        sweep_cmd.extend(["--outdir", args.outdir])

    print("\n>>> Launching Sweep...\n")
    print("Command:", " ".join(sweep_cmd))
    
    try:
        subprocess.run(sweep_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Sweep failed with exit code {e.returncode}. Aborting.")
        sys.exit(1)

    # 2. Locate the sweep summary
    # run_sweep.py produces a top-level summary.json in the outdir or default sweeps outdir
    default_outdir = args.outdir if args.outdir else "outputs/sweeps/unnamed"
    
    # Heuristically find the summary.json. We know `run_sweep.py` writes it to the top level.
    # Parse the sweep file to find the actual defined outdir if one wasn't explicitly passed.
    try:
        import yaml
        with open(args.sweep) as f:
            sweep_data = yaml.safe_load(f)
            sweep_outdir = sweep_data.get("output_dir", default_outdir)
    except Exception:
        import json
        try:
            with open(args.sweep) as f:
                sweep_data = json.load(f)
                sweep_outdir = sweep_data.get("output_dir", default_outdir)
        except Exception:
            sweep_outdir = default_outdir
            
    if args.outdir:
        sweep_outdir = args.outdir
        
    summary_path = Path(sweep_outdir) / "summary.json"
    
    if not summary_path.exists():
        print(f"\nERROR: Could not find sweep summary at {summary_path}. Cannot proceed to training.")
        sys.exit(1)

    # 3. Get the best trial config
    try:
        best_trial = get_best_trial_from_summary(summary_path)
    except Exception as e:
        print(f"\nERROR: Error parsing sweep summary: {e}")
        sys.exit(1)

    best_config_path = best_trial.get("config_path")
    if not best_config_path or not Path(best_config_path).exists():
        print(f"\nERROR: Best trial config not found at expected path: {best_config_path}")
        sys.exit(1)

    # Copy the best config to a permanent location
    final_config_dir = Path("configs/auto_best")
    final_config_dir.mkdir(parents=True, exist_ok=True)
    
    t_dataset = best_trial.get("dataset", "unknown")
    t_stamp = best_trial.get("stamp", "unstamped")
    t_trial = best_trial.get("trial", 0)
    
    final_config_path = final_config_dir / f"best_{t_dataset}_{t_stamp}_trial{t_trial:03d}.json"
    shutil.copy(best_config_path, final_config_path)
    
    print(f"\nExtracted Best Config.")
    print(f"Copied optimal configuration to: {final_config_path}")

    # 4. Launch final training using the best config
    print("\n" + "=" * 80)
    print(">>> Launching Final Full Training...")
    print("=" * 80 + "\n")

    train_cmd = [
        python_exe, "train/train.py",
        "--config", str(final_config_path),
        "--epochs", str(args.train_epochs)
    ]
    
    print("Command:", " ".join(train_cmd))
    
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Final training failed with exit code {e.returncode}.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("FULL WORKFLOW COMPLETED SUCCESSFULLY")
    print(f"Best Configuration: {final_config_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
