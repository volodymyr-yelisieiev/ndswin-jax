#!/usr/bin/env python3
"""Small hyperparameter sweep driver (random search).

Usage:
    python train/run_sweep.py --sweep configs/sweeps/cifar100_hyperparam_sweep.yaml

This script samples hyperparameters, materializes per-trial ExperimentConfigs, and
runs short-budget training to evaluate each trial. It writes trial configs and
results into an output sweep directory.

Notes:
- Designed to work with the repository's `ndswin` API (no external scheduler required).
- Default behavior is sequential execution. Use --dry-run to only generate configs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover - best effort
    yaml = None

from ndswin import ExperimentConfig, NDSwinConfig, TrainingConfig, DataConfig, NDSwinTransformer
from ndswin.training import Trainer, create_data_loader


def load_sweep(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sweep file not found: {path}")
    if p.suffix in ('.yaml', '.yml'):
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML sweep files. Install pyyaml.")
        return yaml.safe_load(p.read_text())
    elif p.suffix == '.json':
        return json.loads(p.read_text())
    else:
        raise ValueError("Unsupported sweep file format. Use .yaml/.yml or .json")


def sample_value(spec: Dict[str, Any]) -> Any:
    kind = spec.get("kind", "choice")
    if kind == "choice":
        return random.choice(spec["values"])
    elif kind == "uniform":
        mn = float(spec["min"])
        mx = float(spec["max"])
        return random.uniform(mn, mx)
    elif kind == "log_uniform":
        mn = float(spec["min"])
        mx = float(spec["max"])
        # sample in log10 space
        return 10 ** random.uniform(math.log10(mn), math.log10(mx))
    else:
        raise ValueError(f"Unknown sampling kind: {kind}")


def set_by_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for p in parts[:-1]:
        cur = getattr(cur, p)
    last = parts[-1]
    # Convert lists to tuples if the attribute expects tuples sometimes (best-effort)
    # Let dataclasses handle strict validation on post-init where applicable.
    setattr(cur, last, value)


def materialize_experiment(base: ExperimentConfig, sampled: Dict[str, Any], budget_epochs: int) -> ExperimentConfig:
    exp = deepcopy(base)

    for k, v in sampled.items():
        set_by_path(exp, k, v)

    # Use short budget for the sweep
    exp.training.epochs = int(budget_epochs)
    exp.training.warmup_epochs = min(int(exp.training.warmup_epochs), exp.training.epochs // 10)

    # If dataset is specified and is cifar100, ensure num_classes matches; otherwise leave as-is
    if getattr(exp.data, "dataset", "").lower() == "cifar100":
        exp.training.num_classes = 100
        exp.model.num_classes = 100

    # Coerce numeric-looking strings to numeric types to avoid dataclass validation errors
    def _coerce_number(val):
        if isinstance(val, str):
            # Try integer then float
            try:
                return int(val)
            except Exception:
                try:
                    return float(val)
                except Exception:
                    return val
        return val

    # Training fields
    for fld in ("learning_rate", "min_learning_rate", "weight_decay", "ema_decay", "stochastic_depth_rate", "label_smoothing"):
        if hasattr(exp.training, fld):
            setattr(exp.training, fld, _coerce_number(getattr(exp.training, fld)))

    if isinstance(exp.training.batch_size, str):
        try:
            exp.training.batch_size = int(exp.training.batch_size)
        except Exception:
            pass

    # Data fields
    for fld in ("mixup_alpha", "cutmix_alpha", "cutout_size"):
        if hasattr(exp.data, fld):
            setattr(exp.data, fld, _coerce_number(getattr(exp.data, fld)))

    # Model fields
    if hasattr(exp.model, "drop_path_rate"):
        exp.model.drop_path_rate = _coerce_number(exp.model.drop_path_rate)

    # Validate model to ensure embed_dim divides cleanly by num_heads per stage.
    try:
        # Reconstruct a fresh NDSwinConfig from the existing one to trigger validation
        _ = NDSwinConfig.from_dict(exp.model.to_dict())
    except Exception as e:
        raise ValueError(f"Invalid model configuration after sampling: {e}")

    return exp


def load_base_experiment(path: str | None) -> ExperimentConfig:
    if path is None:
        cfg = ExperimentConfig()
        # Set sensible defaults for CIFAR-100
        cfg.data.dataset = "cifar100"
        cfg.data.data_dir = "data"
        cfg.data.image_size = (32, 32)
        cfg.data.in_channels = 3
        cfg.training.batch_size = 128
        cfg.training.epochs = 100
        cfg.training.learning_rate = 1e-3
        cfg.model = NDSwinConfig.swin_tiny_2d(num_classes=100)
        cfg.name = "cifar100_sweep_base"
        return cfg

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Base config not found: {path}")

    raw = json.loads(p.read_text())

    # Attempt to coerce existing config formats into ExperimentConfig.from_dict-friendly structure
    exp_dict: Dict[str, Any] = {}
    exp_dict["name"] = raw.get("name", "sweep_base")

    if "model" in raw:
        exp_dict["model"] = raw["model"]
    if "training" in raw:
        exp_dict["training"] = raw["training"]

    # Some configs use top-level 'dataset' and separate 'augmentation'
    data = raw.get("data") or raw.get("dataset") or {}
    aug = raw.get("augmentation", {})
    merged = {**data, **aug}
    # Remove keys that do not map to DataConfig constructor (e.g. 'name', 'num_classes')
    for _k in ("name", "description", "num_classes"):
        merged.pop(_k, None)

    # Some config files use 'input_shape' (channels + spatial dims). Convert to DataConfig fields.
    if "input_shape" in merged:
        inp = merged.pop("input_shape")
        if isinstance(inp, (list, tuple)) and len(inp) >= 2:
            merged["in_channels"] = int(inp[0])
            merged["image_size"] = tuple(int(x) for x in inp[1:])

    # Keep only fields that DataConfig accepts (drop others like crop_padding, color_jitter_strength)
    allowed_data_keys = {
        "dataset",
        "data_dir",
        "download",
        "image_size",
        "in_channels",
        "num_workers",
        "prefetch_size",
        "shuffle_buffer_size",
        "pin_memory",
        "drop_last",
        "train_split",
        "val_split",
        "test_split",
        "augmentation",
        "normalize",
        "mean",
        "std",
        "random_crop",
        "random_flip",
        "random_rotation",
        "color_jitter",
        "auto_augment",
        "mixup_alpha",
        "cutmix_alpha",
        "cutout_size",
    }
    merged = {k: v for k, v in merged.items() if k in allowed_data_keys}

    if merged:
        exp_dict["data"] = merged

    # Fallback: if training/model/data not present, still try to use keys at top level
    return ExperimentConfig.from_dict(exp_dict)


def run_trial(trial_idx: int, exp: ExperimentConfig, out_dir: Path, dry_run: bool = False) -> Dict[str, Any]:
    # Create a standardized per-trial output directory: out_dir/<dataset>/<stamp>/trial_XXX
    dataset_name = getattr(exp.data, "hf_id", None) or getattr(exp.data, "dataset", None) or "dataset"
    dataset_name = str(dataset_name).replace("/", "_").replace(":", "_")

    stamp = getattr(exp, "get_stamp", None)
    stamp = stamp() if callable(stamp) else "unstamped"

    base_dir = out_dir / dataset_name / stamp
    if base_dir.exists() and dry_run:
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    trial_dir = base_dir / f"trial_{trial_idx:03d}"
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=False)

    # Materialize and save config
    cfg_json = exp.to_dict()
    (trial_dir / "config.json").write_text(json.dumps(cfg_json, indent=2))

    if dry_run:
        print(f"[DRY] Trial {trial_idx} config written to {trial_dir}/config.json")
        return {"trial": trial_idx, "status": "dry", "config_path": str(trial_dir / "config.json"), "dataset": dataset_name, "stamp": stamp}

    # Create model and trainer (import heavy libs lazily)
    from ndswin import NDSwinTransformer
    from ndswin.training import Trainer, create_data_loader

    model = NDSwinTransformer(config=exp.model)
    trainer = Trainer(
        model=model,
        config=exp.training,
        seed=exp.training.seed,
        log_every=exp.training.log_interval,
        eval_every=exp.training.eval_interval,
        checkpoint_dir=str(trial_dir / "checkpoints"),
        task=getattr(exp.data, "task", "classification"),
        loss_name=getattr(exp.training, "loss", "cross_entropy"),
    )

    # Compute pad_to from model config so that patch embedding and patch merging
    # will not fail due to odd spatial sizes. Required per-dim size = patch_size[i] * 2^(num_stages-1)
    try:
        patch_size = exp.model.patch_size
        num_stages = exp.model.num_stages
        pad_to = tuple(int(patch_size[i] * (2 ** (num_stages - 1))) for i in range(len(patch_size)))
    except Exception:
        pad_to = None

    # Data loaders (pass pad_to to ensure compatibility)
    train_loader = create_data_loader(exp.data, split="train", batch_size=exp.training.batch_size, pad_to=pad_to)
    val_loader = create_data_loader(exp.data, split="test", batch_size=exp.training.batch_size, pad_to=pad_to)

    # Run short training (the Trainer will print progress)
    start = time.time()
    # Redirect stdout to per-trial log file for clean, per-trial logs and mirror to console
    import sys

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    log_file_path = trial_dir / "train.log"
    old_stdout = sys.stdout
    log_fh = open(log_file_path, "a")
    sys.stdout = Tee(old_stdout, log_fh)
    print(f"Trial {trial_idx} started - writing logs to {log_file_path}")
    try:
        history = trainer.fit(train_loader, val_loader, num_epochs=exp.training.epochs)
    finally:
        sys.stdout = old_stdout
        log_fh.close()
    elapsed = time.time() - start

    # Evaluate final
    val_metrics = trainer.evaluate(val_loader)

    # Save results (adapt to task)
    results = {
        "trial": trial_idx,
        "dataset": dataset_name,
        "stamp": stamp,
        "config_path": str(trial_dir / "config.json"),
        "train_epochs": exp.training.epochs,
        "elapsed_seconds": elapsed,
    }

    if getattr(exp.data, "task", "classification") == "segmentation":
        results["val_dice"] = float(val_metrics.get("val_dice", 0.0))
        results["val_voxel_accuracy"] = float(val_metrics.get("val_voxel_accuracy", 0.0))
        print(f"Trial {trial_idx} done - val_dice={results['val_dice']:.4f} (elapsed {elapsed:.1f}s)")
    else:
        results["val_accuracy"] = float(val_metrics.get("accuracy", 0.0))
        results["val_top5_accuracy"] = float(val_metrics.get("top5_accuracy", 0.0))
        print(f"Trial {trial_idx} done - val_top5={results['val_top5_accuracy']:.4f} (elapsed {elapsed:.1f}s)")

    (trial_dir / "results.json").write_text(json.dumps(results, indent=2))

    return results


def main() -> None:
    parser = argparse.ArgumentParser("Run a simple hyperparameter sweep (random search)")
    parser.add_argument("--sweep", type=str, default="configs/sweeps/cifar100_hyperparam_sweep.yaml")
    parser.add_argument("--base-config", type=str, default=None, help="Optional base experiment JSON file")
    parser.add_argument("--trials", type=int, default=None, help="Override number of trials in sweep file")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--outdir", type=str, default=None, help="Where to write sweep outputs (overrides sweep file)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    sweep = load_sweep(args.sweep)
    trials = args.trials or sweep.get("trials", 20)
    budget = sweep.get("budget_epochs", 25)
    outdir = Path(args.outdir or sweep.get("output_dir", "outputs/sweeps/unnamed"))
    outdir.mkdir(parents=True, exist_ok=True)

    # Seed
    seed = args.seed or sweep.get("seed", 42)
    random.seed(seed)

    base_cfg_path = args.base_config or sweep.get("base_config")
    space = sweep.get("param_space", {})

    summary = []
    print(f"Starting sweep: {trials} trials, budget {budget} epochs, outdir={outdir}")

    if args.dry_run:
        # Lightweight dry-run path: do not import heavy ML libs (JAX/Flax) - just materialize configs
        if base_cfg_path is not None:
            base_raw = json.loads(Path(base_cfg_path).read_text())
        else:
            # Minimal default template
            base_raw = {
                "model": {"num_dims": 2, "embed_dim": 96, "drop_path_rate": 0.1},
                "training": {"epochs": 100, "batch_size": 128, "learning_rate": 1e-3},
                "data": {"dataset": "cifar100", "data_dir": "data", "image_size": [32, 32]},
            }

        def set_by_path_dict(d: Dict[str, Any], path: str, value: Any) -> None:
            parts = path.split('.')
            cur = d
            for p in parts[:-1]:
                if p not in cur or not isinstance(cur[p], dict):
                    cur[p] = {}
                cur = cur[p]
            cur[parts[-1]] = value

        for i in range(trials):
            sampled = {k: sample_value(spec) for k, spec in space.items()}
            conf = deepcopy(base_raw)
            for k, v in sampled.items():
                set_by_path_dict(conf, k, v)
            # Short budget
            conf.setdefault('training', {})['epochs'] = int(budget)

            # Compute dataset/stamp for dry-run outputs
            dataset_name = conf.get('data', {}).get('hf_id') or conf.get('data', {}).get('dataset') or 'dataset'
            dataset_name = str(dataset_name).replace('/', '_').replace(':', '_')
            stamp = conf.get('name', 'dryrun') + '_dry'
            base_dir = outdir / dataset_name / stamp
            if base_dir.exists():
                shutil.rmtree(base_dir)
            base_dir.mkdir(parents=True, exist_ok=True)

            trial_dir = base_dir / f"trial_{i:03d}"
            trial_dir.mkdir(parents=True, exist_ok=False)
            (trial_dir / "config.json").write_text(json.dumps(conf, indent=2))

            res = {"trial": i, "status": "dry", "config_path": str(trial_dir / "config.json"), "dataset": dataset_name, "stamp": stamp}
            summary.append(res)
            # Write dataset-scoped summary
            (base_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            # Also write a top-level summary.json for backward compatibility
            (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

        print("Dry-run complete. Summary written to", str(base_dir / "summary.json"))
        return

    # Full run path (imports moved here to avoid import-time failures for dry-run/testing)
    base_exp = load_base_experiment(base_cfg_path)

    for i in range(trials):
        # Try to sample a valid configuration, with a few retries if model validation fails
        max_attempts = 10
        attempt = 0
        exp = None
        sampled = {}
        while attempt < max_attempts:
            sampled = {k: sample_value(spec) for k, spec in space.items()}

            # If embed dim sampled and it's a string, coerce
            if "model.embed_dim" in sampled and isinstance(sampled["model.embed_dim"], str):
                try:
                    sampled["model.embed_dim"] = int(float(sampled["model.embed_dim"]))
                except Exception:
                    pass

            try:
                exp = materialize_experiment(base_exp, sampled, budget)
                # Try constructing model config to trigger validation
                _ = exp.model  # materialize_experiment already validates but double-check
                break
            except Exception as e:
                # If error mentions embed_dim divisibility, try to adjust embed_dim to nearest valid multiple
                msg = str(e)
                if "embed_dim" in msg and "divisible" in msg and "num_heads" in msg:
                    # Attempt to coerce model.embed_dim to nearest multiple of num_heads[0]
                    try:
                        base_embed = sampled.get("model.embed_dim", base_exp.model.embed_dim)
                        num_heads0 = exp.model.num_heads[0] if exp is not None and hasattr(exp.model, "num_heads") else base_exp.model.num_heads[0]
                    except Exception:
                        num_heads0 = base_exp.model.num_heads[0]
                        base_embed = sampled.get("model.embed_dim", base_exp.model.embed_dim)

                    # Compute nearest lower multiple
                    nearest = int(base_embed - (base_embed % num_heads0))
                    if nearest <= 0:
                        nearest = num_heads0
                    print(f"Adjusting sampled model.embed_dim {base_embed} -> nearest valid {nearest} (multiple of {num_heads0})")
                    sampled["model.embed_dim"] = nearest
                    attempt += 1
                    continue
                # Otherwise, resample (some parameters might cause failures like patch/window mismatch)
                attempt += 1
                print(f"Sampled invalid config on attempt {attempt}/{max_attempts}: {e}; resampling...")
                continue

        if exp is None:
            err_msg = f"Skipping trial {i}: could not find valid config after {max_attempts} attempts"
            print(err_msg)
            summary.append({"trial": i, "status": "error", "error": err_msg})
            (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
            continue

        try:
            res = run_trial(i, exp, outdir, dry_run=False)
            summary.append(res)
            # Write dataset-scoped summary next to trials
            summary_path = outdir / res.get("dataset", "dataset") / res.get("stamp", "stamp") / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2))
        except Exception as e:
            # Record error for this trial and continue with the sweep
            import traceback

            tb = traceback.format_exc()
            print(f"Trial {i} failed with exception: {e}\n{tb}")
            # Ensure trial dir exists
            trial_dir = outdir / f"trial_{i:03d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            (trial_dir / "error.txt").write_text(tb)
            summary.append({"trial": i, "status": "error", "error": str(e), "traceback_file": str(trial_dir / "error.txt")})

        # Write incremental summary (fallback to top-level if dataset unknown)
        try:
            last = summary[-1]
            summary_path = outdir / last.get("dataset", "_global") / last.get("stamp", "_global") / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2))
            # Also write a top-level summary.json for convenience/backcompat
            (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
        except Exception:
            (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("Sweep complete. Summary written to", str(outdir / "summary.json"))


if __name__ == "__main__":
    main()
