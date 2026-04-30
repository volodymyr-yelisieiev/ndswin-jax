import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

yaml = pytest.importorskip("yaml")


def cli_env() -> dict[str, str]:
    """Return an env dict with PYTHONPATH set to include src/."""
    env = os.environ.copy()
    src_path = str(Path("src").resolve())
    env["PYTHONPATH"] = (
        src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]
    )
    return env


def test_run_sweep_dryrun(tmp_path):
    outdir = tmp_path / "sweep_out"
    cmd = [
        sys.executable,
        "-m",
        "ndswin.cli",
        "sweep",
        "--sweep",
        "configs/sweeps/cifar10_hyperparam_sweep.yaml",
        "--dry-run",
        "--trials",
        "2",
        "--outdir",
        str(outdir),
    ]
    # Run the script as a subprocess to avoid heavy JAX initialization
    subprocess.check_call(cmd, env=cli_env())

    # Check that the summary.json exists and contains two entries
    summary = json.loads((outdir / "summary.json").read_text())
    assert summary["metric"] == "val_accuracy"
    assert summary["trials"] == 2
    assert summary["mode"] == "dry-run"
    assert len(summary["results"]) == 2
    for entry in summary["results"]:
        assert entry["status"] == "dry" or "config_path" in entry

    # Clean up
    shutil.rmtree(outdir)


def test_run_sweep_full_run_passes_loaded_sweep_config(tmp_path, monkeypatch):
    sweep_path = tmp_path / "sweep.yaml"
    sweep = {
        "trials": 1,
        "budget_epochs": 3,
        "output_dir": str(tmp_path / "sweep_out"),
        "seed": 123,
        "param_space": {},
    }
    sweep_path.write_text(yaml.safe_dump(sweep))

    import argparse

    import ndswin.cli as cli

    captured: dict[str, object] = {}
    sentinel_exp = SimpleNamespace()

    monkeypatch.setattr(cli, "load_base_experiment", lambda path: SimpleNamespace())
    monkeypatch.setattr(cli, "validate_experiment_dataset_contract", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "sample_valid_experiment",
        lambda base_exp, space, budget, max_attempts=100: (sentinel_exp, 2, {"invalid": 1}),
    )

    def fake_run_trial(trial_idx, exp, out_dir, dry_run=False, sweep_config=None):
        captured["trial_idx"] = trial_idx
        captured["exp"] = exp
        captured["out_dir"] = out_dir
        captured["dry_run"] = dry_run
        captured["sweep_config"] = sweep_config
        return {
            "trial": trial_idx,
            "status": "success",
            "dataset": "dataset",
            "stamp": "stamp",
            "config_path": str(out_dir / "config.json"),
        }

    monkeypatch.setattr(cli, "run_trial", fake_run_trial)
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(clear_caches=lambda: None))
    args = argparse.Namespace(
        sweep=str(sweep_path),
        base_config=None,
        trials=None,
        dry_run=False,
        outdir=str(tmp_path / "custom_outdir"),
        seed=None,
        log_level="INFO",
    )

    cli.run_sweep_command(args)

    assert captured["trial_idx"] == 0
    assert captured["exp"] is sentinel_exp
    assert captured["dry_run"] is False
    assert captured["sweep_config"] == sweep

    summary = json.loads((Path(args.outdir) / "summary.json").read_text())
    assert summary["budget_epochs"] == 3
    assert summary["trials"] == 1
    assert summary["sampling_rejections"] == 1
    assert summary["rejection_reasons"] == {"invalid": 1}
    assert summary["results"][0]["trial"] == 0
    assert summary["results"][0]["sample_attempts"] == 2


def test_materialize_experiment_preserves_useful_warmup_budget():
    import ndswin.cli as cli

    base = cli.load_base_experiment("configs/cifar10.json")
    sampled = {"training.warmup_epochs": 10}

    exp = cli.materialize_experiment(base, sampled, budget_epochs=20)

    assert exp.training.epochs == 20
    assert exp.training.warmup_epochs == 10


def test_materialize_experiment_clamps_inherited_warmup_to_budget_minus_one():
    import ndswin.cli as cli

    base = cli.load_base_experiment("configs/modelnet40.json")

    exp = cli.materialize_experiment(base, sampled={}, budget_epochs=10)

    assert exp.training.epochs == 10
    assert exp.training.warmup_epochs == 9


def test_materialize_experiment_rejects_sampled_warmup_at_budget():
    import ndswin.cli as cli

    base = cli.load_base_experiment("configs/modelnet40.json")

    with pytest.raises(ValueError, match="Sampled warmup_epochs must be less than budget_epochs"):
        cli.materialize_experiment(base, {"training.warmup_epochs": 10}, budget_epochs=10)


def test_materialize_experiment_rejects_invalid_sampled_model_config():
    import ndswin.cli as cli

    base = cli.load_base_experiment("configs/cifar10.json")

    with pytest.raises(ValueError, match="Invalid sampled configuration"):
        cli.materialize_experiment(base, {"model.embed_dim": 128}, budget_epochs=30)


def test_sample_valid_experiment_resamples_and_tracks_rejections(monkeypatch):
    import ndswin.cli as cli

    base = cli.load_base_experiment("configs/modelnet40.json")
    sampled_values = iter([10, 9])
    monkeypatch.setattr(cli, "sample_value", lambda spec: next(sampled_values))

    exp, attempts, rejection_reasons = cli.sample_valid_experiment(
        base,
        {"training.warmup_epochs": {"kind": "choice", "values": [10, 9]}},
        10,
        max_attempts=3,
    )

    assert attempts == 2
    assert exp.training.warmup_epochs == 9
    assert rejection_reasons == {"Sampled warmup_epochs must be less than budget_epochs": 1}


def test_run_trial_uses_sweep_configured_timeout(tmp_path, monkeypatch):
    import ndswin.cli as cli

    exp = cli.load_base_experiment("configs/cifar10.json")
    exp.training.epochs = 3

    captured: dict[str, object] = {}

    def fake_subprocess_run(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_subprocess_run)

    result = cli.run_trial(
        0,
        exp,
        tmp_path,
        dry_run=False,
        sweep_config={"max_steps_per_epoch": 10, "trial_timeout_seconds": 4321},
    )

    assert captured["timeout"] == 4321
    assert result["status"] == "error"
    assert "Metrics file not found" in result["message"]
