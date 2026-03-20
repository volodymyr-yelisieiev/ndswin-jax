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
    assert len(summary) == 2
    for entry in summary:
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

    import train.run_sweep as run_sweep

    captured: dict[str, object] = {}
    sentinel_exp = SimpleNamespace()

    monkeypatch.setattr(run_sweep, "load_base_experiment", lambda path: SimpleNamespace(model=SimpleNamespace(embed_dim=96, num_heads=(3, 6, 12, 24))))
    monkeypatch.setattr(run_sweep, "materialize_experiment", lambda base_exp, sampled, budget: sentinel_exp)

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

    monkeypatch.setattr(run_sweep, "run_trial", fake_run_trial)
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(clear_caches=lambda: None))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_sweep.py", "--sweep", str(sweep_path), "--outdir", str(tmp_path / "custom_outdir")],
    )

    run_sweep.main()

    assert captured["trial_idx"] == 0
    assert captured["exp"] is sentinel_exp
    assert captured["dry_run"] is False
    assert captured["sweep_config"] == sweep
