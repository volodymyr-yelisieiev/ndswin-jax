import json
import shutil
import sys
from pathlib import Path
import subprocess


def test_run_sweep_dryrun(tmp_path):
    try:
        import yaml  # type: ignore
    except Exception:
        # Skip if PyYAML not available in the environment
        print("PyYAML not installed; skipping sweep dry-run test")
        return

    outdir = tmp_path / "sweep_out"
    cmd = [sys.executable, "train/run_sweep.py", "--sweep", "configs/sweeps/cifar100_hyperparam_sweep.yaml", "--dry-run", "--trials", "2", "--outdir", str(outdir)]
    # Run the script as a subprocess to avoid heavy JAX initialization
    subprocess.check_call(cmd)

    # Check that the summary.json exists and contains two entries
    summary = json.loads((outdir / "summary.json").read_text())
    assert len(summary) == 2
    for entry in summary:
        assert entry["status"] == "dry" or "config_path" in entry

    # Clean up
    shutil.rmtree(outdir)
