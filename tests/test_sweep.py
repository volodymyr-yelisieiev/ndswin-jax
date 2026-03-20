import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def cli_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(Path("src").resolve())
    env["PYTHONPATH"] = (
        src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]
    )
    return env


def test_run_sweep_dryrun(tmp_path):
    try:
        __import__("yaml")
    except Exception:
        # Skip if PyYAML not available in the environment
        print("PyYAML not installed; skipping sweep dry-run test")
        return

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
