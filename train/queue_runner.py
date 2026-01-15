#!/usr/bin/env python3
"""Queue runner for sequential training jobs.

Reads a YAML/JSON queue file listing jobs and executes them sequentially.
Supports auto-sweep, sweep, and train job types with robust error handling.

Usage:
    python train/queue_runner.py --queue configs/queues/queue_2d_3d.yaml
    python train/queue_runner.py --queue configs/queues/queue_2d_3d.yaml --dry-run
    make queue QUEUE=configs/queues/queue_2d_3d.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


def load_queue(path: str) -> list[dict[str, Any]]:
    """Load queue file (YAML or JSON)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Queue file not found: {path}")

    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML required for YAML queue files. Install: pip install pyyaml")
        data = yaml.safe_load(text)
    elif p.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported queue file format: {p.suffix}")

    jobs = data if isinstance(data, list) else data.get("jobs", [])
    if not jobs:
        raise ValueError(f"No jobs found in queue file: {path}")

    return jobs


def build_command(job: dict[str, Any], python_exe: str) -> list[str]:
    """Build a subprocess command from a job specification.

    Job schema:
        type: auto-sweep | sweep | train
        sweep: path to sweep YAML (for auto-sweep/sweep)
        config: path to config JSON (for train)
        trials: number of trials (optional, for sweep/auto-sweep)
        train_epochs: epochs for final training (for auto-sweep)
        outdir: output directory (optional)
        args: extra CLI arguments as a list of strings (optional)
    """
    job_type = job.get("type", "auto-sweep")
    extra_args = job.get("args", [])
    if isinstance(extra_args, str):
        extra_args = extra_args.split()

    if job_type == "auto-sweep":
        sweep_file = job.get("sweep")
        if not sweep_file:
            raise ValueError(f"auto-sweep job requires 'sweep' field: {job}")
        cmd = [python_exe, "train/auto_sweep_and_train.py", "--sweep", sweep_file]
        if "trials" in job:
            cmd.extend(["--trials", str(job["trials"])])
        if "train_epochs" in job:
            cmd.extend(["--train-epochs", str(job["train_epochs"])])
        if "outdir" in job:
            cmd.extend(["--outdir", str(job["outdir"])])
        cmd.extend(extra_args)
        return cmd

    elif job_type == "sweep":
        sweep_file = job.get("sweep")
        if not sweep_file:
            raise ValueError(f"sweep job requires 'sweep' field: {job}")
        cmd = [python_exe, "train/run_sweep.py", "--sweep", sweep_file]
        if "trials" in job:
            cmd.extend(["--trials", str(job["trials"])])
        if "outdir" in job:
            cmd.extend(["--outdir", str(job["outdir"])])
        cmd.extend(extra_args)
        return cmd

    elif job_type == "train":
        config_file = job.get("config")
        if not config_file:
            raise ValueError(f"train job requires 'config' field: {job}")
        cmd = [python_exe, "train/train.py", "--config", config_file]
        if "epochs" in job:
            cmd.extend(["--epochs", str(job["epochs"])])
        cmd.extend(extra_args)
        return cmd

    else:
        raise ValueError(f"Unknown job type: {job_type}")


def run_job(
    job_idx: int,
    job: dict[str, Any],
    python_exe: str,
    dry_run: bool = False,
    retry: int = 0,
) -> dict[str, Any]:
    """Execute a single job and return result metadata."""
    job_name = job.get("name", f"job_{job_idx:03d}")
    job_type = job.get("type", "auto-sweep")
    max_retries = job.get("retries", retry)

    print(f"\n{'=' * 80}")
    print(f"JOB {job_idx}: {job_name} (type={job_type})")
    print(f"{'=' * 80}")

    try:
        cmd = build_command(job, python_exe)
    except ValueError as e:
        print(f"  ERROR: {e}")
        return {"job": job_idx, "name": job_name, "status": "error", "error": str(e)}

    print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        print(f"  [DRY RUN] Skipping execution")
        return {"job": job_idx, "name": job_name, "status": "dry", "command": " ".join(cmd)}

    # Execute with retries
    attempt = 0
    last_error = None
    while attempt <= max_retries:
        if attempt > 0:
            print(f"  Retry {attempt}/{max_retries}...")
            time.sleep(5)  # Brief pause before retry

        start = time.time()
        try:
            # Set environment variables for HF cache
            env = os.environ.copy()
            env["HF_HOME"] = ".hf_cache"
            env["HF_DATASETS_CACHE"] = ".hf_cache"

            result = subprocess.run(
                cmd,
                env=env,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            elapsed = time.time() - start
            print(f"\n  Job {job_name} COMPLETED in {elapsed:.1f}s")
            return {
                "job": job_idx,
                "name": job_name,
                "status": "completed",
                "elapsed_seconds": elapsed,
            }
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start
            last_error = str(e)
            print(f"\n  Job {job_name} FAILED (attempt {attempt + 1}/{max_retries + 1}, {elapsed:.1f}s): {e}")
            attempt += 1
        except KeyboardInterrupt:
            print(f"\n  Job {job_name} INTERRUPTED by user")
            return {"job": job_idx, "name": job_name, "status": "interrupted"}

    return {
        "job": job_idx,
        "name": job_name,
        "status": "failed",
        "error": last_error,
        "attempts": attempt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a queue of training jobs sequentially",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--queue", type=str, required=True, help="Path to queue YAML/JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--retry", type=int, default=0, help="Default number of retries per job")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue to next job if current job fails",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop queue if any job fails",
    )

    args = parser.parse_args()
    continue_on_error = not args.stop_on_error

    jobs = load_queue(args.queue)
    python_exe = sys.executable

    print("=" * 80)
    print("QUEUE RUNNER")
    print("=" * 80)
    print(f"Queue file: {args.queue}")
    print(f"Total jobs: {len(jobs)}")
    print(f"Continue on error: {continue_on_error}")
    print(f"Started: {datetime.now().isoformat()}")
    print("-" * 80)

    for i, job in enumerate(jobs):
        job_name = job.get("name", f"job_{i:03d}")
        print(f"  [{i}] {job_name} (type={job.get('type', 'auto-sweep')})")

    results = []
    queue_start = time.time()

    for i, job in enumerate(jobs):
        result = run_job(i, job, python_exe, dry_run=args.dry_run, retry=args.retry)
        results.append(result)

        # Save incremental results
        results_path = Path("logs") / "queue_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, indent=2))

        if result["status"] in ("failed", "error") and not continue_on_error:
            print(f"\nStopping queue due to job failure (--stop-on-error)")
            break

    queue_elapsed = time.time() - queue_start

    # Final summary
    print(f"\n{'=' * 80}")
    print("QUEUE COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total time: {queue_elapsed:.1f}s ({queue_elapsed / 3600:.2f}h)")

    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] in ("failed", "error"))
    skipped = sum(1 for r in results if r["status"] == "dry")

    print(f"Completed: {completed}/{len(jobs)}")
    if failed > 0:
        print(f"Failed: {failed}/{len(jobs)}")
    if skipped > 0:
        print(f"Dry-run: {skipped}/{len(jobs)}")

    for r in results:
        status_icon = "✓" if r["status"] == "completed" else "✗" if r["status"] in ("failed", "error") else "○"
        elapsed_str = f" ({r.get('elapsed_seconds', 0):.0f}s)" if "elapsed_seconds" in r else ""
        print(f"  {status_icon} [{r['job']}] {r['name']}: {r['status']}{elapsed_str}")

    print(f"\nResults saved to: logs/queue_results.json")


if __name__ == "__main__":
    main()
