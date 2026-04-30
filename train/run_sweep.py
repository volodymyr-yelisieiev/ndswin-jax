#!/usr/bin/env python3
"""Thin compatibility wrapper for `ndswin sweep`."""

from __future__ import annotations

from ndswin.cli import get_best_trial_from_summary as get_best_trial_from_summary
from ndswin.cli import load_base_experiment as load_base_experiment
from ndswin.cli import load_sweep as load_sweep
from ndswin.cli import materialize_experiment as materialize_experiment
from ndswin.cli import run_sweep_command as run_sweep_command
from ndswin.cli import run_trial as run_trial
from ndswin.cli import sample_value as sample_value
from ndswin.cli import sweep_main


def main() -> None:
    raise SystemExit(sweep_main())


if __name__ == "__main__":
    main()
