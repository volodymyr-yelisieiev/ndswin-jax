#!/usr/bin/env python3
"""Thin compatibility wrapper for `ndswin auto-sweep`."""

from __future__ import annotations

from ndswin.cli import auto_sweep_main
from ndswin.cli import get_best_trial_from_summary as get_best_trial_from_summary
from ndswin.cli import run_auto_sweep_command as run_auto_sweep_command


def main() -> None:
    raise SystemExit(auto_sweep_main())


if __name__ == "__main__":
    main()
