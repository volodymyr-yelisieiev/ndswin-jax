#!/usr/bin/env python3
"""Thin compatibility wrapper for `ndswin queue`."""

from __future__ import annotations

from ndswin.cli import build_command as build_command
from ndswin.cli import load_completed_jobs as load_completed_jobs
from ndswin.cli import load_queue as load_queue
from ndswin.cli import queue_main
from ndswin.cli import run_job as run_job
from ndswin.cli import run_queue_command as run_queue_command


def main() -> None:
    raise SystemExit(queue_main())


if __name__ == "__main__":
    main()
