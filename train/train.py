#!/usr/bin/env python3
"""Thin compatibility wrapper for `ndswin train`."""

from __future__ import annotations

from ndswin.cli import run_train_command as run_train_command
from ndswin.cli import setup_output_dirs as setup_output_dirs
from ndswin.cli import train_main


def main() -> None:
    raise SystemExit(train_main())


if __name__ == "__main__":
    main()
