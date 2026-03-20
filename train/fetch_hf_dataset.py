#!/usr/bin/env python3
"""Thin compatibility wrapper for `ndswin fetch-data`."""

from __future__ import annotations

from ndswin.cli import export_dataset as export_dataset
from ndswin.cli import fetch_data_main
from ndswin.cli import run_fetch_data_command as run_fetch_data_command


def main() -> None:
    raise SystemExit(fetch_data_main())


if __name__ == "__main__":
    main()
