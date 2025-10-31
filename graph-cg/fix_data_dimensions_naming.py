#!/usr/bin/env python3
"""Standardize raw data filenames to the {kind}-{dimensions} convention."""

from __future__ import annotations

from pathlib import Path

from src.cli.raw_data import standardize_raw_filenames


def main(data_root: Path = Path("/data/projects/graph-cg/data/raw/")) -> None:
    standardize_raw_filenames(data_root)


if __name__ == "__main__":
    main()
