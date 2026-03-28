#!/usr/bin/env python3
"""Build one processed dataset from a dataset config."""

from __future__ import annotations

from pathlib import Path


import typer

from neuralls.shared.constants import (
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
    SYMBOL_SUCCESS,
    SYMBOL_ERROR,
)
from neuralls.composition.generation.process_data import process_data_from_config


def main(
    config: Path = typer.Argument(
        ...,
        help="Path to a dataset config TOML.",
    ),
) -> None:
    """Generate or collect one dataset.

    Start here when you want to inspect a single dataset config before running
    registry-wide training or comparison commands.
    """
    try:
        print(f"Loading data config: {config}")
        output_path = process_data_from_config(config)

        print(f"\n{SYMBOL_SUCCESS} Data processing complete!")
        print(f"  Output: {output_path}")
        print("  Files:")
        for file in sorted(output_path.glob("*")):
            print(f"    - {file.name}")

    except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
        print(f"\n{SYMBOL_ERROR} Error: {e}")
        raise typer.Exit(code=EXIT_FAILURE)


def run() -> None:
    """Entry point for pyproject.toml script registration."""
    typer.run(main)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
