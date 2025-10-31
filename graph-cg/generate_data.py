#!/usr/bin/env python3
"""Generate training data (RHS -> solution) for a linear system using src/ library modules.

Usage:
    uv run python generate_data.py data-configs/generate-90.toml
"""

from __future__ import annotations
from pathlib import Path

import typer

from src.constants import (
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
    SYMBOL_SUCCESS,
    SYMBOL_ERROR,
)
from src.cli.data import generate_data_from_config


def main(
    config: Path = typer.Argument(
        ...,
        help="Path to data-config file (e.g., data-configs/generate-90.toml)",
    ),
):
    """Generate training data using data-config format.

    Usage:
        uv run python generate_data.py data-configs/generate-90.toml
    """
    try:
        print(f"Loading data config: {config}")
        output_path = generate_data_from_config(config)

        print(f"\n{SYMBOL_SUCCESS} Data generation complete!")
        print(f"  Output: {output_path}")
        print("  Files:")
        for file in sorted(output_path.glob("*")):
            print(f"    - {file.name}")

    except Exception as e:
        print(f"\n{SYMBOL_ERROR} Error: {e}")
        raise typer.Exit(code=EXIT_FAILURE)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
