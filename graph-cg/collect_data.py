#!/usr/bin/env python3
"""Collect and process data from existing sources (e.g., SpectralData).

Usage:
    uv run python collect_data.py data-configs/collect-504.toml
"""

from __future__ import annotations

from pathlib import Path

import typer

from src.constants import (
    DEFAULT_CG_TOLERANCE,
    DEFAULT_CG_MAX_ITERATIONS,
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
    SYMBOL_SUCCESS,
    SYMBOL_ERROR,
)
from src.cli.data import collect_data_from_config


def main(
    config: Path = typer.Argument(
        ..., help="Path to data collection config file (TOML)"
    ),
    solve: bool = typer.Option(
        True, help="Solve linear systems via CG (if False, only collects RHS)"
    ),
    cg_tolerance: float = typer.Option(
        DEFAULT_CG_TOLERANCE, help="CG solver tolerance"
    ),
    cg_max_iters: int = typer.Option(
        DEFAULT_CG_MAX_ITERATIONS, help="CG solver max iterations"
    ),
):
    """Collect data from existing sources and solve linear systems."""
    try:
        output_path = collect_data_from_config(
            config,
            solve_systems=solve,
            cg_tolerance=cg_tolerance,
            cg_max_iters=cg_max_iters,
        )

        print(f"\n{SYMBOL_SUCCESS} Data collection complete!")
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
