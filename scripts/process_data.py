#!/usr/bin/env python3
r"""Process data for graph-cg: collection from archives or synthetic generation.

This unified script handles both data collection (from existing RHS archives)
and synthetic data generation through the same pipeline.

Usage:
    # Generate synthetic data
    uv run python graph-cg/scripts/process_data.py data-configs/generate-90.toml

    # Collect data from archives (with solving)
    uv run python graph-cg/scripts/process_data.py data-configs/collect-504.toml

    # Collect without solving (RHS only)
    uv run python graph-cg/scripts/process_data.py data-configs/collect-504.toml --no-solve
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add graph-cg root to Python path so we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer

from src.constants import (
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
    SYMBOL_SUCCESS,
    SYMBOL_ERROR,
)
from src.workflows.data import process_data_from_config


def main(
    config: Path = typer.Argument(
        ...,
        help="Path to data-config file (TOML)",
    ),
    solve: bool = typer.Option(
        True,
        "--solve/--no-solve",
        help="Solve linear systems via CG (for collection mode)",
    ),
):
    """Process data via unified pipeline using configuration files.

    This script automatically detects the data processing mode based on your
    config file's strategy definitions. It handles:

    - Synthetic generation (random, krylov, cg_residual strategies)
    - Archive collection (rhs_archive, solution_archive strategies)
    - Mixed strategies (combination of synthetic and archive)

    CG solver parameters are defined in the data config file.
    """
    try:
        print(f"Loading data config: {config}")
        output_path = process_data_from_config(
            config,
            solve_systems=solve,
        )

        print(f"\n{SYMBOL_SUCCESS} Data processing complete!")
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
