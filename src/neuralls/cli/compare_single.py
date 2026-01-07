#!/usr/bin/env python3
"""CLI wrapper for single experiment comparison."""

from __future__ import annotations

from pathlib import Path


import os
import typer
from loguru import logger

from neuralls.constants import (
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
    SYMBOL_CHECKMARK,
)
from neuralls.workflows.comparison import build_direct_comparisons, run_comparisons
from neuralls.workflows.specs import ComparisonParams

os.environ.setdefault("MPLBACKEND", "Agg")


def main(
    model_config: Path = typer.Option(..., help="Model config path"),
    data_config: Path = typer.Option(..., help="Data config path"),
    solver_config: Path = typer.Option(..., help="Solver config path (solvers.toml)"),
    plots: bool = typer.Option(
        True,
        "--plots/--no-plots",
        help="Save comparison plots to disk (default: True)",
    ),
) -> None:
    """Compare preconditioner methods using configuration files."""
    params = ComparisonParams(save_plots=plots)

    specs = build_direct_comparisons(
        model_config=model_config.resolve(),
        data_config=data_config.resolve(),
        solver_config=solver_config.resolve(),
    )
    if not specs:
        logger.error("No comparison spec could be built. Check checkpoint and paths.")
        raise typer.Exit(code=EXIT_FAILURE)

    logger.info(f"Running single comparison for: {model_config}")
    outcomes = run_comparisons(specs, params)
    failed = [o for o in outcomes if not o.success]
    if failed:
        for item in failed:
            logger.error(f"Comparison failed for {item.name}: {item.error}")
        raise typer.Exit(code=EXIT_FAILURE)
    logger.info(f"\n{SYMBOL_CHECKMARK} Comparison completed successfully!")


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
