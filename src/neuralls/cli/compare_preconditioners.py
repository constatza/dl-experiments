#!/usr/bin/env python3
"""CLI wrapper for preconditioner comparison (aggregated per solver config).

Experiments (optional) are only used to resolve neural checkpoints; the comparison itself
runs once for the provided solver config and writes shared diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path


import os

import typer
from loguru import logger

from neuralls.constants import (
    DEFAULT_EXPERIMENTS_CONFIG,
    DEFAULT_PROJECT_ROOT,
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
    SYMBOL_CHECKMARK,
)
from neuralls.workflows.comparison import run_batch_comparison
from neuralls.workflows.results import ComparisonResult
from neuralls.workflows.specs import (
    ComparisonParams,
    ComparisonOutcome,
)

os.environ.setdefault("MPLBACKEND", "Agg")


def _log_comparison_results(result: ComparisonResult) -> None:
    preconditioners = result.preconditioners
    summary = result.summary
    recommendations = result.recommendations

    logger.info(f"Available preconditioners: {preconditioners}")
    if summary:
        logger.info("=" * 60)
        logger.info(summary)
        logger.info("=" * 60)

    best = recommendations.get("best_overall") if isinstance(recommendations, dict) else None
    if best:
        logger.info(
            f"Best preconditioner: method={best.get('label')} "
            f"iterations={best.get('iterations')} residual={best.get('residual')}"
        )


def _log_outcomes(outcomes: list[ComparisonOutcome]) -> None:
    successful = [o for o in outcomes if o.success]
    failed = [o for o in outcomes if not o.success]
    for outcome in outcomes:
        logger.info("=" * 80)
        logger.info(f"Experiment: {outcome.name}")
        logger.info("=" * 80)
        if outcome.success and outcome.payload:
            _log_comparison_results(outcome.payload)
        else:
            logger.error(f"Comparison failed: {outcome.error}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total experiments: {len(outcomes)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    if failed:
        for item in failed:
            logger.error(f"  ✗ {item.name}: {item.error}")
        raise typer.Exit(code=EXIT_FAILURE)
    logger.info(f"\n{SYMBOL_CHECKMARK} All comparisons completed successfully!")


def main(
    experiments: Path = typer.Option(
        None,
        "--experiments",
        "-e",
        help="Path to experiments config (defaults to configs/experiments.toml)",
    ),
    solver_config: Path = typer.Option(
        ...,
        "--solver-config",
        "-s",
        help="Path to solver config (e.g., configs/solvers/default.toml)",
    ),
    plots: bool = typer.Option(
        True,
        "--plots/--no-plots",
        help="Save comparison plots to disk (default: True)",
    ),
):
    """Compare preconditioner methods using configuration files."""
    experiments_path = (
        experiments if experiments is not None else DEFAULT_PROJECT_ROOT / DEFAULT_EXPERIMENTS_CONFIG
    )
    params = ComparisonParams(save_plots=plots)

    try:
        outcomes = run_batch_comparison(experiments_path, solver_config, params)
    except ValueError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=EXIT_FAILURE) from exc

    _log_outcomes(outcomes)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
