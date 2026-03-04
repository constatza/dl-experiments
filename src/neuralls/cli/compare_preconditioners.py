#!/usr/bin/env python3
"""CLI wrapper for preconditioner comparison.

Supports two modes:
- Pipeline mode: with ``--comparison-run`` from ``train-multiple``.
- Standalone mode: without ``--comparison-run``; comparison config must include
  ``[general.tracking]``.

Usage::

    # Standalone mode (no ComparisonRun needed)
    uv run compare-preconditioners --comparison-config configs/comparison/linear.toml

    # Pipeline mode (with ComparisonRun from train-multiple)
    uv run compare-preconditioners \\
        --comparison-config configs/comparison/linear.toml \\
        --comparison-run /path/to/comparison_run.json
"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from loguru import logger

from neuralls.constants import (
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
    SYMBOL_CHECKMARK,
)
from neuralls.workflows.comparison import run_comparison
from neuralls.workflows.comparison_run import load_comparison_run
from neuralls.workflows.results import ComparisonResult
from neuralls.workflows.specs import (
    ComparisonOutcome,
    ComparisonParams,
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
        logger.info(f"Solver config: {outcome.name}")
        logger.info("=" * 80)
        if outcome.success and outcome.payload:
            _log_comparison_results(outcome.payload)
        else:
            logger.error(f"Comparison failed: {outcome.error}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total comparisons: {len(outcomes)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    if failed:
        for item in failed:
            logger.error(f"  ✗ {item.name}: {item.error}")
        raise typer.Exit(code=EXIT_FAILURE)
    logger.info(f"\n{SYMBOL_CHECKMARK} All comparisons completed successfully!")


def main(
    comparison_config: Path = typer.Option(
        ...,
        "--comparison-config",
        "-s",
        help="Path to comparison config (e.g., configs/comparison/linear.toml)",
    ),
    comparison_run: Path | None = typer.Option(
        default=None,
        help=(
            "Path to comparison_run.json produced by train-multiple "
            "(optional metadata-only context). "
            "Comparison tracking is always taken from [general.tracking]."
        ),
        exists=False,
    ),
    plots: bool = typer.Option(
        True,
        "--plots/--no-plots",
        help="Save comparison plots to disk (default: True)",
    ),
) -> None:
    """Compare preconditioner methods using comparison config.

    Supports pipeline mode (with --comparison-run) and standalone mode (without).
    """
    cr = None
    if comparison_run is not None:
        try:
            cr = load_comparison_run(comparison_run)
        except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
            logger.warning(
                "Could not load comparison run from '{}': {}. Continuing without it.",
                comparison_run,
                exc,
            )
            cr = None
        if cr is not None:
            logger.info(f"Loaded comparison_run_id = {cr.mlflow_run_id}")
    else:
        logger.info("Standalone mode: no comparison_run provided")

    params = ComparisonParams(save_plots=plots)

    try:
        outcomes = run_comparison(comparison_config, params, cr)
    except (ValueError, KeyError) as exc:
        logger.error(str(exc))
        raise typer.Exit(code=EXIT_FAILURE) from exc

    _log_outcomes(outcomes)


def run() -> None:
    """Entry point for pyproject.toml script registration."""
    typer.run(main)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
