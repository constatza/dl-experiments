#!/usr/bin/env python3
"""CLI wrapper for preconditioner comparison - batch mode only.

This script runs preconditioner comparisons for ALL experiments defined in experiments.toml.
It uses the unified configuration architecture where experiments.toml is the single source of truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from typing import Any

import typer
from loguru import logger

from neuralls.constants import (
    DEFAULT_EXPERIMENTS_CONFIG,
    DEFAULT_PROJECT_ROOT,
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
    SYMBOL_CHECKMARK,
)
from neuralls.workflows.comparison import (
    build_batch_comparisons,
    run_comparisons,
)
from neuralls.workflows.specs import (
    ComparisonParams,
    ComparisonOutcome,
)

os.environ.setdefault("MPLBACKEND", "Agg")


def _log_comparison_results(result: dict[str, Any]) -> None:
    logger.info(f"Available preconditioners: {result.get('preconditioners')}")
    metadata = result.get("preconditioner_metadata", {})
    neural = metadata.get("neural") if isinstance(metadata, dict) else None
    if neural is not None:
        residual_iters = getattr(neural, "residual_iters", "unknown")
        applied_iters = getattr(neural, "applied_iters", None)
        applied = applied_iters if applied_iters is not None else residual_iters
        logger.info(f"Neural preconditioner: residual_iters={residual_iters} applied={applied}")
    summary = result.get("summary")
    if summary:
        logger.info("=" * 60)
        logger.info(summary)
        logger.info("=" * 60)
    recs = result.get("recommendations", {}) if isinstance(result, dict) else {}
    best = recs.get("best_overall") if isinstance(recs, dict) else None
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

    specs = build_batch_comparisons(experiments_path)

    if not specs:
        logger.error("No comparison specs were built. Check checkpoints and inputs.")
        raise typer.Exit(code=EXIT_FAILURE)

    params = ComparisonParams(save_plots=plots)

    logger.info(f"Running comparisons for {len(specs)} experiments...")
    logger.info(f"Experiments config: {experiments_path}")
    logger.info("")

    outcomes = run_comparisons(specs, params)
    _log_outcomes(outcomes)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
