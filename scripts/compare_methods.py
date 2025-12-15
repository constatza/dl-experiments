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

from src.constants import (
    DEFAULT_EXPERIMENTS_CONFIG,
    DEFAULT_PROJECT_ROOT,
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
    SYMBOL_CHECKMARK,
    REORTHOG_STRICT_THRESHOLD,
)
from src.workflows.comparison import (
    build_batch_comparisons,
    build_direct_comparisons,
    run_comparisons,
)
from src.workflows.specs import (
    ComparisonParams,
    PreconditionerLimits,
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
    matrix: Path | None = typer.Option(None, help="Override matrix path (npz supported)."),
    rhs: Path | None = typer.Option(None, help="Override RHS path (npz supported)."),
    output_dir: Path | None = typer.Option(
        None, help="Override output directory for all experiments"
    ),
    figures_dir: Path | None = typer.Option(
        None, help="Override figures directory for plots"
    ),
    breakdown_tol: float | None = typer.Option(
        None, help="Breakdown tolerance for CG denominator checks"
    ),
    neural_precond_iters: int | None = typer.Option(
        None, help="Limit neural preconditioning to first L iterations"
    ),
    fallback_preconditioner: str = typer.Option(
        "identity",
        help="Preconditioner after L neural iterations (identity, jacobi, ilu)",
    ),
    precond_every: int = typer.Option(
        1, help="Apply preconditioner every K iterations"
    ),
    precond_first_n: int | None = typer.Option(
        None, help="Only apply preconditioner for first N iterations"
    ),
    reorthogonalize: str = typer.Option(
        "full", help="Reorthogonalization strategy (none, full, partial, selective)"
    ),
    reorthog_window: int = typer.Option(
        10, help="Window size for partial reorthogonalization"
    ),
    reorthog_threshold: float = typer.Option(
        REORTHOG_STRICT_THRESHOLD, help="Threshold for selective reorthogonalization"
    ),
    checkpoint_config: Path | None = typer.Option(
        None, "--checkpoint-config", help="TOML containing checkpoint.path override"
    ),
    checkpoint_path: Path | None = typer.Option(None, help="Direct checkpoint path override"),
):
    """Compare preconditioner methods for all experiments in experiments.toml."""
    experiments_path = (
        experiments if experiments is not None else DEFAULT_PROJECT_ROOT / DEFAULT_EXPERIMENTS_CONFIG
    )

    specs = build_batch_comparisons(
        experiments_path,
        checkpoint_path=checkpoint_path.resolve() if checkpoint_path else None,
        checkpoint_config=checkpoint_config,
        matrix=matrix,
        rhs=rhs,
    )

    if not specs:
        logger.error("No comparison specs were built. Check checkpoints and inputs.")
        raise typer.Exit(code=EXIT_FAILURE)

    limits = PreconditionerLimits(
        apply_every=precond_every,
        first_n=precond_first_n,
        neural_iters=neural_precond_iters,
        fallback_preconditioner=fallback_preconditioner,
    )
    params = ComparisonParams(
        save_plots=plots,
        breakdown_tol=breakdown_tol,
        limits=limits,
        reorthogonalize=reorthogonalize,
        reorthog_window=reorthog_window,
        reorthog_threshold=reorthog_threshold,
        figures_dir=figures_dir,
        output_dir=output_dir,
        matrix=matrix,
        rhs=rhs,
    )

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
