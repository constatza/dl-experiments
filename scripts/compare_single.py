#!/usr/bin/env python3
"""CLI wrapper for single experiment comparison."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import typer
from loguru import logger

from src.constants import (
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
    SYMBOL_CHECKMARK,
    REORTHOG_STRICT_THRESHOLD,
)
from src.workflows.comparison import build_direct_comparisons, run_comparisons
from src.workflows.specs import ComparisonParams, PreconditionerLimits

os.environ.setdefault("MPLBACKEND", "Agg")


def main(
    model_config: Path = typer.Option(..., help="Model config path"),
    data_config: Path = typer.Option(..., help="Data config path"),
    solver_config: Path = typer.Option(..., help="Solver config path (solvers.toml)"),
    checkpoint_path: Path = typer.Option(..., help="Checkpoint path"),
    matrix: Path | None = typer.Option(None, help="Override matrix path (npz supported)."),
    rhs: Path | None = typer.Option(None, help="Override RHS path (npz supported)."),
    output_dir: Path | None = typer.Option(
        None, help="Override output directory for this experiment"
    ),
    figures_dir: Path | None = typer.Option(
        None, help="Override figures directory for plots"
    ),
    plots: bool = typer.Option(
        True,
        "--plots/--no-plots",
        help="Save comparison plots to disk (default: True)",
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
    enable_mlflow: bool = typer.Option(
        False,
        "--enable-mlflow/--no-mlflow",
        help="Enable MLflow logging for the comparison (defaults to disabled)",
    ),
):
    """Compare preconditioner methods for a single experiment (explicit configs)."""
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

    specs = build_direct_comparisons(
        model_config=model_config.resolve(),
        data_config=data_config.resolve(),
        solver_config=solver_config.resolve(),
        output_root=output_dir,
        checkpoint_path=checkpoint_path.resolve(),
        checkpoint_config=None,
        matrix=matrix,
        rhs=rhs,
    )
    if not specs:
        logger.error("No comparison spec could be built. Check checkpoint and paths.")
        raise typer.Exit(code=EXIT_FAILURE)

    logger.info(f"Running single comparison for: {model_config}")
    outcomes = run_comparisons(specs, params, enable_mlflow=enable_mlflow)
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
