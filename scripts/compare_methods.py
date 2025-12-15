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
from dataclasses import dataclass
from typing import Any

import typer
from loguru import logger

from src.constants import (
    DEFAULT_EXPERIMENTS_CONFIG,
    DEFAULT_PROJECT_ROOT,
    NOISE_STRATEGY_NONE,
    DEFAULT_NOISE_LEVEL,
    DEFAULT_NOISE_SEED,
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
    SYMBOL_CHECKMARK,
    REORTHOG_STRICT_THRESHOLD,
)
from src.cli.comparison import compare_preconditioners
from src.configuration.loader import load_experiments
from src.system_loading import get_latest_checkpoint
from src.workflows.utils.paths import resolve_output_root

os.environ.setdefault("MPLBACKEND", "Agg")


@dataclass(frozen=True)
class ComparisonParams:
    """Parameters for preconditioner comparison."""

    matrix: Path | None
    rhs: Path | None
    output_dir: Path | None
    figures_dir: Path | None
    save_plots: bool
    noise_strategy: str
    noise_level: float
    noise_seed: int | None
    breakdown_tol: float | None
    neural_precond_iters: int | None
    fallback_preconditioner: str
    precond_every: int
    precond_first_n: int | None
    reorthogonalize: str
    reorthog_window: int
    reorthog_threshold: float


@dataclass(frozen=True)
class ExperimentResult:
    """Result of running one experiment comparison."""

    name: str
    success: bool
    error_message: str | None = None
    best_preconditioner: dict[str, Any] | None = None
    plot_paths: dict[str, Path] | None = None


def validate_checkpoint_exists(checkpoint_path: Path) -> str | None:
    """Validate that checkpoint file exists."""
    if not checkpoint_path.exists():
        return f"Checkpoint not found: {checkpoint_path}"
    return None


def run_single_comparison(
    exp_name: str,
    model_config: Path,
    data_config: Path,
    solver_config: Path,
    checkpoint_path: Path,
    params: ComparisonParams,
) -> dict[str, Any]:
    """Run comparison for a single experiment."""
    return compare_preconditioners(
        config_path=model_config,
        data_config_path=data_config,
        solver_config_path=solver_config,
        checkpoint_path=checkpoint_path,
        matrix_path=params.matrix,
        rhs_path=params.rhs,
        output_dir=params.output_dir,
        noise_strategy=params.noise_strategy,
        noise_level=params.noise_level,
        noise_seed=params.noise_seed,
        save_plots=params.save_plots,
        figures_dir=params.figures_dir,
        breakdown_tol=params.breakdown_tol,
        neural_precond_iters=params.neural_precond_iters,
        fallback_preconditioner=params.fallback_preconditioner,
        precond_every=params.precond_every,
        precond_first_n=params.precond_first_n,
        reorthogonalize=params.reorthogonalize,
        reorthog_window=params.reorthog_window,
        reorthog_threshold=params.reorthog_threshold,
        custom_combinations=None,
    )


def log_comparison_results(result: dict[str, Any]) -> None:
    """Log comparison results."""
    logger.info(f"Available preconditioners: {result['preconditioners']}")

    # Log neural preconditioner metadata
    precond_metadata = result.get("preconditioner_metadata", {})
    neural_meta = precond_metadata.get("neural")

    if neural_meta is not None:
        residual_iters = getattr(neural_meta, "residual_iters", "unknown")
        applied_iters = getattr(neural_meta, "applied_iters", None)
        applied_str = (
            f"{applied_iters}"
            if applied_iters is not None
            else f"{residual_iters} (default)"
        )
        logger.info("Neural preconditioner metadata:")
        logger.info(f"  - Training iterations (residual_iters): {residual_iters}")
        logger.info(f"  - Applied iterations: {applied_str}")

    logger.info("=" * 60)
    logger.info(result["summary"])
    logger.info("=" * 60)

    # Log any errors in results
    error_results = {
        name: res
        for name, res in result["results"].items()
        if hasattr(res, "error") and res.error is not None
    }

    if error_results:
        logger.warning("Errors detected in the following methods:")
        for name, res in error_results.items():
            logger.error(f"{name}: {res.error}")

    # Log best preconditioner
    recs = result.get("recommendations", {})
    best_overall = recs.get("best_overall") if isinstance(recs, dict) else None

    if best_overall:
        logger.info("Best preconditioner:")
        logger.info(
            f"  method={best_overall['label']} | iterations={best_overall['iterations']} "
            f"| residual={best_overall['residual']:.3e}"
        )


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
    matrix: Path | None = typer.Option(
        None,
        help="Override matrix path (npz supported).",
    ),
    rhs: Path | None = typer.Option(
        None,
        help="Override RHS path (npz supported).",
    ),
    output_dir: Path | None = typer.Option(
        None, help="Override output directory for all experiments"
    ),
    figures_dir: Path | None = typer.Option(
        None, help="Override figures directory for plots"
    ),
    noise_strategy: str = typer.Option(NOISE_STRATEGY_NONE, help="Noise strategy"),
    noise_level: float = typer.Option(
        DEFAULT_NOISE_LEVEL, help="Noise level parameter"
    ),
    noise_seed: int | None = typer.Option(
        DEFAULT_NOISE_SEED, help="Random seed for noise"
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
):
    """Compare preconditioner methods for all experiments in experiments.toml."""
    root_dir = DEFAULT_PROJECT_ROOT

    # Resolve experiments config
    experiments_path = (
        experiments
        if experiments is not None
        else root_dir / DEFAULT_EXPERIMENTS_CONFIG
    )
    experiments_path = Path(experiments_path)

    try:
        # Load experiments using the unified loader
        # This returns tuples: (name, settings, context, model_path, data_path, solver_path)
        experiments_list = load_experiments(experiments_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        raise typer.Exit(code=EXIT_FAILURE)

    output_root = resolve_output_root(None) if output_dir is None else output_dir

    logger.info(f"Running comparisons for {len(experiments_list)} experiments...")
    logger.info(f"Experiments config: {experiments_path}")
    logger.info(f"Output root: {output_root}")
    logger.info("")

    # Build comparison parameters
    params = ComparisonParams(
        matrix=matrix,
        rhs=rhs,
        output_dir=output_dir,
        figures_dir=figures_dir,
        save_plots=plots,
        noise_strategy=noise_strategy,
        noise_level=noise_level,
        noise_seed=noise_seed,
        breakdown_tol=breakdown_tol,
        neural_precond_iters=neural_precond_iters,
        fallback_preconditioner=fallback_preconditioner,
        precond_every=precond_every,
        precond_first_n=precond_first_n,
        reorthogonalize=reorthogonalize,
        reorthog_window=reorthog_window,
        reorthog_threshold=reorthog_threshold,
    )

    results = []
    
    for idx, (exp_name, settings, context, model_path, data_path, solver_path) in enumerate(experiments_list, 1):
        logger.info("=" * 80)
        logger.info(f"Experiment {idx}/{len(experiments_list)}: {exp_name}")
        logger.info("=" * 80)
        
        # Determine checkpoint path
        # 1. Check if checkpoint was overridden in context (not currently supported by loader but good for future)
        # 2. Use get_latest_checkpoint from the training output directory
        checkpoint_dir = context.training.checkpoint_dir
        checkpoint_path = get_latest_checkpoint(checkpoint_dir)
        
        if not checkpoint_path or not checkpoint_path.exists():
            logger.warning(f"⚠ Checkpoint not found in {checkpoint_dir}")
            results.append(ExperimentResult(name=exp_name, success=False, error_message="Checkpoint not found"))
            continue
            
        logger.info(f"Checkpoint: {checkpoint_path}")

        try:
            result = run_single_comparison(
                exp_name, model_path, data_path, solver_path, checkpoint_path, params
            )
            log_comparison_results(result)
            
            recs = result.get("recommendations", {})
            best_overall = recs.get("best_overall") if isinstance(recs, dict) else None
            
            results.append(ExperimentResult(
                name=exp_name,
                success=True,
                best_preconditioner=best_overall,
                plot_paths=result.get("plot_paths")
            ))
            
        except Exception as exc:
            error_msg = f"Comparison failed: {exc}"
            logger.error(error_msg)
            results.append(ExperimentResult(name=exp_name, success=False, error_message=error_msg))

    # Log final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    logger.info(f"Total experiments: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")

    if failed > 0:
        logger.info("\nFailed experiments:")
        for r in results:
            if not r.success:
                logger.error(f"  ✗ {r.name}: {r.error_message}")
        raise typer.Exit(code=EXIT_FAILURE)
    else:
        logger.info(f"\n{SYMBOL_CHECKMARK} All comparisons completed successfully!")


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)