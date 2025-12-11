#!/usr/bin/env python3
"""CLI wrapper for preconditioner comparison - batch mode only.

This script runs preconditioner comparisons for ALL experiments defined in experiments.toml.
It uses the unified configuration architecture where experiments.toml is the single source of truth.

Usage:
    # Run all experiments with default config
    uv run python scripts/compare_methods.py

    # Use custom experiments file
    uv run python scripts/compare_methods.py --experiments custom-experiments.toml

    # Override paths for testing
    uv run python scripts/compare_methods.py --matrix /path/to/test.npz

Checkpoint Path Resolution:
    Checkpoints are auto-derived from model and data configs by default.
    To compare different checkpoint versions, add checkpoint_path to experiments.toml:

    [[experiments]]
    model_template = "configs/ffnn.toml"
    data_config = "data-configs/test-solutions.toml"
    checkpoint_path = "/custom/path/to/checkpoint.ckpt"  # Optional override
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import tomllib
from dataclasses import dataclass
from typing import Any

import typer
from loguru import logger

from src.constants import (
    DEFAULT_EXPERIMENTS_CONFIG,
    NOISE_STRATEGY_NONE,
    DEFAULT_NOISE_LEVEL,
    DEFAULT_NOISE_SEED,
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
    SYMBOL_CHECKMARK,
    REORTHOG_STRICT_THRESHOLD,
)
from src.cli.comparison import compare_preconditioners
from src.system_loading import derive_checkpoint_path

os.environ.setdefault("MPLBACKEND", "Agg")


@dataclass(frozen=True)
class ExperimentConfig:
    """Single experiment configuration."""

    model_template: Path
    data_config: Path
    solver_config: Path
    checkpoint_path: Path


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

    experiment: ExperimentConfig
    success: bool
    error_message: str | None = None
    best_preconditioner: dict[str, Any] | None = None
    plot_paths: dict[str, Path] | None = None


def load_experiments_config(config_path: Path) -> dict[str, Any]:
    """Load experiments configuration from TOML file.

    Pure I/O action - reads config file.

    Args:
        config_path: Path to experiments.toml

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config parsing fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Experiments config not found: {config_path}")

    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except Exception as exc:
        raise ValueError(f"Error loading experiments config: {exc}") from exc


def extract_experiments_list(config: dict[str, Any]) -> list[dict[str, str]]:
    """Extract experiments list from config dictionary.

    Pure function - no I/O.

    Args:
        config: Configuration dictionary

    Returns:
        List of experiment dictionaries

    Raises:
        ValueError: If no experiments defined
    """
    experiments = config.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments defined in config")
    return experiments


def extract_output_root(config: dict[str, Any]) -> Path:
    """Extract output root path from config.

    Pure function - no I/O.

    Args:
        config: Configuration dictionary

    Returns:
        Output root path
    """
    paths_cfg = config.get("paths", {})
    output_root_str = paths_cfg.get("output_root", "/data/projects/graph-cg/data/output")
    return Path(output_root_str)


def build_experiment_config(
    experiment_dict: dict[str, str],
    root_dir: Path,
    output_root: Path,
) -> ExperimentConfig:
    """Build ExperimentConfig from dictionary.

    Pure function - only path construction.

    Args:
        experiment_dict: Raw experiment dictionary from TOML
        root_dir: Project root directory
        output_root: Output root path

    Returns:
        Structured experiment configuration
    """
    model_template = root_dir / experiment_dict["model_template"]
    data_config = root_dir / experiment_dict["data_config"]
    solver_config = root_dir / experiment_dict.get("solver_config", "solver-configs/default.toml")

    # Check for optional checkpoint_path override
    checkpoint_path_str = experiment_dict.get("checkpoint_path")

    if checkpoint_path_str is not None:
        # Use explicit checkpoint path
        checkpoint_path = Path(checkpoint_path_str)

        # Resolve relative paths against project root
        if not checkpoint_path.is_absolute():
            checkpoint_path = root_dir / checkpoint_path
    else:
        # Fallback to convention-based derivation
        checkpoint_path = derive_checkpoint_path(model_template, data_config, output_root)

    return ExperimentConfig(
        model_template=model_template,
        data_config=data_config,
        solver_config=solver_config,
        checkpoint_path=checkpoint_path,
    )


def validate_checkpoint_exists(experiment: ExperimentConfig) -> str | None:
    """Validate that checkpoint file exists.

    I/O action - checks filesystem.

    Args:
        experiment: Experiment configuration

    Returns:
        Error message if checkpoint missing, None otherwise
    """
    if not experiment.checkpoint_path.exists():
        return f"Checkpoint not found: {experiment.checkpoint_path}"
    return None


def log_experiment_header(idx: int, total: int, experiment: ExperimentConfig) -> None:
    """Log experiment header information.

    I/O action - logs to stdout.

    Args:
        idx: Current experiment index (1-based)
        total: Total number of experiments
        experiment: Experiment configuration
    """
    logger.info("=" * 80)
    logger.info(f"Experiment {idx}/{total}")
    logger.info("=" * 80)
    logger.info(f"Model:      {experiment.model_template.name}")
    logger.info(f"Data:       {experiment.data_config.name}")
    logger.info(f"Solver:     {experiment.solver_config.name}")
    logger.info(f"Checkpoint: {experiment.checkpoint_path}")
    logger.info("")


def run_single_comparison(
    experiment: ExperimentConfig,
    params: ComparisonParams,
) -> dict[str, Any]:
    """Run comparison for a single experiment.

    I/O action - calls comparison function.

    Args:
        experiment: Experiment configuration
        params: Comparison parameters

    Returns:
        Comparison result dictionary
    """
    return compare_preconditioners(
        config_path=experiment.model_template,
        data_config_path=experiment.data_config,
        solver_config_path=experiment.solver_config,
        checkpoint_path=experiment.checkpoint_path,
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
    """Log comparison results.

    I/O action - logs to stdout.

    Args:
        result: Comparison result dictionary
    """
    logger.info(f"Available preconditioners: {result['preconditioners']}")

    # Log neural preconditioner metadata
    precond_metadata = result.get("preconditioner_metadata", {})
    neural_meta = precond_metadata.get("neural")

    if neural_meta is not None:
        residual_iters = getattr(neural_meta, "residual_iters", "unknown")
        applied_iters = getattr(neural_meta, "applied_iters", None)
        applied_str = f"{applied_iters}" if applied_iters is not None else f"{residual_iters} (default)"
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

    # Log plot paths (only if plots were saved)
    plot_paths = result.get("plot_paths", {})
    if plot_paths:
        logger.info("Generated plots:")
        for plot_type, path in plot_paths.items():
            logger.info(f"  {plot_type}: {path}")


def process_experiment(
    experiment: ExperimentConfig,
    params: ComparisonParams,
    idx: int,
    total: int,
) -> ExperimentResult:
    """Process a single experiment comparison.

    I/O orchestrator - coordinates experiment execution.

    Args:
        experiment: Experiment configuration
        params: Comparison parameters
        idx: Current experiment index (1-based)
        total: Total number of experiments

    Returns:
        Experiment result
    """
    log_experiment_header(idx, total, experiment)

    # Guard: Validate checkpoint exists
    checkpoint_error = validate_checkpoint_exists(experiment)
    if checkpoint_error is not None:
        logger.warning(f"⚠ Skipping: {checkpoint_error}")
        return ExperimentResult(
            experiment=experiment,
            success=False,
            error_message=checkpoint_error,
        )

    # Run comparison
    try:
        result = run_single_comparison(experiment, params)
        log_comparison_results(result)
        logger.info(f"{SYMBOL_CHECKMARK} Comparison complete for experiment {idx}!")

        recs = result.get("recommendations", {})
        best_overall = recs.get("best_overall") if isinstance(recs, dict) else None

        return ExperimentResult(
            experiment=experiment,
            success=True,
            best_preconditioner=best_overall,
            plot_paths=result["plot_paths"],
        )

    except Exception as exc:
        error_msg = f"Comparison failed: {exc}"
        logger.error(error_msg)
        logger.exception("Full traceback:")
        return ExperimentResult(
            experiment=experiment,
            success=False,
            error_message=error_msg,
        )


def log_final_summary(results: list[ExperimentResult]) -> None:
    """Log final summary of all experiments.

    I/O action - logs to stdout.

    Args:
        results: List of experiment results
    """
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful

    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total experiments: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")

    # Log failed experiments
    failed_results = [r for r in results if not r.success]
    if failed_results:
        logger.info("")
        logger.info("Failed experiments:")
        for result in failed_results:
            exp = result.experiment
            logger.error(
                f"  ✗ {exp.model_template.name} + {exp.data_config.name}: {result.error_message}"
            )
    else:
        logger.info(f"{SYMBOL_CHECKMARK} All comparisons completed successfully!")


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
        help="Override matrix path (npz supported). If omitted, uses solver-config or derived paths.",
    ),
    rhs: Path | None = typer.Option(
        None,
        help="Override RHS path (npz supported). If omitted, uses solver-config or derived paths.",
    ),
    output_dir: Path | None = typer.Option(
        None, help="Override output directory for all experiments"
    ),
    figures_dir: Path | None = typer.Option(
        None, help="Override figures directory for plots"
    ),
    noise_strategy: str = typer.Option(NOISE_STRATEGY_NONE, help="Noise strategy"),
    noise_level: float = typer.Option(DEFAULT_NOISE_LEVEL, help="Noise level parameter"),
    noise_seed: int | None = typer.Option(DEFAULT_NOISE_SEED, help="Random seed for noise"),
    breakdown_tol: float | None = typer.Option(
        None, help="Breakdown tolerance for CG denominator checks"
    ),
    neural_precond_iters: int | None = typer.Option(
        None, help="Limit neural preconditioning to first L iterations"
    ),
    fallback_preconditioner: str = typer.Option(
        "identity", help="Preconditioner after L neural iterations (identity, jacobi, ilu)"
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
    """Compare preconditioner methods for all experiments in experiments.toml.

    This command runs preconditioner comparisons for every experiment defined in the
    experiments config. Each experiment specifies its model, data, and solver configs.

    Checkpoint paths are auto-derived from the configuration by default, or can be
    explicitly specified in experiments.toml using the optional checkpoint_path field.
    This enables comparing multiple checkpoint versions of the same model.

    Example:
        $ uv run python scripts/compare_methods.py
        $ uv run python scripts/compare_methods.py --experiments custom.toml
        $ uv run python scripts/compare_methods.py --matrix /path/to/test.npz
    """
    root_dir = Path(__file__).resolve().parent.parent

    # Resolve experiments config
    experiments_path = experiments if experiments is not None else root_dir / DEFAULT_EXPERIMENTS_CONFIG

    # Load and validate configuration
    try:
        config = load_experiments_config(experiments_path)
        experiments_list = extract_experiments_list(config)
        output_root = extract_output_root(config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        raise typer.Exit(code=EXIT_FAILURE)

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

    # Build experiment configurations
    experiment_configs = [
        build_experiment_config(exp_dict, root_dir, output_root)
        for exp_dict in experiments_list
    ]

    # Process all experiments
    results = [
        process_experiment(exp, params, idx, len(experiment_configs))
        for idx, exp in enumerate(experiment_configs, 1)
    ]

    # Log final summary
    log_final_summary(results)

    # Exit with error if any experiments failed
    if any(not r.success for r in results):
        raise typer.Exit(code=EXIT_FAILURE)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
