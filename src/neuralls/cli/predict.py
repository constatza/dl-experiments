#!/usr/bin/env python3
"""Lightweight CLI wrapper around neuralls.composition.experiments.inference.run_inference."""

from __future__ import annotations

from pathlib import Path


import os

import typer
from loguru import logger

from neuralls.shared.constants import (
    EXIT_KEYBOARD_INTERRUPT,
)
from neuralls.composition.experiments.inference import run_inference
from neuralls.platform.config.settings import CASE_CONFIG_ENV_VAR, load_case_settings

os.environ.setdefault("MPLBACKEND", "Agg")


def main(
    config: Path = typer.Option(..., help="Path to model config file"),
    data_config: Path = typer.Option(
        ...,
        help="Path to data config providing dataset metadata",
    ),
    case_config: Path | None = typer.Option(
        None,
        "--case-config",
        help="Path to case config TOML.",
    ),
    checkpoint: Path | None = typer.Option(None, help="Override checkpoint path"),
    features: Path | None = typer.Option(
        None, help="Override features path (required if config has no DATASET)"
    ),
    targets: Path | None = typer.Option(
        None, help="Override targets path (required if config has no DATASET)"
    ),
    no_plots: bool = typer.Option(False, help="Skip saving plots"),
    figures_dir: Path | None = typer.Option(None, help="Override directory for saved plots"),
    enable_mlflow: bool = typer.Option(
        False,
        "--enable-mlflow/--no-mlflow",
        help="Enable MLflow logging (defaults to disabled)",
    ),
    synthetic: bool = typer.Option(
        False,
        "--synthetic",
        help="Run synthetic benchmark (x_true=ones, b=Ax) instead of data loading",
    ),
    comparison_config: Path | None = typer.Option(
        None,
        help="Path to comparison config (used for matrix path in synthetic mode)",
    ),
    env_file: Path | None = typer.Option(
        None,
        help="Optional env file to load before config resolution.",
    ),
) -> None:
    """Run inference using a DLKit configuration."""
    logger.info(f"Loading configuration from: {config}")
    resolved_case_config = case_config
    if resolved_case_config is None:
        configured = os.getenv(CASE_CONFIG_ENV_VAR)
        if configured is not None and configured.strip():
            resolved_case_config = Path(configured)
    if resolved_case_config is None:
        raise typer.BadParameter(
            "This command requires a case config. Pass --case-config or set "
            "NEURALLS_CASE_CONFIG."
        )
    settings = load_case_settings(resolved_case_config, env_file)

    results = run_inference(
        config_path=config,
        settings=settings,
        case_config_path=resolved_case_config,
        data_config_path=data_config,
        checkpoint_path=checkpoint,
        features_path=features,
        targets_path=targets,
        save_plots=not no_plots,
        figures_dir=figures_dir,
        enable_mlflow=enable_mlflow,
        synthetic_benchmark=synthetic,
        comparison_config_path=comparison_config,
    )

    if results.y_pred is not None:
        import numpy as np

        logger.info(
            f"Prediction summary: mean={float(np.mean(results.y_pred)):.4e}, std={float(np.std(results.y_pred)):.4e}"
        )

    if results.y_true is not None and results.y_pred is not None:
        logger.info(f"Generated predictions for {len(results.y_true)} samples")
        if results.plot_path:
            logger.info(f"Saved parity plot to: {results.plot_path}")
        if results.diagnostic_plot_path:
            logger.info(f"Saved diagnostic plot to: {results.diagnostic_plot_path}")
        elif not no_plots:
            logger.warning("Could not extract matching prediction/target arrays for plotting.")

        logger.info(f"Inference completed in {results.duration_seconds:.2f}s")


def run() -> None:
    """Entry point for pyproject.toml script registration."""
    typer.run(main)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
