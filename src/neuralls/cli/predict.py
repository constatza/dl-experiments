#!/usr/bin/env python3
"""Lightweight CLI wrapper around neuralls.workflows.prediction.run_inference."""

from __future__ import annotations

from pathlib import Path


import os

import typer
from dlkit.core.postprocessing import summarize
from loguru import logger

from neuralls.constants import (
    DEFAULT_MODEL_CONFIG,
    DEFAULT_DATA_CONFIG,
    DEFAULT_PROJECT_ROOT,
    EXIT_KEYBOARD_INTERRUPT,
)
from neuralls.workflows.prediction import run_inference

os.environ.setdefault("MPLBACKEND", "Agg")


def main(
    config: Path = typer.Option(None, help="Path to config file"),
    data_config: Path = typer.Option(
        None,
        help="Path to data config providing dataset metadata",
    ),
    checkpoint: Path | None = typer.Option(None, help="Override checkpoint path"),
    features: Path | None = typer.Option(
        None, help="Override features path (required if config has no DATASET)"
    ),
    targets: Path | None = typer.Option(
        None, help="Override targets path (required if config has no DATASET)"
    ),
    no_plots: bool = typer.Option(False, help="Skip saving plots"),
    figures_dir: Path | None = typer.Option(
        None, help="Override directory for saved plots"
    ),
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
    solver_config: Path | None = typer.Option(
        None,
        help="Path to solver config (used for matrix path in synthetic mode)",
    ),
) -> None:
    """Run inference using a DLKit configuration."""
    # Resolve defaults
    if config is None:
        config = DEFAULT_PROJECT_ROOT / DEFAULT_MODEL_CONFIG
    if data_config is None:
        data_config = DEFAULT_PROJECT_ROOT / DEFAULT_DATA_CONFIG

    logger.info(f"Loading configuration from: {config}")

    results = run_inference(
        config_path=config,
        data_config_path=data_config,
        checkpoint_path=checkpoint,
        features_path=features,
        targets_path=targets,
        save_plots=not no_plots,
        figures_dir=figures_dir,
        enable_mlflow=enable_mlflow,
        synthetic_benchmark=synthetic,
        solver_config_path=solver_config,
    )

    logger.info(f"Prediction summary: {summarize(results['predictions'])}")

    if results["y_true"] is not None and results["y_pred"] is not None:
        logger.info(f"Generated predictions for {len(results['y_true'])} samples")
        if results["plot_path"]:
            logger.info(f"Saved parity plot to: {results['plot_path']}")
        if results.get("diagnostic_plot_path"):
            logger.info(f"Saved diagnostic plot to: {results['diagnostic_plot_path']}")
        else:
            logger.warning(
                "Could not extract matching prediction/target arrays for plotting."
            )

        logger.info(f"Inference completed in {results['duration_seconds']:.2f}s")


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
