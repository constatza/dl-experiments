#!/usr/bin/env python3
"""Lightweight CLI wrapper around src.cli.prediction.run_inference."""

from __future__ import annotations

import sys
from pathlib import Path

# Add graph-cg root to Python path so we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

import numpy as np
import pandas as pd
import typer
from dlkit.core.postprocessing import summarize
from loguru import logger

from src.constants import (
    DEFAULT_MODEL_CONFIG,
    DEFAULT_DATA_CONFIG,
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
)
from src.cli.prediction import run_inference

os.environ.setdefault("MPLBACKEND", "Agg")


def _save_sample_prediction_to_csv(
    y_true: np.ndarray, y_pred: np.ndarray, savepath: Path
) -> None:
    """Saves a random sample's true value, prediction, and error to a CSV file."""
    if y_true.ndim != 2 or y_pred.ndim != 2:
        logger.warning("y_true and y_pred must be 2D arrays.")
        return
    num_samples = y_true.shape[0]
    random_index = np.random.randint(0, num_samples)
    true_val = y_true[random_index, :]
    predicted_val = y_pred[random_index, :]
    error_val = true_val - predicted_val

    sample_data = pd.DataFrame(
        {"Target": [true_val], "Prediction": [predicted_val], "Error": [error_val]}
    )

    savepath.mkdir(parents=True, exist_ok=True)

    try:
        sample_data.to_csv(savepath, index=False, float_format="%.6e")
        logger.info(f"Saved random sample prediction, target, and error to {savepath}")
    except Exception as e:
        logger.error(
            f"Failed to save random sample prediction, target, and error to {savepath}: {e}"
        )


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
):
    """Run inference using a DLKit configuration."""
    graph_cg_root = Path(__file__).resolve().parent.parent

    # Resolve defaults
    if config is None:
        config = graph_cg_root / DEFAULT_MODEL_CONFIG
    if data_config is None:
        data_config = graph_cg_root / DEFAULT_DATA_CONFIG

    logger.info(f"Loading configuration from: {config}")

    try:
        results = run_inference(
            config_path=config,
            data_config_path=data_config,
            checkpoint_path=checkpoint,
            features_path=features,
            targets_path=targets,
            save_plots=not no_plots,
            figures_dir=figures_dir,
        )

        try:
            logger.info(f"Prediction summary: {summarize(results['predictions'])}")
        except Exception:
            pass

        if results["y_true"] is not None and results["y_pred"] is not None:
            logger.info(f"Generated predictions for {len(results['y_true'])} samples")
            _save_sample_prediction_to_csv(
                results["y_true"],
                results["y_pred"],
                figures_dir.parent / "diagnostics" / "sample_prediction.csv",
            )
            if results["plot_path"]:
                logger.info(f"Saved parity plot to: {results['plot_path']}")
            if results.get("diagnostic_plot_path"):
                logger.info(
                    f"Saved diagnostic plot to: {results['diagnostic_plot_path']}"
                )
        else:
            logger.warning(
                "Could not extract matching prediction/target arrays for plotting."
            )

        logger.info(f"Inference completed in {results['duration_seconds']:.2f}s")

    except Exception as exc:  # noqa: BLE001
        logger.error(f"Error: {exc}")
        raise typer.Exit(code=EXIT_FAILURE)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
