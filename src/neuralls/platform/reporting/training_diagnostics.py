"""Pure diagnostic functions for evaluating model predictions.

This module provides stateless, pure functions for computing prediction
error metrics. No side effects — caller decides where to log results.

Functions:
    compute_diagnostics: Compute error metrics from prediction arrays.

Domain models:
    PredictionDiagnostics: Immutable container for computed metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PredictionDiagnostics:
    """Immutable container for prediction error metrics.

    Attributes:
        mae: Mean absolute error
        mse: Mean squared error
        duration_seconds: Inference duration (0.0 if unknown)
        metrics: MLflow-ready flat dict of all metrics under 'eval/' prefix
    """

    mae: float
    mse: float
    duration_seconds: float
    metrics: dict[str, float]


def compute_diagnostics(
    y_pred: NDArray,
    y_true: NDArray,
    duration: float = 0.0,
) -> PredictionDiagnostics:
    """Compute error metrics from prediction and target arrays.

    Pure function: no I/O, no side effects.

    Args:
        y_pred: Predicted values, shape (N,) or (N, D)
        y_true: True target values, same shape as y_pred
        duration: Elapsed inference time in seconds

    Returns:
        PredictionDiagnostics with mae, mse, and mlflow-ready metrics
    """
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    true = np.asarray(y_true, dtype=np.float64).ravel()

    diff = pred - true
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))

    metrics = {
        "eval/mae": mae,
        "eval/mse": mse,
        "eval/duration_seconds": duration,
    }

    return PredictionDiagnostics(
        mae=mae,
        mse=mse,
        duration_seconds=duration,
        metrics=metrics,
    )


def write_diagnostics_figure(
    y_true: Any,
    y_pred: Any,
    figures_dir: Path,
) -> Path:
    """Write a parity + residuals diagnostics figure to disk.

    Args:
        y_true: True target values.
        y_pred: Predicted values.
        figures_dir: Directory to write the figure into.

    Returns:
        Path to the saved figure file.
    """
    from neuralls.platform.reporting.plots import plot_parity_and_residuals

    figure_path = figures_dir / "diagnostics_training.png"
    plot_parity_and_residuals(
        y_true.ravel(),
        y_pred.ravel(),
        save_path=figure_path,
    )
    return figure_path
