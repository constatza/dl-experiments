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

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PredictionDiagnostics:
    """Immutable container for prediction error metrics.

    Attributes:
        rel_error: Relative L2 error ||y_pred - y_true|| / ||y_true||
        mae: Mean absolute error
        mse: Mean squared error
        duration_seconds: Inference duration (0.0 if unknown)
        metrics: MLflow-ready flat dict of all metrics under 'eval/' prefix
    """

    rel_error: float
    mae: float
    mse: float
    duration_seconds: float
    metrics: dict[str, float]


_EPSILON = 1e-12


def _per_sample_rel_l2(
    y_pred: NDArray,
    y_true: NDArray,
    eps: float = _EPSILON,
) -> float:
    """Compute mean per-sample normalized L2 error matching NormalizedVectorNormError.

    Formula: mean_i( ||pred_i - true_i||_2 / max(||true_i||_2, eps) )

    If inputs are 1D, falls back to global relative L2 norm for backward compatibility.

    Args:
        y_pred: Predicted values, shape (N,) or (N, D)
        y_true: True target values, same shape as y_pred
        eps: Epsilon to avoid division by zero

    Returns:
        Scalar mean normalized L2 error
    """
    pred = np.asarray(y_pred, dtype=np.float64)
    true = np.asarray(y_true, dtype=np.float64)
    if pred.ndim == 2:
        diff_norms = np.linalg.norm(pred - true, axis=1)
        true_norms = np.maximum(np.linalg.norm(true, axis=1), eps)
        return float(np.mean(diff_norms / true_norms))
    # 1D fallback: global relative L2
    diff = pred.ravel() - true.ravel()
    norm_true = float(np.linalg.norm(true.ravel()))
    return float(np.linalg.norm(diff)) / max(norm_true, eps)


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
        PredictionDiagnostics with rel_error, mae, mse, and mlflow-ready metrics
    """
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    true = np.asarray(y_true, dtype=np.float64).ravel()

    diff = pred - true
    rel_error = _per_sample_rel_l2(y_pred, y_true)
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))

    metrics = {
        "eval/rel_error": rel_error,
        "eval/mae": mae,
        "eval/mse": mse,
        "eval/duration_seconds": duration,
    }

    return PredictionDiagnostics(
        rel_error=rel_error,
        mae=mae,
        mse=mse,
        duration_seconds=duration,
        metrics=metrics,
    )
