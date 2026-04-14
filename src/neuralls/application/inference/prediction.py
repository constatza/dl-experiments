"""Prediction execution logic for inference workflow.

This module provides pure functions for running model predictions,
including batch processing and result collection.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from loguru import logger

from neuralls.application.inference.models import InferenceData, InferencePredictions


def iterate_feature_batches(
    feature_arrays: dict[str, np.ndarray],
    batch_size: int,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield contiguous feature batches for streaming prediction.

    Args:
        feature_arrays: Feature arrays keyed by name.
        batch_size: Number of samples per batch.

    Yields:
        Batched feature dictionaries.
    """
    total = next(iter(feature_arrays.values())).shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield {name: array[start:end] for name, array in feature_arrays.items()}


def collect_predictions(
    predictor: Any,
    feature_arrays: dict[str, np.ndarray],
    batch_size: int,
) -> tuple[list[torch.Tensor], float]:
    """Run inference for every feature batch.

    Args:
        predictor: DLKit CheckpointPredictor instance.
        feature_arrays: Feature arrays keyed by name.
        batch_size: Number of samples per batch.

    Returns:
        Tuple of (predictions_list, 0.0) — duration not available in new API.

    Raises:
        ValueError: If predictor returns no predictions.
    """
    predictions: list[torch.Tensor] = []
    for batch in iterate_feature_batches(feature_arrays, batch_size):
        tensor_batch = {
            k: torch.from_numpy(np.asarray(v, dtype=np.float64)) for k, v in batch.items()
        }
        result = predictor.predict(**tensor_batch)
        primary = result[0] if isinstance(result, tuple) else result
        predictions.append(primary)
    if not predictions:
        raise ValueError("Predictor returned no predictions.")
    return predictions, 0.0


def process_predictions(
    raw_predictions: list[torch.Tensor],
) -> np.ndarray:
    """Process raw prediction batches into single array.

    Args:
        raw_predictions: List of prediction tensors from predictor.

    Returns:
        Flattened prediction array.
    """
    normalized = [
        prediction if prediction.ndim > 0 else prediction.unsqueeze(0)
        for prediction in raw_predictions
    ]
    stacked = torch.cat(normalized, dim=0).detach().cpu().numpy()
    return stacked.ravel()


def run_prediction(
    predictor: Any,
    data: InferenceData,
    batch_size: int = 256,
) -> InferencePredictions:
    """Execute prediction on inference data.

    Transforms are applied automatically via checkpoint-based transforms
    (enabled with apply_transforms=True in predictor creation).

    Args:
        predictor: DLKit predictor instance with transforms enabled.
        data: Inference data with features and targets.
        batch_size: Batch size for inference (defaults to 256).

    Returns:
        InferencePredictions with results and metadata.
    """
    feature_name, feature_array = next(iter(data.features.items()))
    feature_arrays = {feature_name: feature_array}

    logger.info(f"Running inference on {data.metadata.get('source', 'unknown')} data...")
    raw_preds, duration = collect_predictions(predictor, feature_arrays, batch_size)

    predictions = process_predictions(raw_preds)

    targets = next(iter(data.targets.values())) if data.targets else np.zeros_like(predictions)

    return InferencePredictions(
        predictions={"y_pred": predictions},
        targets={"y_true": targets},
        metadata={
            **data.metadata,
            "duration_seconds": duration,
            "num_predictions": len(predictions),
        },
    )
