"""Prediction execution logic for inference workflow.

This module provides pure functions for running model predictions,
including predictor creation, batch processing, and result collection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Iterator

import numpy as np
from loguru import logger

from dlkit import load_predictor
from dlkit.core.postprocessing import stack_batches
from dlkit.tools.config.precision.strategy import PrecisionStrategy
from dlkit.tools.config.data_entries import Feature, FeatureType, Target, TargetType
from dlkit.tools.config.dataset_settings import DatasetSettings

from neuralls.workflows.inference.config import InferenceData, InferencePredictions


# NOTE: ensure_dataset_settings() removed - no longer needed with InferenceWorkflowConfig
# DATASET section is optional for inference; transforms loaded from checkpoint metadata


def resolve_batch_size(settings: Any) -> int:
    """Resolve batch size from settings.

    Args:
        settings: DLKit GeneralSettings object

    Returns:
        Batch size (defaults to 256 if not configured)
    """
    datamodule = getattr(settings, "DATAMODULE", None)
    dataloader = getattr(datamodule, "dataloader", None) if datamodule else None
    configured = getattr(dataloader, "batch_size", None)
    try:
        return int(configured) if configured is not None else 256
    except (TypeError, ValueError):
        return 256


def iterate_feature_batches(
    feature_arrays: dict[str, np.ndarray],
    batch_size: int,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield contiguous feature batches for streaming prediction.

    Args:
        feature_arrays: Feature arrays keyed by name
        batch_size: Number of samples per batch

    Yields:
        Batched feature dictionaries
    """
    total = next(iter(feature_arrays.values())).shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield {name: array[start:end] for name, array in feature_arrays.items()}


def collect_predictions(
    predictor: Any,
    feature_arrays: dict[str, np.ndarray],
    batch_size: int,
) -> tuple[list[Any], float]:
    """Run inference for every feature batch.

    Args:
        predictor: DLKit predictor instance
        feature_arrays: Feature arrays keyed by name
        batch_size: Number of samples per batch

    Returns:
        Tuple of (predictions_list, total_duration_seconds)

    Raises:
        ValueError: If predictor returns no predictions
    """
    predictions: list[Any] = []
    total_duration = 0.0
    for batch in iterate_feature_batches(feature_arrays, batch_size):
        result = predictor.predict(batch)
        predictions.append(result.predictions)
        total_duration += getattr(result, "duration_seconds", 0.0)
    if not predictions:
        raise ValueError("Predictor returned no predictions.")
    return predictions, total_duration


def process_predictions(
    raw_predictions: list[Any],
) -> np.ndarray:
    """Process raw prediction batches into single array.

    Args:
        raw_predictions: List of prediction batches from predictor

    Returns:
        Flattened prediction array
    """
    stacked = stack_batches(raw_predictions, mode="stack")
    if isinstance(stacked, dict):
        return next(iter(stacked.values())).ravel()
    return stacked.ravel()


def create_predictor(
    checkpoint_path: Path,
    settings: Any,
) -> Any:
    """Create DLKit predictor from checkpoint with fitted transforms.

    Uses FULL_64 precision and enables checkpoint-based transforms.
    Fitted transforms (normalization, scaling, etc.) are automatically
    loaded from checkpoint metadata and applied during prediction.

    Args:
        checkpoint_path: Path to model checkpoint
        settings: DLKit GeneralSettings (for validation)

    Returns:
        DLKit predictor context manager with fitted transforms enabled
    """
    logger.debug(f"Loading checkpoint from: {checkpoint_path}")
    logger.debug("Using apply_transforms=True with precision=FULL_64")

    predictor = load_predictor(
        str(checkpoint_path),
        apply_transforms=True,  # Enable checkpoint-based transforms
        precision=PrecisionStrategy.FULL_64,
    )

    # Verify transforms loaded from checkpoint
    if hasattr(predictor, "model_state"):
        feature_transforms = predictor.model_state.feature_transforms
        target_transforms = predictor.model_state.target_transforms

        if feature_transforms:
            logger.info(
                f"Loaded {len(feature_transforms)} feature transform chain(s) from checkpoint"
            )
        if target_transforms:
            logger.info(
                f"Loaded {len(target_transforms)} target transform chain(s) from checkpoint"
            )
    else:
        logger.warning("Predictor missing model_state - cannot verify transforms")

    return predictor


def run_prediction(
    predictor: Any,
    data: InferenceData,
    settings: Any,
    batch_size: int | None = None,
) -> InferencePredictions:
    """Execute prediction on inference data.

    Transforms are applied automatically via checkpoint-based transforms
    (enabled with apply_transforms=True in predictor creation).

    Args:
        predictor: DLKit predictor instance with transforms enabled
        data: Inference data with features and targets
        settings: DLKit settings (for batch size fallback)
        batch_size: Optional batch size override (defaults to settings or 256)

    Returns:
        InferencePredictions with results and metadata
    """
    # Resolve batch size (override > settings > 256 default)
    if batch_size is None:
        batch_size = resolve_batch_size(settings)

    # Get feature name from data (first key in features dict)
    # No longer depends on settings.DATASET!
    feature_name, feature_array = next(iter(data.features.items()))

    # Prepare feature dict for prediction
    feature_arrays = {feature_name: feature_array}

    # Collect predictions
    logger.info(f"Running inference on {data.metadata.get('source', 'unknown')} data...")
    raw_preds, duration = collect_predictions(predictor, feature_arrays, batch_size)

    # Process predictions
    predictions = process_predictions(raw_preds)

    # Extract targets (may be None for some workflows)
    targets = (
        next(iter(data.targets.values()))
        if data.targets
        else np.zeros_like(predictions)
    )

    return InferencePredictions(
        predictions={"y_pred": predictions},
        targets={"y_true": targets},
        metadata={
            **data.metadata,
            "duration_seconds": duration,
            "num_predictions": len(predictions),
        },
    )
