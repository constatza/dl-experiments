"""Functions for saving prediction diagnostics."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


def save_prediction_samples_to_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    filename_prefix: str,
    num_samples: int = 1,
    seed: int | None = None,
) -> list[Path] | None:
    """Save sampled true, predicted, and error vectors to individual CSV files.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        output_dir: Directory to save the CSV file.
        filename_prefix: Prefix for the CSV filename.
        num_samples: Number of vectors to sample (defaults to 1).
        seed: Optional seed for the random number generator for reproducibility.

    Returns:
        List of saved CSV paths, or None if unable to save.
    """
    if y_true is None or y_pred is None:
        logger.warning("y_true or y_pred is None; skipping CSV sample saving.")
        return None

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Align shapes when predictions were flattened
    if y_true.ndim > 1 and y_pred.ndim == 1 and y_true.size == y_pred.size:
        y_pred = y_pred.reshape(y_true.shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(seed)

    if y_true.shape != y_pred.shape and not (
        y_true.ndim == 1 and y_pred.ndim == 1 and y_true.size == y_pred.size
    ):
        logger.warning("Shape mismatch between y_true and y_pred; cannot save samples to CSV.")
        return None

    num_vectors = y_true.shape[0] if y_true.ndim > 1 else 1
    if num_vectors == 0:
        logger.info("No vectors to save for CSV.")
        return None

    effective_samples = max(0, min(num_samples, num_vectors))
    if effective_samples == 0:
        logger.info("Requested zero samples; skipping CSV save.")
        return None

    sample_indices = (
        rng.choice(num_vectors, size=effective_samples, replace=False)
        if num_vectors > 1
        else np.array([0])
    )

    saved_paths: list[Path] = []

    for idx in sample_indices:
        true_vec = y_true[idx] if y_true.ndim > 1 else y_true
        pred_vec = y_pred[idx] if y_pred.ndim > 1 else y_pred

        if true_vec.shape != pred_vec.shape:
            logger.warning(f"Shape mismatch for sampled vector {idx}; skipping.")
            continue

        error_vec = true_vec - pred_vec
        records = []
        for t_val, p_val, e_val in zip(
            true_vec.ravel(), pred_vec.ravel(), error_vec.ravel()
        ):
            records.append(
                {
                    "y_true": t_val,
                    "y_pred": p_val,
                    "error": e_val,
                }
            )

        if not records:
            logger.warning(f"No entries recorded for sampled vector {idx}; skipping.")
            continue

        df = pd.DataFrame.from_records(
            records, columns=["y_true", "y_pred", "error"]
        )

        dim_count = true_vec.size
        filepath = output_dir / f"{filename_prefix}_predictions_sample_{idx}_dim{dim_count}.csv"
        df.to_csv(filepath, index=False)
        saved_paths.append(filepath)
        logger.info(f"Saved {len(records)} prediction entries to {filepath}")

    if not saved_paths:
        logger.warning("No prediction samples recorded; CSV not written.")
        return None

    return saved_paths
