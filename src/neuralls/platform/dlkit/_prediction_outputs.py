"""Private DLKit prediction-result normalization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class SupportsPredictionTensor(Protocol):
    """Structural contract for DLKit prediction results used by this repo."""

    predictions: torch.Tensor


def extract_prediction_tensor(result: object) -> torch.Tensor:
    """Extract the primary tensor from a DLKit predictor result."""
    if not isinstance(result, SupportsPredictionTensor):
        raise TypeError(
            "Unsupported predictor output type: "
            f"{type(result).__name__}. Expected an object with a 'predictions' tensor."
        )

    primary = result.predictions

    if not isinstance(primary, torch.Tensor):
        raise TypeError(
            "Unsupported primary predictor output type: "
            f"{type(primary).__name__}. Expected 'predictions' to be a torch.Tensor."
        )

    return primary


def feature_batch_to_tensors(feature_batch: dict[str, NDArray]) -> dict[str, torch.Tensor]:
    """Convert one numpy feature batch to float64 tensors."""
    return {
        name: torch.from_numpy(np.asarray(values, dtype=np.float64))
        for name, values in feature_batch.items()
    }


def prediction_tensor_to_numpy(predictions: torch.Tensor) -> NDArray:
    """Convert one prediction tensor to a float64 numpy array."""
    array = predictions.detach().cpu().numpy()
    return array.astype(np.float64, copy=False)
