"""DLKit-backed adapter for application-level batch inference."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from dlkit import load_model
from dlkit.infrastructure.precision.strategy import PrecisionStrategy
from loguru import logger

from neuralls.domain.inference_ports import InferencePredictorPort

from ._prediction_outputs import (
    extract_prediction_tensor,
    feature_batch_to_tensors,
    prediction_tensor_to_numpy,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from dlkit.interfaces.inference import CheckpointPredictor


class DLKitInferencePredictor(InferencePredictorPort):
    """DLKit-backed predictor for batched inference use cases."""

    def __init__(self, predictor: CheckpointPredictor, device: str) -> None:
        self._predictor: CheckpointPredictor = predictor
        self._device: str = device

    def predict_batch(self, feature_batch: dict[str, NDArray]) -> NDArray:
        """Predict one batch from named numpy feature arrays."""
        if not feature_batch:
            raise ValueError("Feature batch must not be empty.")

        try:
            tensor_batch = feature_batch_to_tensors(feature_batch)
            output = self._predictor.predict(**tensor_batch)
            primary = extract_prediction_tensor(output)
            return prediction_tensor_to_numpy(primary)

        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                f"GPU out of memory during inference on device {self._device}. "
                f"Feature keys: {tuple(feature_batch.keys())}. "
                "Try reducing batch size or using CPU."
            ) from e

        except RuntimeError as e:
            if "device" in str(e).lower():
                raise RuntimeError(
                    f"Device mismatch: model on {self._device}, "
                    f"but tensor operation failed. Error: {e}"
                ) from e
            raise RuntimeError(
                f"Inference batch prediction failed: {e}. "
                f"Model device: {self._device}, Feature keys: {tuple(feature_batch.keys())}"
            ) from e

        except (TypeError, AttributeError, ValueError) as e:
            raise RuntimeError(
                f"Unexpected error during inference batch prediction: {type(e).__name__}: {e}"
            ) from e

    def cleanup(self) -> None:
        """No-op — use as context manager; __exit__ handles unload."""
        pass

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Delegate lifecycle to CheckpointPredictor's context manager."""
        self._predictor.__exit__(exc_type, exc_val, exc_tb)


def create_inference_predictor(checkpoint_path: Path, settings: Any) -> InferencePredictorPort:
    """Create a DLKit-backed inference predictor with fitted transforms."""
    del settings

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Ensure the path is correct and the file exists."
        )

    try:
        logger.debug(f"Loading inference checkpoint from: {checkpoint_path}")
        predictor = load_model(
            checkpoint_path,
            device="auto",
            apply_transforms=True,
            precision=PrecisionStrategy.FULL_64,
        )

        model = predictor.model
        if model is None:
            raise RuntimeError(
                f"Failed to load model from checkpoint: {checkpoint_path}. "
                "Checkpoint may be corrupted or incompatible."
            )

        device_param = next(model.parameters(), None)
        device = str(device_param.device) if device_param is not None else "cpu"
        if device_param is None:
            logger.warning("Model has no parameters, defaulting to CPU")

        logger.info(f"Loaded inference model from {checkpoint_path} on device {device}")
        return DLKitInferencePredictor(predictor, device)

    except (FileNotFoundError, OSError, ValueError, RuntimeError) as e:
        raise RuntimeError(
            f"Failed to load inference model from {checkpoint_path}: {type(e).__name__}: {e}"
        ) from e


def resolve_batch_size(settings: Any) -> int:
    """Resolve inference batch size from DLKit workflow settings."""
    data_cfg = getattr(settings, "data", None)
    configured = getattr(data_cfg, "batch_size", None) if data_cfg else None
    try:
        return int(configured) if configured is not None else 256
    except TypeError, ValueError:
        return 256
