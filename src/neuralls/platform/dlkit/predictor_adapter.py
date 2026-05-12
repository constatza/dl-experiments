"""Framework-specific adapters for neural predictors.

This module implements adapters that convert framework-specific APIs
(DLKit, PyTorch) to the framework-agnostic PredictorPort interface.

Design Principles:
    - Adapter pattern: Framework-specific code isolated here
    - Resource management: Context managers ensure GPU cleanup
    - Error boundaries: Catch framework errors, raise domain errors
    - Separation: Actions (I/O, GPU) separated from domain logic
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from dlkit import load_model
from dlkit.infrastructure.precision.strategy import PrecisionStrategy
from loguru import logger

from neuralls.domain.solver.preconditioners.ports import PredictorAdapter, PredictorPort
from neuralls.domain.solver.preconditioners.tensor_utils import (
    extract_model_output,
    prepare_model_input,
)

from ._prediction_outputs import extract_prediction_tensor

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from dlkit.interfaces.inference import CheckpointPredictor


class DLKitPredictor(PredictorPort):
    """DLKit-based predictor with lifecycle management.

    Implements PredictorPort using DLKit framework. Wraps CheckpointPredictor
    and delegates lifecycle management to it via context manager.

    Lifecycle:
        Use as a context manager — `__exit__` delegates to CheckpointPredictor,
        which calls `unload()` automatically. Do NOT call `cleanup()` directly.

    Error Handling:
        - GPU OOM → RuntimeError with helpful message
        - Device mismatch → RuntimeError
        - Model failure → RuntimeError with context
    """

    def __init__(self, predictor: CheckpointPredictor, device: str) -> None:
        """Initialize predictor with loaded CheckpointPredictor.

        Args:
            predictor: DLKit CheckpointPredictor (already loaded, eval mode)
            device: Device string ("cpu", "cuda", "mps")
        """
        self._predictor: CheckpointPredictor = predictor
        self._device: str = device

    def apply(self, residual: NDArray) -> NDArray:
        """Apply neural network to residual.

        Args:
            residual: Numpy residual vector

        Returns:
            Predicted correction (float64 numpy)

        Raises:
            RuntimeError: If predictor unloaded or GPU error
        """
        try:
            input_tensor = prepare_model_input(residual, self._device)
            output = self._predictor.predict(input_tensor)
            primary = extract_prediction_tensor(output)
            return extract_model_output(primary)

        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                f"GPU out of memory during inference on device {self._device}. "
                f"Residual shape: {residual.shape}. "
                "Try reducing batch size or using CPU."
            ) from e

        except RuntimeError as e:
            if "device" in str(e).lower():
                raise RuntimeError(
                    f"Device mismatch: model on {self._device}, "
                    f"but tensor operation failed. Error: {e}"
                ) from e
            raise RuntimeError(
                f"Neural inference failed: {e}. "
                f"Model device: {self._device}, Residual shape: {residual.shape}"
            ) from e

        except (TypeError, AttributeError, ValueError) as e:
            raise RuntimeError(
                f"Unexpected error during neural inference: {type(e).__name__}: {e}"
            ) from e

    def cleanup(self) -> None:
        """No-op — use as context manager; __exit__ handles unload."""
        pass

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Delegate lifecycle to CheckpointPredictor's context manager."""
        self._predictor.__exit__(exc_type, exc_val, exc_tb)


class DLKitAdapter(PredictorAdapter):
    """Adapter for DLKit framework.

    Converts DLKit's API to framework-agnostic PredictorPort interface.
    Handles model loading, device management, and resource cleanup.

    Example:
        >>> adapter = DLKitAdapter()
        >>> with adapter.create_predictor(Path("model.ckpt")) as pred:
        ...     result = pred.apply(residual)
        ...     # Automatic GPU cleanup on exit
    """

    def create_predictor(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> PredictorPort:
        """Create predictor from DLKit checkpoint.

        Args:
            checkpoint_path: Path to DLKit checkpoint
            config_path: Unused (kept for API compatibility)
            data_config_path: Unused (kept for API compatibility)

        Returns:
            DLKitPredictor instance (use as context manager)

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If model loading fails
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Ensure the path is correct and the file exists."
            )

        try:
            logger.debug(f"Loading checkpoint from: {checkpoint_path}")
            dlkit_predictor = load_model(
                checkpoint_path,
                device="auto",
                apply_transforms=False,
                precision=PrecisionStrategy.FULL_64,
            )

            model = dlkit_predictor.model
            if model is None:
                raise RuntimeError(
                    f"Failed to load model from checkpoint: {checkpoint_path}. "
                    "Checkpoint may be corrupted or incompatible."
                )

            device_param = next(model.parameters(), None)
            device = str(device_param.device) if device_param is not None else "cpu"
            if device_param is None:
                logger.warning("Model has no parameters, defaulting to CPU")

            logger.info(f"Loaded model from {checkpoint_path} on device {device}")
            return DLKitPredictor(dlkit_predictor, device)

        except (FileNotFoundError, OSError, ValueError, RuntimeError) as e:
            raise RuntimeError(
                f"Failed to load model from {checkpoint_path}: {type(e).__name__}: {e}"
            ) from e
