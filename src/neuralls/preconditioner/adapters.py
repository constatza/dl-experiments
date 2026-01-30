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
from loguru import logger

from neuralls.preconditioner.ports import PredictorPort, PredictorAdapter
from neuralls.preconditioner.tensor_utils import (
    prepare_model_input,
    extract_model_output,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DLKitPredictor(PredictorPort):
    """DLKit-based predictor with lifecycle management.

    Implements PredictorPort using DLKit framework. Manages model
    lifecycle: load → apply (many times) → cleanup.

    Resource Management:
        - Model loaded to GPU in __init__
        - Multiple apply() calls reuse same model
        - cleanup() frees GPU memory
        - Context manager ensures cleanup

    Error Handling:
        - GPU OOM → RuntimeError with helpful message
        - Device mismatch → RuntimeError
        - Model failure → RuntimeError with context
    """

    def __init__(self, model: torch.nn.Module, device: str) -> None:
        """Initialize predictor with loaded model.

        Args:
            model: PyTorch model (already loaded, in eval mode)
            device: Device string ("cpu", "cuda", "mps")

        Note:
            Model should already be in eval mode and on correct device.
        """
        self._model = model
        self._device = device
        self._loaded = True

    def apply(self, residual: NDArray) -> NDArray:
        """Apply neural network to residual.

        Args:
            residual: Numpy residual vector

        Returns:
            Predicted correction (float64 numpy)

        Raises:
            RuntimeError: If predictor unloaded or GPU error
            ValueError: If residual shape incompatible
        """
        if not self._loaded:
            raise RuntimeError(
                "Predictor has been unloaded. Create a new predictor or "
                "ensure cleanup() is not called before inference completes."
            )

        try:
            # Prepare input (pure function)
            input_tensor = prepare_model_input(residual, self._device)

            # Forward pass (action - GPU computation)
            with torch.no_grad():
                output = self._model(input_tensor)

            # Extract output (pure function)
            result = extract_model_output(output)
            return result

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
        """Release GPU memory and model resources.

        Idempotent - safe to call multiple times.
        """
        if not self._loaded:
            return  # Already cleaned up

        try:
            # Move model to CPU to free GPU memory
            if self._device != "cpu":
                logger.debug(f"Moving model from {self._device} to CPU for cleanup")
                self._model.cpu()

            # Clear references
            self._model = None  # type: ignore[assignment]
            self._loaded = False

            # Force garbage collection
            import gc
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.debug("Predictor cleanup complete")

        except (RuntimeError, OSError, AttributeError) as e:
            logger.warning(f"Error during predictor cleanup: {e}")
            # Don't raise - cleanup should be best-effort


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
            ImportError: If DLKit not installed

        Example:
            >>> adapter = DLKitAdapter()
            >>> predictor = adapter.create_predictor(Path("model.ckpt"))
            >>> with predictor:
            ...     result = predictor.apply(residual)
        """
        # Validate checkpoint exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Ensure the path is correct and the file exists."
            )

        try:
            # Lazy import to avoid hard dependency
            from dlkit import load_model
            from dlkit.tools.config.precision.strategy import PrecisionStrategy

            logger.debug(f"Loading checkpoint from: {checkpoint_path}")

            # Load predictor from DLKit
            dlkit_predictor = load_model(
                checkpoint_path,
                device="auto",
                apply_transforms=False,  # Data pre-normalized
                precision=PrecisionStrategy.FULL_64,
            )

            # Extract model (uses public API, not private _model_state)
            model = dlkit_predictor.model
            if model is None:
                raise RuntimeError(
                    f"Failed to load model from checkpoint: {checkpoint_path}. "
                    "Checkpoint may be corrupted or incompatible."
                )

            # Get device through public property (avoiding _model_state access)
            # Use the model's device directly
            device_param = next(model.parameters(), None)
            if device_param is not None:
                device = str(device_param.device)
            else:
                # Fallback to CPU if no parameters
                device = "cpu"
                logger.warning("Model has no parameters, defaulting to CPU")

            # Ensure model is in eval mode
            model.eval()

            logger.info(
                f"Loaded model from {checkpoint_path} on device {device}"
            )

            # Return wrapped predictor
            return DLKitPredictor(model, device)

        except ImportError as e:
            raise ImportError(
                "DLKit not installed. Install with: pip install dlkit"
            ) from e

        except (FileNotFoundError, OSError, ValueError, RuntimeError) as e:
            raise RuntimeError(
                f"Failed to load model from {checkpoint_path}: {type(e).__name__}: {e}"
            ) from e
