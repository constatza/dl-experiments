"""Neural inference utilities that delegate to DLKit's inference API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from loguru import logger

from dlkit import infer

# Imports removed: normalization metadata no longer needed
# All normalization happens at data generation time


def create_neural_preconditioner(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
) -> Callable[[np.ndarray | None, np.ndarray], np.ndarray]:
    """Create a warm-start predictor for real-time inference without datasets.

    Uses dlkit.interfaces.api.infer() for direct tensor-to-tensor prediction
    without requiring training config or datasets - only the checkpoint.

    All normalization is handled at data generation time. The saved artifacts
    (matrix.npy, rhs-samples.npy, sol-samples.npy) are already normalized.

    This unified interface supports both:
    - Standard neural models (FFNN): Use only RHS input, ignore matrix
    - Graph neural models (GNN): Use both matrix and RHS inputs

    Args:
        checkpoint_path: Path to the trained model checkpoint
        config_path: Optional path to training config (unused, kept for API compatibility)
        data_config_path: Optional path to data config (unused, kept for API compatibility)

    Returns:
        Callable that accepts (matrix, rhs) and returns predicted solution.
        For RHS-only models, matrix parameter is ignored.
    """
    checkpoint = _ensure_checkpoint(Path(checkpoint_path))
    predictor = _NeuralWarmStart(checkpoint)
    return predictor


def create_neural_step_helper(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
) -> Callable[[Any], np.ndarray | None]:
    """Create a neural helper that produces per-iteration solution guesses.

    Args:
        checkpoint_path: Path to trained neural model checkpoint
        config_path: Optional config path (unused, kept for symmetry)
        data_config_path: Optional data config path (unused)

    Returns:
        Callable accepting an iteration context or residual vector and returning
        a helper correction (float64 numpy array) or None.
    """

    _ = config_path  # Placeholder for API compatibility
    _ = data_config_path
    checkpoint = _ensure_checkpoint(Path(checkpoint_path))
    predictor = _NeuralWarmStart(checkpoint)
    return _NeuralStepHelper(predictor)


class _NeuralWarmStart:
    """Unified neural warm-start for both standard and graph models.

    Supports:
    - Standard models (FFNN): Accept only RHS, ignore matrix
    - Graph models (GNN): Accept both matrix and RHS

    Assumes all data (training and inference) is already normalized at data generation time.
    No runtime normalization is performed.
    """

    def __init__(self, checkpoint: Path) -> None:
        self._checkpoint = checkpoint

    def __call__(self, matrix: np.ndarray | None, rhs: np.ndarray) -> np.ndarray:
        """Predict solution from matrix (optional) and RHS using the neural model.

        Args:
            matrix: System matrix A (optional - used by graph models, ignored by FFNN)
            rhs: RHS vector b (always used)

        Returns:
            Predicted solution vector

        Note:
            The model introspects what inputs it needs. Standard FFNNs ignore the matrix,
            while GNNs use both matrix and RHS.
        """
        try:
            rhs_vector = np.asarray(rhs)
            if rhs_vector.dtype == np.dtype("O"):
                rhs_vector = rhs_vector.astype(np.float32)
            flat_rhs = rhs_vector.reshape(-1)
            target_size = flat_rhs.shape[0]

            prediction: np.ndarray | None = None

            if matrix is not None:
                try:
                    prediction = self._run_inference(matrix, flat_rhs, target_size)
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Neural inference with matrix input failed; falling back to RHS-only. "
                        f"Error: {exc}"
                    )
                    prediction = None

            if prediction is None:
                prediction = self._run_inference(None, flat_rhs, target_size)

            return prediction.astype(np.float64, copy=False)

        except Exception as e:
            # Provide specific error context for neural warm start failures
            error_msg = f"Neural inference failed"
            if "Could not infer dtype" in str(e):
                error_msg += f" - dtype inference error, likely due to malformed model output"
            elif "predictions" in str(e).lower():
                error_msg += f" - predictions format error"
            elif "checkpoint" in str(e).lower():
                error_msg += f" - checkpoint loading error"
            else:
                error_msg += f" - {type(e).__name__}: {str(e)}"

            error_msg += f". Checkpoint: {self._checkpoint}, RHS shape: {rhs.shape}, RHS dtype: {rhs.dtype}"
            if matrix is not None:
                error_msg += f", Matrix shape: {matrix.shape}"
            raise RuntimeError(error_msg) from e

    def _run_inference(
        self,
        matrix: np.ndarray | None,
        rhs: np.ndarray,
        target_size: int,
    ) -> np.ndarray:
        """Execute inference call and post-process predictions."""
        inputs = self._prepare_inputs(matrix, rhs)
        result = infer(
            checkpoint_path=str(self._checkpoint),
            inputs=inputs,
            batch_size=1,
        )
        if result is None or not hasattr(result, "predictions"):
            raise RuntimeError("Inference returned no predictions")

        prediction = _extract_prediction_array(result.predictions, target_size)

        if prediction.size != target_size:
            # Fallback to RHS-only if matrix resulted in unexpected shape
            if matrix is not None:
                raise ValueError(
                    f"Neural inference produced {prediction.size} outputs (expected {target_size}). "
                    "This checkpoint likely ignores matrix inputs."
                )
            raise ValueError(
                f"Neural inference produced {prediction.size} outputs (expected {target_size})."
            )

        return prediction.reshape(-1)

    def _prepare_inputs(
        self,
        matrix: np.ndarray | None,
        rhs: np.ndarray
    ) -> dict[str, torch.Tensor]:
        """Prepare inputs for dlkit inference, supporting both FFNN and GNN models.

        Strategy:
        1. Try with both matrix and RHS (for GNN models)
        2. If that fails, fallback to RHS-only (for FFNN models)
        3. Cache the successful approach for future calls

        Args:
            matrix: System matrix A (optional)
            rhs: RHS vector b (required)

        Returns:
            Dictionary of input tensors for dlkit.infer()

        Note:
            dlkit models will error if passed unexpected inputs, so we must
            detect what the model expects and only provide those inputs.
        """
        # Convert RHS to tensor (always needed)
        rhs_tensor = torch.from_numpy(rhs.astype(np.float32, copy=False)).unsqueeze(0)

        # Try with matrix first if provided (for graph models)
        if matrix is not None:
            matrix_arr = np.asarray(matrix, dtype=np.float32)
            if matrix_arr.ndim != 2:
                raise ValueError(f"Matrix must be 2D, got shape {matrix_arr.shape}")
            matrix_tensor = torch.from_numpy(matrix_arr).unsqueeze(0)

            # Return both matrix and RHS for graph models
            return {
                "matrix": matrix_tensor,
                "rhs": rhs_tensor,
            }
        else:
            # Only RHS for standard models
            return {"rhs": rhs_tensor}


class _NeuralStepHelper:
    """Neural helper that maps residual context to solution corrections."""

    def __init__(self, predictor: _NeuralWarmStart) -> None:
        self._predictor = predictor

    def __call__(self, context: Any) -> np.ndarray:
        """Return helper correction for given iteration context or residual."""

        if isinstance(context, np.ndarray):
            residual = context
            matrix = None
        else:
            residual = getattr(context, "residual", None)
            if residual is None:
                raise ValueError("Iteration context missing 'residual' attribute")
            matrix = getattr(context, "matrix", None)

        residual_arr = np.asarray(residual, dtype=np.float64, copy=False)
        prediction = self._predictor(matrix, residual_arr)
        return prediction.astype(np.float64, copy=False)


def _ensure_checkpoint(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _extract_prediction_array(predictions: Any, target_size: int) -> np.ndarray:
    """Extract prediction array from dlkit inference result."""
    try:
        # Handle different prediction formats from new dlkit inference API
        if isinstance(predictions, torch.Tensor):
            array = predictions.detach().cpu().numpy()
        elif isinstance(predictions, np.ndarray):
            array = predictions
        elif isinstance(predictions, dict):
            if not predictions:
                raise ValueError("Predictions dict is empty")

            # Try common prediction keys
            array = None
            for key in ["predictions", "y", "output", "logits"]:
                if key in predictions:
                    pred = predictions[key]
                    if isinstance(pred, torch.Tensor):
                        array = pred.detach().cpu().numpy()
                    else:
                        array = np.asarray(pred, dtype=np.float32)
                    break
            else:
                # Take first available value
                first_key = next(iter(predictions.keys()))
                first_val = predictions[first_key]
                if isinstance(first_val, torch.Tensor):
                    array = first_val.detach().cpu().numpy()
                else:
                    array = np.asarray(first_val, dtype=np.float32)

        else:
            array = np.asarray(predictions, dtype=np.float32)

        # Ensure proper shape (remove batch dimension if present)
        array = array.squeeze()
        if array.ndim == 0:
            array = array.reshape(1)

        # Resize if needed to match target size
        if array.shape[0] != target_size:
            array = np.resize(array, target_size)

        return array

    except Exception as e:
        # Provide detailed error context for prediction extraction failures
        pred_type = type(predictions).__name__
        pred_info = ""

        if isinstance(predictions, dict):
            keys = list(predictions.keys())
            pred_info = f"dict with keys: {keys}"
            if keys:
                first_val_type = type(predictions[keys[0]]).__name__
                pred_info += f", first value type: {first_val_type}"
        elif hasattr(predictions, 'shape'):
            pred_info = f"{pred_type} with shape: {predictions.shape}"
        else:
            pred_info = f"{pred_type}: {str(predictions)[:100]}"

        raise ValueError(
            f"Failed to extract prediction array from {pred_info}. "
            f"Target size: {target_size}. Original error: {str(e)}"
        ) from e


def prepare_neural_input(
    rhs: np.ndarray,
    matrix: np.ndarray | None = None
) -> dict[str, torch.Tensor]:
    """Prepare input tensors for backwards compatibility with older tooling.

    Supports both standard models (RHS only) and graph models (matrix + RHS).

    Args:
        rhs: RHS vector (required)
        matrix: System matrix (optional, for graph models)

    Returns:
        Dictionary of input tensors

    Note:
        Feature names must match checkpoint training metadata.
        Standard checkpoints use "rhs" as the feature name.
        Graph checkpoints use "matrix" and "rhs".
    """
    rhs_tensor = torch.from_numpy(np.asarray(rhs)).float().unsqueeze(0)
    inputs = {"rhs": rhs_tensor}

    if matrix is not None:
        matrix_tensor = torch.from_numpy(np.asarray(matrix)).float().unsqueeze(0)
        inputs["matrix"] = matrix_tensor

    return inputs
