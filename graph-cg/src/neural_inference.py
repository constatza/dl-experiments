"""Neural inference utilities that delegate to DLKit's inference API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from dlkit.interfaces.api import predict_with_config
from dlkit import GeneralSettings
try:
    from dlkit.interfaces.inference.api import infer_with_config
    from dlkit.interfaces.inference import InferenceConfig
except ImportError:
    # Fallback if the API has changed
    from dlkit import infer


def create_neural_preconditioner(
    checkpoint_path: str | Path,
    config_path: str | Path = None,  # No longer needed but kept for API compatibility
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a warm-start predictor for real-time inference without datasets.

    Uses dlkit.interfaces.api.infer() for direct tensor-to-tensor prediction
    without requiring training config or datasets - only the checkpoint.

    Args:
        checkpoint_path: Path to the trained model checkpoint
        config_path: (Deprecated) No longer needed - kept for compatibility
    """
    checkpoint = _ensure_checkpoint(Path(checkpoint_path))
    predictor = _NeuralWarmStart(checkpoint)
    return predictor


class _NeuralWarmStart:
    """Wrap DLKit inference to predict a solution for an RHS vector."""

    def __init__(self, checkpoint: Path) -> None:
        self._checkpoint = checkpoint

    def __call__(self, rhs: np.ndarray) -> np.ndarray:
        """Predict solution vector from RHS vector using the neural model."""
        try:
            vector = np.asarray(rhs)
            rhs_dtype = vector.dtype if vector.dtype != np.dtype("O") else np.float32

            # For FFNN model, provide raw tensor input
            # The model expects input with shape (24,) based on checkpoint metadata
            input_tensor = torch.from_numpy(vector).float()
            # Ensure tensor matches expected shape (24,) from the checkpoint metadata
            if input_tensor.shape != (24,):
                input_tensor = input_tensor.reshape(24)

            # Try different inference approaches
            inputs = {"x": input_tensor}  # Use dict format as per documentation

            # Try the newer infer_with_config API first
            try:
                config = InferenceConfig(
                    model_checkpoint_path=str(self._checkpoint),
                    batch_size=1,
                    apply_transforms=False
                )
                result = infer_with_config(config, inputs)
            except (NameError, AttributeError) as api_error:
                # Fallback to original infer API
                from dlkit import infer
                try:
                    result = infer(
                        checkpoint_path=str(self._checkpoint),
                        inputs=inputs,
                        batch_size=1
                    )
                except Exception as infer_error:
                    raise RuntimeError(f"Both inference APIs failed. Config API error: {api_error}. Infer API error: {infer_error}") from infer_error

            if result is None:
                raise RuntimeError("Inference returned None result")

            # Debug: log the result structure before processing
            result_type = type(result).__name__
            result_attrs = dir(result) if hasattr(result, '__dict__') else []

            if not hasattr(result, 'predictions'):
                raise RuntimeError(f"Inference result missing predictions attribute. Result type: {result_type}, attributes: {result_attrs}")

            # Debug: log predictions structure
            pred_type = type(result.predictions).__name__
            pred_info = ""
            if isinstance(result.predictions, dict):
                pred_info = f"dict keys: {list(result.predictions.keys())}"
            elif hasattr(result.predictions, 'shape'):
                pred_info = f"shape: {result.predictions.shape}"
            else:
                pred_info = f"value: {str(result.predictions)[:200]}"

            try:
                prediction = _extract_prediction_array(result.predictions, vector.shape[0])
            except Exception as extract_error:
                raise RuntimeError(
                    f"Prediction extraction failed. Result type: {result_type}, "
                    f"predictions type: {pred_type}, predictions info: {pred_info}. "
                    f"Extraction error: {extract_error}"
                ) from extract_error
            return prediction.astype(rhs_dtype, copy=False)

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

            error_msg += f". Checkpoint: {self._checkpoint}, Input shape: {rhs.shape}, Input dtype: {rhs.dtype}"
            raise RuntimeError(error_msg) from e


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


def prepare_neural_input(x: np.ndarray) -> dict[str, torch.Tensor]:
    """Prepare input tensor for backwards compatibility with older tooling."""

    x_tensor = torch.from_numpy(np.asarray(x)).float().unsqueeze(0)
    return {"x": x_tensor}
