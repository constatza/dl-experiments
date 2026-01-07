"""Inference workflow module.

This module provides a clean, functional approach to running model inference
with single-responsibility functions and typed domain models.

Key Components:
    - Domain models for configuration and data (config.py)
    - Data loading strategies (data_loading.py)
    - Prediction execution (prediction.py)
    - Output saving and MLflow logging (output.py)
"""

from neuralls.workflows.inference.config import (
    InferenceConfig,
    InferenceData,
    InferenceOutputs,
    InferencePredictions,
)
from neuralls.workflows.inference.data_loading import (
    load_inference_data,
    load_standard_data,
    load_synthetic_data,
)
from neuralls.workflows.inference.prediction import (
    create_predictor,
    run_prediction,
)
from neuralls.workflows.inference.output import (
    save_inference_outputs,
    start_mlflow_run,
    finalize_mlflow_run,
    save_synthetic_predictions,
)

__all__ = [
    # Domain models
    "InferenceConfig",
    "InferenceData",
    "InferencePredictions",
    "InferenceOutputs",
    # Data loading
    "load_inference_data",
    "load_standard_data",
    "load_synthetic_data",
    # Prediction
    "create_predictor",
    "run_prediction",
    # Output
    "save_inference_outputs",
    "start_mlflow_run",
    "finalize_mlflow_run",
    "save_synthetic_predictions",
]
