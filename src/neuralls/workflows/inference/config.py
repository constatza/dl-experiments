"""Domain models for inference workflow.

This module defines typed data structures for the inference workflow,
following the single responsibility principle and functional design patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for inference workflow.

    Bundles all inference parameters into a single, immutable configuration.
    This eliminates the need for 12 separate function parameters.

    Args:
        config_path: Path to model configuration file
        data_config_path: Path to data configuration file (optional)
        checkpoint_path: Explicit checkpoint path (optional)
        features_path: Path to features file (optional)
        targets_path: Path to targets file (optional)
        save_plots: Whether to generate diagnostic plots
        figures_dir: Custom figures directory (optional)
        enable_mlflow: Whether to log to MLflow
        output_root: Custom output root (optional)
        synthetic_benchmark: Whether to use synthetic benchmark data
        solver_config_path: Path to solver configuration (required for synthetic)
    """

    config_path: Path
    data_config_path: Path | None = None
    checkpoint_path: Path | None = None
    features_path: Path | None = None
    targets_path: Path | None = None
    save_plots: bool = True
    figures_dir: Path | None = None
    enable_mlflow: bool = False
    output_root: Path | None = None
    synthetic_benchmark: bool = False
    solver_config_path: Path | None = None


@dataclass(frozen=True)
class InferenceData:
    """Loaded data for inference.

    Represents the input data for running predictions, either from
    standard files or synthetic generation.

    Args:
        features: Feature arrays keyed by name (e.g., "rhs", "matrix")
        targets: Target arrays keyed by name (optional for synthetic)
        metadata: Additional metadata about the data
    """

    features: dict[str, np.ndarray]
    targets: dict[str, np.ndarray] | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class InferencePredictions:
    """Prediction results from model inference.

    Represents the output of running the predictor on input data.

    Args:
        predictions: Predicted arrays keyed by name
        targets: True target arrays for comparison
        metadata: Additional metadata about predictions
    """

    predictions: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class InferenceOutputs:
    """Saved outputs from inference workflow.

    Represents the artifacts created during inference execution.

    Args:
        csv_paths: Paths to saved CSV prediction files
        plot_paths: Paths to saved diagnostic plots
        metrics: Computed metrics (e.g., errors, norms)
    """

    csv_paths: list[Path]
    plot_paths: list[Path]
    metrics: dict[str, float]
