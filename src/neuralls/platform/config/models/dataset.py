"""Dataset construction helpers for DLKit runtime settings."""

from __future__ import annotations

from typing import Any

import numpy as np
from dlkit.infrastructure.config.core.patching import patch_model
from dlkit.infrastructure.config.data_entries import ValueFeature, ValueTarget
from dlkit.infrastructure.config.workflow_types import WorkflowConfig


def create_features_from_array(
    array: np.ndarray,
    name: str,
) -> tuple[Any, ...]:
    """Create Feature tuple from numpy array.

    Args:
        array: Input numpy array with feature data.
        name: Name of the feature.

    Returns:
        Tuple containing single Feature created from array.
    """
    return (ValueFeature(name=name, value=array),)


def create_matrix_feature(
    array: np.ndarray,
    name: str = "matrix",
) -> Any:
    """Create matrix Feature from numpy array.

    Args:
        array: Input numpy array with matrix data.

    Returns:
        Feature object for matrix data.
    """
    return ValueFeature(name=name, value=array)


def create_targets_from_array(
    array: np.ndarray,
    name: str,
) -> tuple[Any, ...]:
    """Create Target tuple from numpy array.

    Args:
        array: Input numpy array with target data.
        name: Name of the target.

    Returns:
        Tuple containing single Target created from array.
    """
    return (ValueTarget(name=name, value=array),)


def with_dataset_arrays(
    settings: WorkflowConfig,
    rhs: np.ndarray,
    solutions: np.ndarray,
    matrix: np.ndarray | None = None,
    *,
    primary_input_name: str,
    target_name: str,
    matrix_input_name: str = "matrix",
) -> Any:
    """Inject dataset arrays into DLKit workflow settings.

    Creates runtime features from RHS (and matrix if provided) and maps the
    solutions artifact into the caller-supplied supervised target entry before
    injecting everything via ``patch_model()``.

    For datasets named "GraphDataset", includes matrix in features.

    Args:
        settings: DLKit workflow settings object to update.
        rhs: RHS feature array.
        solutions: Target array (solutions).
        matrix: Optional matrix array (included for GraphDataset).

    Returns:
        Updated settings object with injected dataset arrays.
    """
    features = list(create_features_from_array(rhs, name=primary_input_name))

    # Include matrix if provided and dataset is GraphDataset
    dataset_name = getattr(settings.DATASET, "name", None) if hasattr(settings, "DATASET") else None

    if matrix is not None and dataset_name == "GraphDataset":
        features.append(create_matrix_feature(matrix, name=matrix_input_name))

    targets = create_targets_from_array(solutions, name=target_name)

    # Build DATASET dict
    dataset_dict = {
        "features": tuple(features),
        "targets": targets,
    }

    # Inject via patch_model
    return patch_model(settings, {"DATASET": dataset_dict})
