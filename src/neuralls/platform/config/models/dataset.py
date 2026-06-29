"""Dataset construction helpers for DLKit runtime settings."""

from __future__ import annotations

from typing import Any

import numpy as np
from dlkit.infrastructure.config.core.patching import patch_model
from dlkit.infrastructure.config.data_entries import DataRole, ValueEntry


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
    return (ValueEntry(name=name, value=array),)


def create_matrix_feature(
    array: np.ndarray,
    name: str = "matrix",
) -> Any:
    """Create matrix Feature from numpy array.

    Matrix entries are context tensors and must not be forwarded to model.forward().
    The model_input flag is set to False.

    Args:
        array: Input numpy array with matrix data.
        name: Name of the matrix feature (default: "matrix").

    Returns:
        Feature object for matrix data with model_input=False.
    """
    return ValueEntry(name=name, value=array, model_input=False)


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
    return (ValueEntry(name=name, value=array, data_role=DataRole.TARGET),)


def with_dataset_arrays(
    settings: Any,
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
