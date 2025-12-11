"""Dataset construction from arrays.

This module provides functions to build dlkit dataset structures (Features and
Targets) from numpy arrays, and inject them into GeneralSettings objects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from dlkit import GeneralSettings
from dlkit.tools.config.data_entries import ValueFeature, ValueTarget
from dlkit.tools.config.core.updater import update_settings


def create_features_from_array(
    array: np.ndarray,
    name: str = "rhs",
) -> tuple[Any, ...]:
    """Create Feature tuple from numpy array.

    Args:
        array: Input numpy array with feature data.
        name: Name of the feature.

    Returns:
        Tuple containing single Feature created from array.
    """
    return (ValueFeature(name=name, value=array),)


def create_matrix_feature(array: np.ndarray) -> Any:
    """Create matrix Feature from numpy array.

    Args:
        array: Input numpy array with matrix data.

    Returns:
        Feature object for matrix data.
    """
    return ValueFeature(name="matrix", value=array)


def create_targets_from_array(
    array: np.ndarray,
    name: str = "solutions",
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
    settings: GeneralSettings,
    rhs: np.ndarray,
    solutions: np.ndarray,
    matrix: np.ndarray | None = None,
) -> Any:
    """Inject dataset arrays into GeneralSettings.

    Creates Features from RHS (and matrix if provided) and Targets from
    solutions, then injects them into settings via update_settings().

    For datasets named "GraphDataset", includes matrix in features.

    Args:
        settings: GeneralSettings object to update.
        rhs: RHS feature array.
        solutions: Target array (solutions).
        matrix: Optional matrix array (included for GraphDataset).

    Returns:
        Updated settings object with injected dataset arrays.
    """
    # Create features from RHS
    features = list(create_features_from_array(rhs, name="rhs"))

    # Include matrix if provided and dataset is GraphDataset
    dataset_name = getattr(settings.DATASET, "name", None) if hasattr(
        settings, "DATASET"
    ) else None

    if matrix is not None and dataset_name == "GraphDataset":
        features.append(create_matrix_feature(matrix))

    # Create targets from solutions
    targets = create_targets_from_array(solutions, name="solutions")

    # Build DATASET dict
    dataset_dict = {
        "features": tuple(features),
        "targets": targets,
    }

    # Inject via update_settings
    return update_settings(settings, {"DATASET": dataset_dict})
