"""Configuration management for graph-cg experiments.

This module provides a simplified, functional approach to loading and managing
configurations for model training, data generation, and solver parameters.

Public API:
    load_config: Load model, data, and solver configurations
    get_solver_params: Extract solver parameters from settings
    SolverParams: Immutable dataclass for solver parameters
    with_dataset_arrays: Inject in-memory arrays into dataset configuration
"""

from .loader import load_config, load_data_context
from .solver import (
    SolverParams,
    extract_solver_params_from_config,
    get_solver_params,
    create_solver_from_params,
)
from .dataset import (
    create_features_from_array,
    create_matrix_feature,
    create_targets_from_array,
    with_dataset_arrays,
)

__all__ = [
    # Configuration loading
    "load_config",
    "load_data_context",
    # Solver parameters
    "SolverParams",
    "extract_solver_params_from_config",
    "get_solver_params",
    "create_solver_from_params",
    # Dataset building
    "create_features_from_array",
    "create_matrix_feature",
    "create_targets_from_array",
    "with_dataset_arrays",
]
