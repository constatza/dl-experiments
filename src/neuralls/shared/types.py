"""Shared domain types used across multiple neuralls subpackages.

Types placed here satisfy two criteria:
- They carry no domain logic (pure data / TypedDict / StrEnum)
- They are imported by more than one top-level subpackage

Moving cross-boundary types here prevents upward dependencies, e.g. io → generation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, TypedDict

DatasetFormat = Literal["zarr", "npy", "hdf5"]
"""Supported dataset storage families for generated datasets."""

EntryRole = Literal["feature", "target"]
"""Runtime dataset entry roles understood by neuralls bridge code."""


class LayoutType(StrEnum):
    """Physical matrix layout in a generated dataset.

    Attributes:
        MANY_MATRICES: Each logical sample has its own physical matrix row.
        BROADCAST_SINGLE: One physical matrix row shared across all logical samples.
    """

    MANY_MATRICES = "many_matrices"
    BROADCAST_SINGLE = "broadcast_single"


class MatrixNormType(StrEnum):
    """Matrix norm types for dataset metadata.

    Single source of truth for all supported matrix norms.
    To add a new norm:
    1. Add member here
    2. Add calculate_<name>_norm() function in domain/linalg.py
    3. Add entry to the _NORM_STRATEGIES mapping in domain/linalg.py
    """

    SPECTRAL = "spectral"  # Spectral norm (2-norm): largest singular value
    FROBENIUS = "frobenius"  # Frobenius norm: sqrt(sum of squared entries)
    NUCLEAR = "nuclear"  # Nuclear norm: sum of singular values
    ONE = "one"  # 1-norm: max column sum
    INF = "inf"  # Infinity norm: max row sum


class RhsKind(StrEnum):
    """Semantic category of a persisted RHS row."""

    UNKNOWN = "unknown"
    NON_RESIDUAL = "non_residual"
    RESIDUAL = "residual"
    SEARCH_DIRECTION_PRODUCT = "search_direction_product"


class TargetKind(StrEnum):
    """Semantic category of a persisted target row."""

    UNKNOWN = "unknown"
    SOLUTION = "solution"
    ITERATE = "iterate"
    ERROR = "error"
    SEARCH_DIRECTION = "search_direction"


class ComparisonRhsGenerationKind(StrEnum):
    """Supported direct RHS generators for comparison workflows."""

    GAUSSIAN = "gaussian"
    SPARSE = "sparse"


class GenerationStrategyKind(StrEnum):
    """Canonical generation strategy identifiers used after config validation."""

    RANDOM = "random"
    NORMAL = "normal"
    KRYLOV = "krylov"
    RHS_ARCHIVE = "rhs_archive"
    SOLUTION_ARCHIVE = "solution_archive"
    VALIDATED_ARCHIVE = "validated_archive"
    SCALED_SOLUTIONS = "scaled_solutions"
    SPARSE_RHS = "sparse_rhs"
    RESIDUAL_TRACES = "residual_traces"
    RESIDUALS = "residuals"
    GAUSSIAN_RESIDUALS = "gaussian_residuals"
    SEARCH_DIRECTIONS = "search_directions"
    EIGENVECTOR_FORWARD = "eigenvector_forward"
    EIGENVECTOR_INVERSE = "eigenvector_inverse"
    GAUSSIAN_FORWARD = "gaussian_forward"
    GAUSSIAN_INVERSE = "gaussian_inverse"
    UNIFORM_FORWARD = "uniform_forward"
    UNIFORM_INVERSE = "uniform_inverse"
    CONSTANT_FORWARD = "constant_forward"
    CONSTANT_INVERSE = "constant_inverse"
    NEUTRAL_ONES = "neutral_ones"


class ScaleMetadata(TypedDict, total=False):
    """Type-safe schema for scale metadata dictionary.

    Stores scaling information applied to matrices during generation.

    Attributes:
        spectral_radius_bound: Upper bound on spectral radius used
            in matrix or spectral normalization.
        dimension_scale: Scaling factor derived from matrix dimension.
        rhs_norm: Optional. Norm of right-hand side if computed.
        diagonal_sqrt_inv: Optional diagonal normalization payload.
    """

    spectral_radius_bound: float
    dimension_scale: float
    rhs_norm: float
    diagonal_sqrt_inv: list[float]
