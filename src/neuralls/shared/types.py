"""Shared domain types used across multiple neuralls subpackages.

Types placed here satisfy two criteria:
- They carry no domain logic (pure data / TypedDict / StrEnum)
- They are imported by more than one top-level subpackage

Moving cross-boundary types here prevents upward dependencies, e.g. io → generation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, TypedDict

DatasetFormat = Literal["zarr", "npy"]
"""Supported dataset storage families for generated datasets."""

EntryRole = Literal["feature", "target"]
"""Runtime dataset entry roles understood by neuralls bridge code."""


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
