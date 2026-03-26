"""Shared domain types used across multiple neuralls subpackages.

Types placed here satisfy two criteria:
- They carry no domain logic (pure data / TypedDict / StrEnum)
- They are imported by more than one top-level subpackage

Moving cross-boundary types here prevents upward dependencies, e.g. io → generation.
"""

from __future__ import annotations

from typing import TypedDict


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
