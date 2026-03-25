"""Data types for generation framework."""

from __future__ import annotations

from typing import Literal, TypedDict


# Type for normalization methods
NormalizeType = Literal["none", "matrix", "rhs", "spectral", "diagonal"]


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
