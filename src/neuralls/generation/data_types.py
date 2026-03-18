"""Data types for generation framework."""

from __future__ import annotations

from typing import Literal, Required, TypedDict


# Type for normalization methods
NormalizeType = Literal["none", "matrix", "rhs", "spectral", "diagonal"]


class ScaleMetadata(TypedDict, total=False):
    """Type-safe schema for scale metadata dictionary.

    Stores scaling information applied to matrices during generation.

    Attributes:
        spectral_radius_bound: Required. Upper bound on spectral radius used
                               in spectral normalization.
        dimension_scale: Required. Scaling factor derived from matrix dimension.
        rhs_norm: Optional. Norm of right-hand side if computed.
    """

    spectral_radius_bound: Required[float]
    dimension_scale: Required[float]
    rhs_norm: float
