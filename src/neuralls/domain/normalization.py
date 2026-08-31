"""Normalization utilities for the domain layer.

Canonicalized on matrix-based (Gershgorin-bound) normalization only. Provides
a small, composable API: an ABC interface for the scale strategy, a frozen
dataclass implementation, and pure factory functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from neuralls.domain.linalg import compute_dim_scale, compute_spectral_bound

# =============================================================================
# ABC Interfaces
# =============================================================================


class IScale(ABC):
    """Interface for scaling strategies.

    Encapsulates all scaling logic for a normalization strategy.
    """

    @abstractmethod
    def scale_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Scale the system matrix A."""
        ...

    @abstractmethod
    def scale_rhs(self, rhs: np.ndarray) -> np.ndarray:
        """Scale a right-hand side vector b."""
        ...

    @abstractmethod
    def to_dict(self) -> Mapping[str, float | list[float]]:
        """Serialize scale parameters to dictionary.

        Returns:
            Mapping of scale parameters suitable for saving to dataset manifests.
        """
        ...


# =============================================================================
# Frozen Dataclasses: Scaling Strategy
# =============================================================================


@dataclass(frozen=True)
class MatrixScale(IScale):
    """Matrix normalization: scale by spectral radius bound * sqrt(d).

    Transformation summary:
        A' = A / (spectral_radius_bound * sqrt(d))
        b' = b / (spectral_radius_bound * sqrt(d))

    The normalized system A' @ x = b' has the same solution as A @ x = b.
    """

    spectral_radius_bound: float
    dimension_scale: float

    @property
    def composite_scale(self) -> float:
        """Combined scale factor."""
        return self.spectral_radius_bound * self.dimension_scale

    def scale_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return matrix / self.composite_scale

    def scale_rhs(self, rhs: np.ndarray) -> np.ndarray:
        return rhs / self.composite_scale

    def to_dict(self) -> dict[str, float]:
        """Serialize scale parameters to dictionary."""
        return {
            "spectral_radius_bound": self.spectral_radius_bound,
            "dimension_scale": self.dimension_scale,
        }


# =============================================================================
# Frozen Dataclasses: Trace Samples
# =============================================================================


@dataclass(frozen=True)
class ResidualTraceSamples:
    """Residual → solution trace pairs from CG iterations.

    Captures residuals r_k and corresponding solutions x_k at iteration k.
    Training mapping: N(r_k) → x_k
    """

    residuals: np.ndarray
    solutions: np.ndarray
    sample_indices: np.ndarray
    iteration_indices: np.ndarray
    search_directions: np.ndarray | None = None
    search_direction_products: np.ndarray | None = None


@dataclass(frozen=True)
class ErrorTraceSamples:
    """Error correction traces: r_k → (x* - x_k).

    Training mapping: N(r_k) → (x* - x_k) where r_k is the network input.
    solutions_current (x_k) is stored for error computation but NOT as input.
    """

    residuals: np.ndarray
    solutions_current: np.ndarray  # x_k (for error computation only)
    errors: np.ndarray  # x* - x_k (training targets)
    true_solutions: np.ndarray  # x* per sample
    sample_indices: np.ndarray
    iteration_indices: np.ndarray


# =============================================================================
# Pure Functions: Scale Factories
# =============================================================================


def _create_matrix_scale(
    matrix: np.ndarray,
    spectral_radius_bound: float | None = None,
    **_kwargs: Any,
) -> MatrixScale:
    """Create matrix normalization scale.

    Args:
        matrix: System matrix A
        spectral_radius_bound: Optional spectral radius bound (computed if None)

    Returns:
        MatrixScale object
    """
    dimension = matrix.shape[0]
    if spectral_radius_bound is None:
        spectral_radius_bound = compute_spectral_bound(matrix)
    dimension_scale = compute_dim_scale(dimension)
    return MatrixScale(
        spectral_radius_bound=spectral_radius_bound,
        dimension_scale=dimension_scale,
    )


def load_scale_from_metadata(
    normalize_type: str,
    metadata: dict,
) -> IScale | None:
    """Reconstruct scale object from saved metadata.

    Args:
        normalize_type: Type of normalization ("none" or "matrix")
        metadata: Dictionary containing scale parameters

    Returns:
        Reconstructed scale object, or None if normalize_type is "none"

    Raises:
        ValueError: If normalize_type is invalid or required parameters are missing
    """
    if normalize_type == "none":
        return None

    if normalize_type == "matrix":
        if "spectral_radius_bound" not in metadata or "dimension_scale" not in metadata:
            raise ValueError(
                "Matrix normalization requires 'spectral_radius_bound' and 'dimension_scale' in metadata"
            )
        return MatrixScale(
            spectral_radius_bound=metadata["spectral_radius_bound"],
            dimension_scale=metadata["dimension_scale"],
        )

    raise ValueError(f"Invalid normalize_type: {normalize_type}. Must be one of: none, matrix")


# =============================================================================
# Pure Functions: Config-based Scale Creation
# =============================================================================


_SCALE_CREATORS = {
    "matrix": _create_matrix_scale,
}


def create_scale_from_config(
    normalize_type: Literal["none", "matrix"],
    matrix: np.ndarray,
    *,
    spectral_radius_bound: float | None = None,
) -> IScale | None:
    """Create normalization scale object from configuration.

    Args:
        normalize_type: Type of normalization ("none" or "matrix")
        matrix: System matrix A
        spectral_radius_bound: For matrix normalization. If None, computed from matrix.

    Returns:
        Scale object, or None (for "none" type).

    Raises:
        ValueError: If normalize_type is invalid.

    Examples:
        >>> scale = create_scale_from_config("matrix", A)
        >>> isinstance(scale, MatrixScale)
        True
    """
    if normalize_type == "none":
        return None

    creator = _SCALE_CREATORS.get(normalize_type)
    if creator is None:
        raise ValueError(f"Invalid normalize_type: {normalize_type}. Must be one of: none, matrix")

    return creator(matrix=matrix, spectral_radius_bound=spectral_radius_bound)


__all__ = [
    "ErrorTraceSamples",
    # Interfaces
    "IScale",
    # Scale
    "MatrixScale",
    # Traces
    "ResidualTraceSamples",
    # Public API
    "create_scale_from_config",
    "load_scale_from_metadata",
]
