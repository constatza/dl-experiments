"""Normalization utilities for the domain layer.

This module provides a clean, composable API for normalizing linear systems:
- ABC interfaces for polymorphic design
- Frozen dataclasses for immutability
- Pure functions for testability
- Tiny functions (3-5 lines) for clarity
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from neuralls.domain.linalg import (
    calculate_spectral_norm,
    compute_dim_scale,
    compute_rhs_norms,
    compute_spectral_bound,
    extract_diagonal,
    validate_diagonal,
)


# =============================================================================
# ABC Interfaces
# =============================================================================


class ILinearSystem(ABC):
    """Interface for a single linear system Ax = b.

    Abstracts storage details (dense, sparse, distributed) and enforces
    shape consistency through properties.
    """

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """System matrix A."""
        ...

    @property
    @abstractmethod
    def rhs(self) -> np.ndarray:
        """Right-hand side vector b."""
        ...

    @property
    @abstractmethod
    def solution(self) -> np.ndarray | None:
        """Solution vector x (may be None if unknown)."""
        ...

    @property
    def dimension(self) -> int:
        """System dimension (number of rows/columns in A)."""
        return self.matrix.shape[0]


class ILinearSystemBatch(ABC):
    """Interface for batch of linear systems sharing the same matrix.

    All systems share matrix A but have different RHS vectors b_i and
    solutions x_i.
    """

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """Shared system matrix A."""
        ...

    @property
    @abstractmethod
    def rhs_samples(self) -> np.ndarray:
        """Batch of RHS vectors, shape (N, n)."""
        ...

    @property
    @abstractmethod
    def sol_samples(self) -> np.ndarray:
        """Batch of solution vectors, shape (N, n)."""
        ...

    @property
    def num_samples(self) -> int:
        """Number of samples in the batch."""
        return len(self.rhs_samples)

    @property
    def dimension(self) -> int:
        """System dimension (number of rows/columns in A)."""
        return self.matrix.shape[0]


class IScale(ABC):
    """Interface for scaling strategies.

    Encapsulates all scaling logic for a normalization strategy. Each
    implementation defines how to scale matrix, RHS, solution, and residuals,
    and their inverse denormalization operations.
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
    def scale_solution(self, sol: np.ndarray) -> np.ndarray:
        """Scale a solution vector x."""
        ...

    @abstractmethod
    def scale_residual(self, residual: np.ndarray) -> np.ndarray:
        """Scale a residual vector r."""
        ...

    @abstractmethod
    def denormalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Convert normalized matrix back to raw space.

        Inverse of scale_matrix(). Given A' (normalized matrix),
        returns A (raw matrix).

        Args:
            matrix: Normalized matrix A'.

        Returns:
            Raw matrix A.
        """
        ...

    @abstractmethod
    def denormalize_rhs(self, rhs: np.ndarray) -> np.ndarray:
        """Convert normalized RHS back to raw space.

        Inverse of scale_rhs(). Given b' (normalized RHS),
        returns b (raw RHS).

        Args:
            rhs: Normalized RHS vector b'.

        Returns:
            Raw RHS vector b.
        """
        ...

    @abstractmethod
    def denormalize_solution(self, sol: np.ndarray) -> np.ndarray:
        """Convert normalized solution back to raw space.

        Inverse of scale_solution(). Given x' (solution to normalized system A' @ x' = b'),
        returns x (solution to raw system A @ x = b).

        Args:
            sol: Normalized solution vector x'.

        Returns:
            Raw solution vector x.
        """
        ...

    @abstractmethod
    def to_dict(self) -> Mapping[str, float | list[float]]:
        """Serialize scale parameters to dictionary.

        Returns:
            Mapping of scale parameters suitable for saving to dataset manifests.
        """
        ...


class ITraceSamples(ABC):
    """Base interface for all trace sample types.

    Traces capture intermediate states during iterative solver runs.
    Concrete implementations should have sample_indices and iteration_indices fields.
    """

    @property
    def num_traces(self) -> int:
        """Total number of trace points.

        Requires concrete class to have sample_indices attribute.
        """
        sample_indices = getattr(self, "sample_indices", None)
        if not isinstance(sample_indices, np.ndarray):
            raise TypeError("Trace samples must expose numpy sample_indices")
        return len(sample_indices)


# =============================================================================
# Frozen Dataclasses: Linear Systems
# =============================================================================


@dataclass(frozen=True)
class LinearSystem(ILinearSystem):
    """Immutable single linear system Ax = b."""

    _matrix: np.ndarray
    _rhs: np.ndarray
    _solution: np.ndarray | None = None

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def rhs(self) -> np.ndarray:
        return self._rhs

    @property
    def solution(self) -> np.ndarray | None:
        return self._solution


@dataclass(frozen=True)
class LinearSystemBatch(ILinearSystemBatch):
    """Immutable batch of linear systems with shared matrix."""

    _matrix: np.ndarray
    _rhs_samples: np.ndarray
    _sol_samples: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def rhs_samples(self) -> np.ndarray:
        return self._rhs_samples

    @property
    def sol_samples(self) -> np.ndarray:
        return self._sol_samples


# =============================================================================
# Frozen Dataclasses: Scaling Strategies
# =============================================================================


@dataclass(frozen=True)
class MatrixScale(IScale):
    """Matrix normalization: scale by spectral radius bound * sqrt(d).

    Transformation summary:
        Forward (normalization):
            A' = A / (spectral_radius_bound * sqrt(d))
            b' = b / (spectral_radius_bound * sqrt(d))
            x' = x  (unchanged)

        Inverse (denormalization):
            A = A' * (spectral_radius_bound * sqrt(d))
            b = b' * (spectral_radius_bound * sqrt(d))
            x = x'  (unchanged)

    Solutions are stored in raw space for numerical stability.
    The normalized system A' @ x' = b' has the same solution as A @ x = b.
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

    def scale_solution(self, sol: np.ndarray) -> np.ndarray:
        """Scale solution vector (identity for MatrixScale).

        For reference/validation only. Solutions are stored in raw space
        for numerical stability, as the normalized system A' @ x' = b'
        has the same solution as the raw system A @ x = b.
        """
        return sol  # Solution unchanged

    def scale_residual(self, residual: np.ndarray) -> np.ndarray:
        return residual / self.composite_scale

    def denormalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Convert normalized matrix back to raw space.

        Inverse of scale_matrix().

        Formula:
            A = A' * (spectral_radius_bound * sqrt(d))

        Args:
            matrix: Normalized matrix A'.

        Returns:
            Raw matrix A.
        """
        return matrix * self.composite_scale

    def denormalize_rhs(self, rhs: np.ndarray) -> np.ndarray:
        """Convert normalized RHS back to raw space.

        Inverse of scale_rhs().

        Formula:
            b = b' * (spectral_radius_bound * sqrt(d))

        Args:
            rhs: Normalized RHS vector b'.

        Returns:
            Raw RHS vector b.
        """
        return rhs * self.composite_scale

    def denormalize_solution(self, sol: np.ndarray) -> np.ndarray:
        """Convert normalized solution back to raw space.

        Inverse of scale_solution(). For MatrixScale, solutions are unchanged.

        Formula:
            x = x' (no transformation)

        Args:
            sol: Normalized solution vector x'.

        Returns:
            Raw solution vector x.
        """
        return sol  # Solution unchanged

    def to_dict(self) -> dict[str, float]:
        """Serialize scale parameters to dictionary."""
        return {
            "spectral_radius_bound": self.spectral_radius_bound,
            "dimension_scale": self.dimension_scale,
        }


@dataclass(frozen=True)
class DiagonalScale(IScale):
    """Symmetric diagonal normalization: D^(-1/2) A D^(-1/2).

    This is the proper diagonal normalization for data preprocessing.
    It preserves matrix symmetry and spectral properties.

    Transformation summary:
        Forward (normalization):
            A' = D^(-1/2) @ A @ D^(-1/2)  (symmetric diagonal scaling)
            b' = D^(-1/2) @ b
            x' = D^(1/2) @ x

        Inverse (denormalization):
            A = D^(1/2) @ A' @ D^(1/2)
            b = D^(1/2) @ b'
            x = D^(-1/2) @ x'

    where D = diag(A) is the diagonal matrix of A.

    Properties:
    - Preserves symmetry: if A = A^T, then A' = A'^T
    - Diagonal of A' has all ones: diag(A') = I
    - Suitable for data normalization and preprocessing

    NOTE: This is DIFFERENT from Jacobi preconditioning.
    For Jacobi preconditioner (row-only scaling), see preconditioner_factory.py.
    """

    diagonal_sqrt_inv: np.ndarray  # D^(-1/2) = 1 / sqrt(diag(A))

    @property
    def diagonal_sqrt(self) -> np.ndarray:
        """D^(1/2) for denormalization."""
        return 1.0 / self.diagonal_sqrt_inv

    def scale_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Scale matrix: A' = D^(-1/2) @ A @ D^(-1/2)."""
        # Apply D^(-1/2) to both rows and columns (symmetric scaling)
        return self.diagonal_sqrt_inv[:, None] * matrix * self.diagonal_sqrt_inv[None, :]

    def scale_rhs(self, rhs: np.ndarray) -> np.ndarray:
        """Scale RHS: b' = D^(-1/2) @ b."""
        return rhs * self.diagonal_sqrt_inv

    def scale_solution(self, sol: np.ndarray) -> np.ndarray:
        """Scale solution: x' = D^(1/2) @ x."""
        return sol * self.diagonal_sqrt

    def scale_residual(self, residual: np.ndarray) -> np.ndarray:
        """Scale residual: r' = D^(-1/2) @ r."""
        return residual * self.diagonal_sqrt_inv

    def denormalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Convert normalized matrix back to raw space.

        Inverse of scale_matrix().

        Formula:
            A = D^(1/2) @ A' @ D^(1/2)

        Args:
            matrix: Normalized matrix A'.

        Returns:
            Raw matrix A.
        """
        return self.diagonal_sqrt[:, None] * matrix * self.diagonal_sqrt[None, :]

    def denormalize_rhs(self, rhs: np.ndarray) -> np.ndarray:
        """Convert normalized RHS back to raw space.

        Inverse of scale_rhs().

        Formula:
            b = D^(1/2) @ b'

        Args:
            rhs: Normalized RHS vector b'.

        Returns:
            Raw RHS vector b.
        """
        return rhs * self.diagonal_sqrt

    def denormalize_solution(self, sol: np.ndarray) -> np.ndarray:
        """Convert normalized solution back to raw space.

        Inverse of scale_solution().

        Formula:
            x = D^(-1/2) @ x'

        Args:
            sol: Normalized solution vector x'.

        Returns:
            Raw solution vector x.
        """
        return sol * self.diagonal_sqrt_inv

    def to_dict(self) -> dict[str, list[float]]:
        """Serialize scale parameters to dictionary."""
        return {
            "diagonal_sqrt_inv": self.diagonal_sqrt_inv.tolist(),
        }


@dataclass(frozen=True)
class SpectralScale(IScale):
    """Spectral normalization: per-sample scaling by spectral norm and RHS norm.

    Transformation summary:
        Forward (normalization):
            A' = A / (spectral_norm * sqrt(d))
            b' = b / (||b|| * sqrt(d))
            x' = x * spectral_norm / ||b||  (solution scaling for normalized space)

        Inverse (denormalization):
            A = A' * (spectral_norm * sqrt(d))
            b = b' * (||b|| * sqrt(d))
            x = x' * (||b|| / spectral_norm)

    Note: Unlike MatrixScale and DiagonalScale, SpectralScale stores solutions
    in normalized space (x'). The normalized system A' @ x' = b' has a different
    solution representation than A @ x = b. Use denormalize_solution() to convert
    back to raw space.
    """

    spectral_norm: float
    dimension_scale: float
    rhs_norm: float  # Per-sample RHS norm

    def scale_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return matrix / (self.spectral_norm * self.dimension_scale)

    def scale_rhs(self, rhs: np.ndarray) -> np.ndarray:
        return rhs / (self.rhs_norm * self.dimension_scale)

    def scale_solution(self, sol: np.ndarray) -> np.ndarray:
        """Scale solution vector for normalized space.

        For SpectralScale, solutions are stored in normalized space to maintain
        consistency with the normalized system A' @ x' = b'. This differs from
        MatrixScale and DiagonalScale where solutions remain unchanged.

        Formula:
            x' = x * spectral_norm / ||b||
        """
        return sol * self.spectral_norm / self.rhs_norm

    def scale_residual(self, residual: np.ndarray) -> np.ndarray:
        return residual / (self.rhs_norm * self.dimension_scale)

    def denormalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Convert normalized matrix back to raw space.

        Inverse of scale_matrix().

        Formula:
            A = A' * (spectral_norm * sqrt(d))

        Args:
            matrix: Normalized matrix A'.

        Returns:
            Raw matrix A.
        """
        return matrix * (self.spectral_norm * self.dimension_scale)

    def denormalize_rhs(self, rhs: np.ndarray) -> np.ndarray:
        """Convert normalized RHS back to raw space.

        Inverse of scale_rhs().

        Formula:
            b = b' * (||b|| * sqrt(d))

        Args:
            rhs: Normalized RHS vector b'.

        Returns:
            Raw RHS vector b.
        """
        return rhs * (self.rhs_norm * self.dimension_scale)

    def denormalize_solution(self, sol: np.ndarray) -> np.ndarray:
        """Convert normalized solution back to raw space.

        Inverse of scale_solution(). For SpectralScale, this performs
        the inverse transformation to convert from normalized solution space.

        Formula:
            x = x' * (||b|| / spectral_norm)

        Args:
            sol: Normalized solution vector x'.

        Returns:
            Raw solution vector x.
        """
        return sol * self.rhs_norm / self.spectral_norm

    def to_dict(self) -> dict[str, float]:
        """Serialize scale parameters to dictionary."""
        return {
            "spectral_norm": self.spectral_norm,
            "dimension_scale": self.dimension_scale,
            "rhs_norm": self.rhs_norm,
        }


# =============================================================================
# Frozen Dataclasses: Trace Samples
# =============================================================================


@dataclass(frozen=True)
class ResidualTraceSamples(ITraceSamples):
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

    # ITraceSamples interface is automatically satisfied by dataclass fields


@dataclass(frozen=True)
class ErrorTraceSamples(ITraceSamples):
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

    # ITraceSamples interface is automatically satisfied by dataclass fields


@dataclass(frozen=True)
class SearchDirectionsSamples(ITraceSamples):
    """Search direction traces: (A @ p_k) → p_k for neural preconditioner training.

    Training mapping: NN(A @ p_k) ≈ p_k, meaning NN ≈ A^{-1}
    This learns the inverse operator without requiring exact solutions.

    Collected from CG iterations where:
    - p_k: search direction at iteration k
    - A @ p_k: search direction product (computed in CG anyway)
    """

    search_direction_products: np.ndarray  # A @ p_k (inputs)
    search_directions: np.ndarray  # p_k (targets)
    sample_indices: np.ndarray
    iteration_indices: np.ndarray

    # ITraceSamples interface is automatically satisfied by dataclass fields


# =============================================================================
# Frozen Dataclasses: Normalization Results
# =============================================================================


@dataclass(frozen=True)
class NormalizedSystem:
    """Result of applying normalization to a linear system batch.

    Contains the normalized system, scaling metadata, and optional normalized
    trace samples.
    """

    system: ILinearSystemBatch
    scale: IScale | None
    residual_traces: ResidualTraceSamples | None = None
    error_traces: ErrorTraceSamples | None = None


# =============================================================================
# Pure Functions: Scale Factories
# =============================================================================


def make_matrix_scale(system: ILinearSystemBatch) -> MatrixScale:
    """Create matrix normalization scale from system."""
    bound = compute_spectral_bound(system.matrix)
    dim_scale = compute_dim_scale(system.dimension)
    return MatrixScale(bound, dim_scale)


def make_diagonal_scale(system: ILinearSystemBatch) -> DiagonalScale:
    """Create symmetric diagonal normalization scale from system.

    This creates D^(-1/2) scaling, NOT Jacobi preconditioning.
    """
    diag = extract_diagonal(system.matrix)
    validate_diagonal(diag)
    return DiagonalScale(1.0 / np.sqrt(diag))


def make_spectral_scales(system: ILinearSystemBatch) -> list[SpectralScale]:
    """Create per-sample spectral normalization scales from system."""
    spec_norm = compute_spectral_bound(system.matrix)
    dim_scale = compute_dim_scale(system.dimension)
    rhs_norms = compute_rhs_norms(system.rhs_samples)
    return [SpectralScale(spec_norm, dim_scale, norm) for norm in rhs_norms]


def load_scale_from_metadata(
    normalize_type: str,
    metadata: dict,
) -> IScale | None:
    """Reconstruct scale object from saved metadata.

    Args:
        normalize_type: Type of normalization ("none", "matrix", "diagonal", "spectral")
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

    if normalize_type == "diagonal":
        # Support both new format (diagonal_sqrt_inv) and legacy format (diagonal_inv)
        if "diagonal_sqrt_inv" in metadata:
            return DiagonalScale(
                diagonal_sqrt_inv=np.array(metadata["diagonal_sqrt_inv"], dtype=np.float64)
            )
        elif "diagonal_inv" in metadata:
            # Legacy format: diagonal_inv was 1/diag(A), convert to diagonal_sqrt_inv
            diagonal_inv = np.array(metadata["diagonal_inv"], dtype=np.float64)
            diagonal = 1.0 / diagonal_inv
            return DiagonalScale(diagonal_sqrt_inv=1.0 / np.sqrt(diagonal))
        else:
            raise ValueError(
                "Diagonal normalization requires 'diagonal_sqrt_inv' or 'diagonal_inv' in metadata"
            )

    if normalize_type == "spectral":
        if (
            "spectral_norm" not in metadata
            or "dimension_scale" not in metadata
            or "rhs_norm" not in metadata
        ):
            raise ValueError(
                "Spectral normalization requires 'spectral_norm', 'dimension_scale', and 'rhs_norm' in metadata"
            )
        return SpectralScale(
            spectral_norm=metadata["spectral_norm"],
            dimension_scale=metadata["dimension_scale"],
            rhs_norm=metadata["rhs_norm"],
        )

    raise ValueError(
        f"Invalid normalize_type: {normalize_type}. Must be one of: none, matrix, diagonal, spectral"
    )


# =============================================================================
# Pure Functions: System Scaling
# =============================================================================


def scale_system(system: ILinearSystemBatch, scale: IScale) -> LinearSystemBatch:
    """Scale system using any IScale implementation (polymorphic)."""
    return LinearSystemBatch(
        _matrix=scale.scale_matrix(system.matrix),
        _rhs_samples=np.array([scale.scale_rhs(rhs) for rhs in system.rhs_samples]),
        _sol_samples=np.array([scale.scale_solution(sol) for sol in system.sol_samples]),
    )


def scale_system_spectral(
    system: ILinearSystemBatch, scales: list[SpectralScale]
) -> LinearSystemBatch:
    """Scale system with per-sample spectral scales.

    Each sample uses its own SpectralScale (different rhs_norm).
    """
    return LinearSystemBatch(
        _matrix=scales[0].scale_matrix(system.matrix),
        _rhs_samples=np.array(
            [scale.scale_rhs(rhs) for scale, rhs in zip(scales, system.rhs_samples)]
        ),
        _sol_samples=np.array(
            [scale.scale_solution(sol) for scale, sol in zip(scales, system.sol_samples)]
        ),
    )


# =============================================================================
# Pure Functions: Trace Scaling
# =============================================================================


def scale_residual_traces(traces: ResidualTraceSamples, scale: IScale) -> ResidualTraceSamples:
    """Scale residual traces using any IScale implementation."""
    search_directions = (
        None
        if traces.search_directions is None
        else np.array([scale.scale_solution(p) for p in traces.search_directions])
    )
    search_direction_products = (
        None
        if traces.search_direction_products is None
        else np.array([scale.scale_residual(ap) for ap in traces.search_direction_products])
    )
    return ResidualTraceSamples(
        residuals=np.array([scale.scale_residual(r) for r in traces.residuals]),
        solutions=np.array([scale.scale_solution(s) for s in traces.solutions]),
        sample_indices=traces.sample_indices,
        iteration_indices=traces.iteration_indices,
        search_directions=search_directions,
        search_direction_products=search_direction_products,
    )


def scale_residual_traces_spectral(
    traces: ResidualTraceSamples, scales: list[SpectralScale]
) -> ResidualTraceSamples:
    """Scale residual traces with per-sample spectral scales.

    Maps each trace to its corresponding sample's SpectralScale via sample_indices.
    """
    residuals_scaled = np.array(
        [
            scales[int(idx)].scale_residual(r)
            for idx, r in zip(traces.sample_indices, traces.residuals)
        ]
    )
    solutions_scaled = np.array(
        [
            scales[int(idx)].scale_solution(s)
            for idx, s in zip(traces.sample_indices, traces.solutions)
        ]
    )
    search_directions_scaled = (
        None
        if traces.search_directions is None
        else np.array(
            [
                scales[int(idx)].scale_solution(p)
                for idx, p in zip(traces.sample_indices, traces.search_directions)
            ]
        )
    )
    search_direction_products_scaled = (
        None
        if traces.search_direction_products is None
        else np.array(
            [
                scales[int(idx)].scale_residual(ap)
                for idx, ap in zip(traces.sample_indices, traces.search_direction_products)
            ]
        )
    )
    return ResidualTraceSamples(
        residuals=residuals_scaled,
        solutions=solutions_scaled,
        sample_indices=traces.sample_indices,
        iteration_indices=traces.iteration_indices,
        search_directions=search_directions_scaled,
        search_direction_products=search_direction_products_scaled,
    )


def scale_error_traces(traces: ErrorTraceSamples, scale: IScale) -> ErrorTraceSamples:
    """Scale error traces using any IScale implementation."""
    return ErrorTraceSamples(
        residuals=np.array([scale.scale_residual(r) for r in traces.residuals]),
        solutions_current=traces.solutions_current,  # Not scaled (for reference only)
        errors=np.array([scale.scale_residual(e) for e in traces.errors]),
        true_solutions=traces.true_solutions,  # Not scaled (for reference only)
        sample_indices=traces.sample_indices,
        iteration_indices=traces.iteration_indices,
    )


def scale_error_traces_spectral(
    traces: ErrorTraceSamples, scales: list[SpectralScale]
) -> ErrorTraceSamples:
    """Scale error traces with per-sample spectral scales.

    Maps each trace to its corresponding sample's SpectralScale via sample_indices.
    """
    residuals_scaled = np.array(
        [
            scales[int(idx)].scale_residual(r)
            for idx, r in zip(traces.sample_indices, traces.residuals)
        ]
    )
    errors_scaled = np.array(
        [scales[int(idx)].scale_residual(e) for idx, e in zip(traces.sample_indices, traces.errors)]
    )
    return ErrorTraceSamples(
        residuals=residuals_scaled,
        solutions_current=traces.solutions_current,
        errors=errors_scaled,
        true_solutions=traces.true_solutions,
        sample_indices=traces.sample_indices,
        iteration_indices=traces.iteration_indices,
    )


# =============================================================================
# Pure Functions: Normalization Strategies
# =============================================================================


def normalize_none(
    system: ILinearSystemBatch,
    residual_traces: ResidualTraceSamples | None = None,
    error_traces: ErrorTraceSamples | None = None,
) -> NormalizedSystem:
    """No-op normalization: returns defensive copies."""
    return NormalizedSystem(
        system=LinearSystemBatch(
            _matrix=system.matrix.copy(),
            _rhs_samples=system.rhs_samples.copy(),
            _sol_samples=system.sol_samples.copy(),
        ),
        scale=None,
        residual_traces=(
            ResidualTraceSamples(
                residuals=residual_traces.residuals.copy(),
                solutions=residual_traces.solutions.copy(),
                sample_indices=residual_traces.sample_indices.copy(),
                iteration_indices=residual_traces.iteration_indices.copy(),
                search_directions=(
                    None
                    if residual_traces.search_directions is None
                    else residual_traces.search_directions.copy()
                ),
                search_direction_products=(
                    None
                    if residual_traces.search_direction_products is None
                    else residual_traces.search_direction_products.copy()
                ),
            )
            if residual_traces
            else None
        ),
        error_traces=(
            ErrorTraceSamples(
                residuals=error_traces.residuals.copy(),
                solutions_current=error_traces.solutions_current.copy(),
                errors=error_traces.errors.copy(),
                true_solutions=error_traces.true_solutions.copy(),
                sample_indices=error_traces.sample_indices.copy(),
                iteration_indices=error_traces.iteration_indices.copy(),
            )
            if error_traces
            else None
        ),
    )


def normalize_matrix(
    system: ILinearSystemBatch,
    residual_traces: ResidualTraceSamples | None = None,
    error_traces: ErrorTraceSamples | None = None,
) -> NormalizedSystem:
    """Matrix normalization: scale by spectral radius bound * sqrt(d)."""
    scale = make_matrix_scale(system)
    normalized = scale_system(system, scale)
    res = scale_residual_traces(residual_traces, scale) if residual_traces else None
    err = scale_error_traces(error_traces, scale) if error_traces else None
    return NormalizedSystem(normalized, scale, res, err)


def normalize_diagonal(
    system: ILinearSystemBatch,
    residual_traces: ResidualTraceSamples | None = None,
    error_traces: ErrorTraceSamples | None = None,
) -> NormalizedSystem:
    """Diagonal (Jacobi) normalization: scale by inverse diagonal."""
    scale = make_diagonal_scale(system)
    normalized = scale_system(system, scale)
    res = scale_residual_traces(residual_traces, scale) if residual_traces else None
    err = scale_error_traces(error_traces, scale) if error_traces else None
    return NormalizedSystem(normalized, scale, res, err)


def normalize_spectral(
    system: ILinearSystemBatch,
    residual_traces: ResidualTraceSamples | None = None,
    error_traces: ErrorTraceSamples | None = None,
) -> NormalizedSystem:
    """Spectral normalization: per-sample scaling by spectral norm and RHS norm."""
    scales = make_spectral_scales(system)
    normalized = scale_system_spectral(system, scales)
    res = scale_residual_traces_spectral(residual_traces, scales) if residual_traces else None
    err = scale_error_traces_spectral(error_traces, scales) if error_traces else None
    # Return first scale as metadata (all share spectral_norm and dimension_scale)
    return NormalizedSystem(normalized, scales[0] if scales else None, res, err)


# =============================================================================
# Impure Functions: Logging
# =============================================================================


def log_normalization_stats(strategy: str, result: NormalizedSystem) -> None:
    """Log normalization statistics (SIDE EFFECT: prints to stdout)."""
    print(f"Applied {strategy} normalization")

    if result.scale is None:
        print("  No scaling applied")
        return

    if isinstance(result.scale, MatrixScale):
        print(f"  Spectral radius bound: {result.scale.spectral_radius_bound:.6e}")
        print(f"  Dimension scale: {result.scale.dimension_scale:.6f}")
        print(f"  Composite scale: {result.scale.composite_scale:.6e}")

    elif isinstance(result.scale, DiagonalScale):
        print("  Diagonal scaling (Jacobi)")
        # Compute spectral radius of normalized matrix for diagnostics
        spec_bound = compute_spectral_bound(result.system.matrix)
        print(f"  Normalized spectral radius bound: {spec_bound:.6e}")

    elif isinstance(result.scale, SpectralScale):
        print(f"  Spectral norm: {result.scale.spectral_norm:.6e}")
        print(f"  Dimension scale: {result.scale.dimension_scale:.6f}")
        print(f"  Per-sample RHS norm (example): {result.scale.rhs_norm:.6e}")


# =============================================================================
# Pure Functions: Config-based Scale Creation
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


def _create_diagonal_scale(
    matrix: np.ndarray,
    **_kwargs: Any,
) -> DiagonalScale:
    """Create diagonal (Jacobi) normalization scale.

    Args:
        matrix: System matrix A

    Returns:
        DiagonalScale object
    """
    diagonal = extract_diagonal(matrix)
    validate_diagonal(diagonal)
    diagonal_sqrt_inv = 1.0 / np.sqrt(diagonal)
    return DiagonalScale(diagonal_sqrt_inv=diagonal_sqrt_inv)


def _create_spectral_scale(
    matrix: np.ndarray,
    rhs_samples: np.ndarray | None = None,
    solution_samples: np.ndarray | None = None,
    **_kwargs: Any,
) -> list[SpectralScale]:
    """Create spectral normalization scales (one per sample).

    Args:
        matrix: System matrix A
        rhs_samples: Optional RHS samples (shape: n_samples x dimension)
        solution_samples: Optional solution samples (shape: n_samples x dimension)

    Returns:
        List of SpectralScale objects (one per sample)

    Raises:
        ValueError: If both or neither samples provided
    """
    # Guard: Validate mutually exclusive parameters
    if rhs_samples is not None and solution_samples is not None:
        raise ValueError(
            "Cannot provide both rhs_samples and solution_samples for spectral normalization. "
            "Provide only one."
        )

    if rhs_samples is None and solution_samples is None:
        raise ValueError("Spectral normalization requires either rhs_samples or solution_samples")

    dimension = matrix.shape[0]
    spectral_norm = calculate_spectral_norm(matrix)
    dimension_scale = compute_dim_scale(dimension)

    # Case A: RHS archive provided
    if rhs_samples is not None:
        rhs_norms = compute_rhs_norms(rhs_samples)
        return [
            SpectralScale(
                spectral_norm=spectral_norm,
                dimension_scale=dimension_scale,
                rhs_norm=rhs_norm,
            )
            for rhs_norm in rhs_norms
        ]

    # Case B: Solution archive provided (compute RHS as b = A @ x)
    # Type guard ensures solution_samples is not None here
    assert solution_samples is not None
    scales = []
    for solution in solution_samples:
        # Compute what RHS would be: b = A @ x
        rhs = matrix @ solution
        rhs_norm = float(np.linalg.norm(rhs, ord=2))
        scales.append(
            SpectralScale(
                spectral_norm=spectral_norm,
                dimension_scale=dimension_scale,
                rhs_norm=rhs_norm,
            )
        )
    return scales


# Simple registry - no elaborate framework, just dict dispatch
_SCALE_CREATORS: dict[str, Callable[..., IScale | Sequence[IScale]]] = {
    "matrix": _create_matrix_scale,
    "diagonal": _create_diagonal_scale,
    "spectral": _create_spectral_scale,
}


def create_scale_from_config(
    normalize_type: Literal["none", "matrix", "spectral", "diagonal"],
    matrix: np.ndarray,
    *,
    spectral_radius_bound: float | None = None,
    rhs_samples: np.ndarray | None = None,
    solution_samples: np.ndarray | None = None,
) -> IScale | list[IScale] | None:
    """Create normalization scale object from configuration.

    Pure function that creates IScale objects based on normalization type
    and required parameters. Uses dispatch pattern for clean extensibility.

    Args:
        normalize_type: Type of normalization ("none", "matrix", "spectral", "diagonal")
        matrix: System matrix A
        spectral_radius_bound: For matrix normalization. If None, computed from matrix.
        rhs_samples: For spectral normalization (RHS archive case). Shape (n_samples, dimension).
        solution_samples: For spectral normalization (solution archive case). Shape (n_samples, dimension).

    Returns:
        Scale object, list of scale objects (for spectral with per-sample scales),
        or None (for "none" type).

    Raises:
        ValueError: If required parameters missing for normalization type or if both
            rhs_samples and solution_samples are provided for spectral normalization.

    Notes:
        - For "none": Returns None (no-op)
        - For "matrix": Uses spectral_radius_bound * sqrt(d)
        - For "diagonal": Computes from matrix diagonal
        - For "spectral": Creates per-sample scales based on RHS or computed RHS norms

    Examples:
        >>> # Matrix normalization
        >>> scale = create_scale_from_config("matrix", A)
        >>> isinstance(scale, MatrixScale)
        True

        >>> # Diagonal normalization
        >>> scale = create_scale_from_config("diagonal", A)
        >>> isinstance(scale, DiagonalScale)
        True

        >>> # Spectral normalization with RHS samples
        >>> scales = create_scale_from_config("spectral", A, rhs_samples=R)
        >>> isinstance(scales, list)
        True
    """
    # Guard: Handle "none" early
    if normalize_type == "none":
        return None

    # Guard: Validate normalize_type
    creator = _SCALE_CREATORS.get(normalize_type)
    if creator is None:
        raise ValueError(
            f"Invalid normalize_type: {normalize_type}. "
            f"Must be one of: none, matrix, spectral, diagonal"
        )

    # Dispatch to appropriate creator
    result = creator(
        matrix=matrix,
        spectral_radius_bound=spectral_radius_bound,
        rhs_samples=rhs_samples,
        solution_samples=solution_samples,
    )
    # Convert Sequence to list for return type compatibility
    if isinstance(result, list):
        return cast(list[IScale], result)
    if isinstance(result, Sequence) and not isinstance(result, (IScale, list)):
        return list(result)
    return cast(IScale | None, result)


# =============================================================================
# Public API
# =============================================================================


def apply_normalization(
    normalize: Literal["none", "matrix", "spectral", "diagonal"],
    A_original: np.ndarray,
    R: np.ndarray,
    X: np.ndarray,
    dataset_dir: Path,
    residual_traces: ResidualTraceSamples | None = None,
    error_traces: ErrorTraceSamples | None = None,
    verbose: bool = True,
) -> NormalizationResult:
    """Apply normalization strategy with optional logging.

    This is the main public API. All normalization logic is pure and testable;
    logging is isolated here as a side effect.

    Args:
        normalize: Normalization strategy ("none", "matrix", "spectral", "diagonal")
        A_original: System matrix
        R: Batch of RHS vectors, shape (N, n)
        X: Batch of solution vectors, shape (N, n)
        dataset_dir: Dataset directory (for compatibility; unused in current implementation)
        residual_traces: Optional residual trace samples to normalize
        error_traces: Optional error trace samples to normalize
        verbose: If True, log normalization statistics

    Returns:
        NormalizedSystem containing normalized data and metadata

    Raises:
        ValueError: If normalize is not a valid strategy
    """
    del dataset_dir  # Unused in current implementation

    # Create system batch
    system = LinearSystemBatch(_matrix=A_original, _rhs_samples=R, _sol_samples=X)

    # Map strategy name to function
    strategies = {
        "none": normalize_none,
        "matrix": normalize_matrix,
        "diagonal": normalize_diagonal,
        "spectral": normalize_spectral,
    }

    if normalize not in strategies:
        raise ValueError(
            f"Invalid normalize value: {normalize}. Must be one of: {list(strategies.keys())}"
        )

    # Pure computation
    result = strategies[normalize](system, residual_traces, error_traces)

    # Side effect (optional)
    if verbose:
        log_normalization_stats(normalize, result)

    # Convert to legacy format for backward compatibility
    return NormalizationResult.from_normalized_system(result, normalize)


# =============================================================================
# Compatibility Layer (for backward compatibility with existing code)
# =============================================================================


@dataclass
class NormalizationResult:
    """Backward-compatible result format.

    Provides the same interface as the old NormalizationResult for existing
    callers. New code should use NormalizedSystem directly.
    """

    matrix: np.ndarray
    rhs_samples: np.ndarray
    sol_samples: np.ndarray
    normalize_type: str = "none"
    matrix_scale: float = 1.0
    spectral_radius_bound: float | None = None
    spectral_norm: float | None = None
    residual_traces: ResidualTraceSamples | None = None
    error_traces: ErrorTraceSamples | None = None
    scale: IScale | None = None  # Scale object for generic metadata persistence

    @staticmethod
    def from_normalized_system(
        result: NormalizedSystem, normalize_type: str
    ) -> NormalizationResult:
        """Convert NormalizedSystem to legacy NormalizationResult format.

        Args:
            result: The NormalizedSystem to convert
            normalize_type: The normalization type applied ("none", "matrix", "diagonal", "spectral")
        """
        # Extract scale metadata
        matrix_scale = 1.0
        spectral_radius_bound = None
        spectral_norm = None

        if isinstance(result.scale, MatrixScale):
            matrix_scale = result.scale.spectral_radius_bound
            spectral_radius_bound = result.scale.spectral_radius_bound
        elif isinstance(result.scale, SpectralScale):
            spectral_norm = result.scale.spectral_norm

        return NormalizationResult(
            matrix=result.system.matrix,
            rhs_samples=result.system.rhs_samples,
            sol_samples=result.system.sol_samples,
            normalize_type=normalize_type,
            matrix_scale=matrix_scale,
            spectral_radius_bound=spectral_radius_bound,
            spectral_norm=spectral_norm,
            residual_traces=result.residual_traces,
            error_traces=result.error_traces,
            scale=result.scale,  # Preserve scale object for generic persistence
        )


__all__ = [
    # Interfaces
    "ILinearSystem",
    "ILinearSystemBatch",
    "IScale",
    "ITraceSamples",
    # Linear Systems
    "LinearSystem",
    "LinearSystemBatch",
    # Scales
    "MatrixScale",
    "DiagonalScale",
    "SpectralScale",
    # Traces
    "ResidualTraceSamples",
    "ErrorTraceSamples",
    # Results
    "NormalizedSystem",
    "NormalizationResult",  # Legacy compatibility
    # Public API
    "apply_normalization",
    "create_scale_from_config",
    # Pure functions (for advanced users)
    "make_matrix_scale",
    "make_diagonal_scale",
    "make_spectral_scales",
    "scale_system",
    "scale_system_spectral",
    "scale_residual_traces",
    "scale_error_traces",
    "normalize_none",
    "normalize_matrix",
    "normalize_diagonal",
    "normalize_spectral",
    "load_scale_from_metadata",
]
