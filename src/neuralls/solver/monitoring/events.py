"""Type-safe event types for solver iteration logging.

This module defines EventType enum to replace magic string-based event names
with a type-safe enumeration. This eliminates typos, provides IDE autocomplete,
and enables compile-time validation.

Design Principles:
    - Single Responsibility: Only defines event type constants
    - Type Safety: Enum prevents invalid event names
    - Self-Documenting: Each event type has clear docstring
    - Extensibility: Easy to add new event types

Usage:
    >>> from neuralls.solver.monitoring.events import EventType
    >>> logger.log(EventType.RESIDUAL_NORM, 1.5)  # Type-safe!
    >>> logger.log("residual_norm", 1.5)  # Old way - avoided
"""

from __future__ import annotations

from enum import StrEnum


class EventType(StrEnum):
    """Type-safe event types for solver iteration logging.

    Events are categorized into scalars (single values) and vectors (arrays).
    The is_scalar/is_vector properties enable type checking at runtime.

    Scalar Events (float values):
        ITERATION: Iteration counter (k)
        RESIDUAL_NORM: Absolute residual norm ||r_k||_2
        RESIDUAL_NORM_REL: Relative residual norm ||r_k||/||b||
        STEP_LENGTH: CG step length α_k
        CURVATURE: Direction curvature (p_k, A p_k) for breakdown detection
        BETA_COEFFICIENT: PCG recurrence coefficient β_k

    Vector Events (NDArray):
        RESIDUAL: Residual vector r_k = b - A x_k
        SOLUTION: Solution vector x_k
        DIRECTION: Search direction p_k
        PRECOND_RESIDUAL: Preconditioned residual z_k = M^{-1} r_k
        MATRIX_PRODUCT: Matrix-vector product q_k = A p_k

    Boolean Events (stored as 0.0 or 1.0):
        CONVERGED: Convergence criterion satisfied
        BREAKDOWN: Numerical breakdown detected (NaN/Inf)
        DIVERGENCE: Divergence detected (||r|| growing)
        REORTHOGONALIZED: Reorthogonalization applied this iteration

    Mathematical Context:
        Krylov methods maintain working vectors and scalars at each iteration k:
        - x_k: Current solution estimate
        - r_k: Residual r_k = b - A x_k
        - p_k: Search direction
        - q_k: Matrix-vector product q_k = A p_k
        - α_k: Step length
        - β_k: Recurrence coefficient (method-specific)

    Memory Usage:
        - Scalars: ~8 bytes per value per iteration
        - Vectors: ~8n bytes per vector per iteration (n = problem size)
        - Booleans: 8 bytes (stored as float)

        Example (n=10000, 100 iterations):
        - Logging 5 scalars: ~4 KB
        - Logging 3 vectors: ~24 MB

    References:
        - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems.
        - Golub & Van Loan (2013). Matrix Computations.
    """

    # Scalar events - numerical values at each iteration
    ITERATION = "iteration"
    """Iteration number k (0-indexed). Always an integer cast to float."""

    RESIDUAL_NORM = "residual_norm"
    """Absolute residual norm ||r_k||_2."""

    RESIDUAL_NORM_REL = "residual_norm_rel"
    """Relative residual norm ||r_k||/||b||."""

    STEP_LENGTH = "step_length"
    """CG step length α_k = (r_k, z_k) / (p_k, A p_k)."""

    CURVATURE = "curvature"
    """Direction curvature (p_k, A p_k) for breakdown detection."""

    BETA_COEFFICIENT = "beta_coefficient"
    """PCG recurrence coefficient β_k for direction update."""

    # Vector events - full arrays at each iteration
    RESIDUAL = "residual"
    """Residual vector r_k = b - A x_k (shape: n)."""

    SOLUTION = "solution"
    """Solution vector x_k (shape: n)."""

    DIRECTION = "direction"
    """Search direction p_k (shape: n)."""

    PRECOND_RESIDUAL = "precond_residual"
    """Preconditioned residual z_k = M^{-1} r_k (shape: n)."""

    MATRIX_PRODUCT = "matrix_product"
    """Matrix-vector product q_k = A p_k (shape: n)."""

    # Boolean events - flags stored as 0.0 or 1.0
    CONVERGED = "converged"
    """Convergence flag: True if ||r_k|| <= max(rtol*||b||, atol)."""

    BREAKDOWN = "breakdown"
    """Breakdown flag: True if NaN/Inf detected or curvature <= 0."""

    DIVERGENCE = "divergence"
    """Divergence flag: True if ||r_k|| > divergence_threshold * ||r_0||."""

    REORTHOGONALIZED = "reorthogonalized"
    """Reorthogonalization flag: True if corrective orthogonalization applied."""

    @property
    def is_scalar(self) -> bool:
        """Check if event type stores scalar (float) values.

        Returns:
            True if event stores scalars, False if vectors.

        Examples:
            >>> EventType.ITERATION.is_scalar
            True
            >>> EventType.RESIDUAL.is_scalar
            False
        """
        return self in {
            EventType.ITERATION,
            EventType.RESIDUAL_NORM,
            EventType.RESIDUAL_NORM_REL,
            EventType.STEP_LENGTH,
            EventType.CURVATURE,
            EventType.BETA_COEFFICIENT,
            EventType.CONVERGED,
            EventType.BREAKDOWN,
            EventType.DIVERGENCE,
            EventType.REORTHOGONALIZED,
        }

    @property
    def is_vector(self) -> bool:
        """Check if event type stores vector (NDArray) values.

        Returns:
            True if event stores vectors, False if scalars.

        Examples:
            >>> EventType.RESIDUAL.is_vector
            True
            >>> EventType.ITERATION.is_vector
            False
        """
        return not self.is_scalar
