"""Structural type protocols for solver state fields.

This module defines Protocol classes for type-safe access to state fields without
tight coupling to the state hierarchy. Protocols enable structural typing (duck
typing with type checking) for state-specific attributes.

Design Principles:
    - Dependency Inversion: Depend on protocols (interfaces), not concrete classes
    - Structural Typing: Type safety without inheritance constraints
    - Single Responsibility: Each protocol defines one structural contract
    - Open/Closed: Easy to add new protocols without modifying existing code

Why Protocols Over Inheritance:
    1. No variance issues - isinstance() works at runtime
    2. No type: ignore comments needed
    3. Multiple protocols can be satisfied by same class
    4. Loose coupling - implementation can change freely

Theory:
    Protocols implement structural subtyping (PEP 544). A class satisfies a
    Protocol if it has the required attributes/methods, regardless of inheritance.
    This is "duck typing" with compile-time type checking.

Usage:
    >>> from neuralls.solver.models.protocols import HasVectors
    >>>
    >>> def process_state(state: SolverState) -> None:
    >>>     if isinstance(state, HasVectors):
    >>>         # Type checker knows state.x, state.r exist
    >>>         residual = state.r  # ✅ No type: ignore needed
    >>>     else:
    >>>         # Handle base state without vectors
    >>>         pass

References:
    - PEP 544: Protocols (Structural Subtyping)
    - Liskov Substitution Principle (SOLID)
    - Interface Segregation Principle (SOLID)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .history import DirectionHistory, ResidualHistory


@runtime_checkable
class HasVectors(Protocol):
    """Protocol for states with Krylov iteration vectors.

    Classes satisfying this protocol have the five working vectors maintained
    during Krylov subspace iteration: solution, residual, preconditioned
    residual, search direction, and matrix-vector product.

    Satisfied By:
        - KrylovState
        - CGState (inherits from KrylovState)

    Mathematical Context:
        Krylov methods maintain these vectors at each iteration k:
        - x_k: Current solution estimate
        - r_k: Residual r_k = b - A x_k
        - z_k: Preconditioned residual z_k = M^{-1} r_k
        - p_k: Search direction (A-conjugate to previous directions)
        - q_k: Matrix-vector product q_k = A p_k

    Usage:
        >>> def extract_residual(state: SolverState) -> NDArray | None:
        >>>     if isinstance(state, HasVectors):
        >>>         return state.r  # ✅ Type-safe access
        >>>     return None

    Theory:
        The Krylov subspace K_k(A, r_0) = span{r_0, A*r_0, ..., A^(k-1)*r_0}
        is implicitly represented by these vectors. The search direction p_k
        is constructed to be A-conjugate to all previous directions.

    References:
        - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems.
        - Greenbaum, A. (1997). Iterative Methods for Solving Linear Systems.
    """

    x: NDArray
    """Solution vector x_k at current iteration."""

    r: NDArray
    """Residual vector r_k = b - A x_k."""

    z: NDArray
    """Preconditioned residual z_k = M^{-1} r_k."""

    p: NDArray
    """Search direction p_k (A-conjugate to previous directions)."""

    q: NDArray
    """Matrix-vector product q_k = A p_k."""


@runtime_checkable
class HasDirectionHistory(Protocol):
    """Protocol for states with CG direction and residual history tracking.

    Classes satisfying this protocol maintain sliding window histories of
    search directions and residuals for orthogonalization and diagnostics.

    Satisfied By:
        - CGState

    Mathematical Context:
        Flexible CG uses explicit orthogonalization of search directions:
            p_k ⊥_A p_j for j in sliding window [k-m, k-1]

        This requires storing recent direction vectors (p_j) and their
        matrix products (q_j = A p_j) for computing orthogonalization
        coefficients: α_j = (z_k, q_j) / (p_j, q_j)

    Usage:
        >>> def orthogonalize(state: SolverState, z: NDArray) -> NDArray:
        >>>     if isinstance(state, HasDirectionHistory):
        >>>         history = state.direction_history
        >>>         p_vectors = history.p_vectors
        >>>         q_vectors = history.q_vectors
        >>>         # Compute orthogonalization...
        >>>     return z  # No orthogonalization if no history

    Theory:
        Truncated orthogonalization maintains approximate A-conjugacy:
            (p_k, A p_j) ≈ 0 for j in window [k-m, k-1]

        This prevents numerical breakdown when the preconditioner M_k is
        not symmetric positive definite (SPD). Full orthogonalization
        would require O(k) storage; truncation reduces to O(m).

    References:
        - Notay, Y. (2000). Flexible Conjugate Gradients.
        - Saad, Y. (2003). Section 9.4: Flexible Variants.
    """

    direction_history: DirectionHistory
    """Sliding window history of search directions (p_k, q_k) for orthogonalization."""

    residual_history: ResidualHistory
    """History of residual norms (absolute and relative) for diagnostics."""
