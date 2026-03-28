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
    >>> from neuralls.domain.solver.models.protocols import HasVectors
    >>>
    >>> def process_state(state: SolverState) -> None:
    >>>     if isinstance(state, HasVectors):
    >>> # Type checker knows state.x, state.r exist
    >>>         residual = state.r  # ✅ No type: ignore needed
    >>>     else:
    >>> # Handle base state without vectors
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

    Mathematical Context (Notay 2000):
        Krylov methods maintain these vectors at each iteration i:
        - u_i: Current solution estimate (Notay 2000)
        - r_i: Residual r_i = b - A u_i
        - w_i: Preconditioned residual w_i = M^{-1} r_i (Notay 2000)
        - d_i: Search direction (A-conjugate to previous directions) (Notay 2000)
        - q_i: Matrix-vector product q_i = A d_i

    Usage:
        >>> def extract_residual(state: SolverState) -> NDArray | None:
        >>>     if isinstance(state, HasVectors):
        >>>         return state.r  # ✅ Type-safe access
        >>>     return None

    Theory:
        The Krylov subspace K_i(A, r_0) = span{r_0, A*r_0, ..., A^(i-1)*r_0}
        is implicitly represented by these vectors. The search direction d_i
        is constructed to be A-conjugate to all previous directions.

    References:
        - Notay, Y. (2000). Flexible Conjugate Gradients.
        - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems.
        - Greenbaum, A. (1997). Iterative Methods for Solving Linear Systems.
    """

    u: NDArray
    """Solution vector u_i at current iteration (Notay 2000)."""

    r: NDArray
    """Residual vector r_i = b - A u_i."""

    w: NDArray
    """Preconditioned residual w_i = M^{-1} r_i (Notay 2000)."""

    d: NDArray
    """Search direction d_i (A-conjugate to previous directions) (Notay 2000)."""

    q: NDArray
    """Matrix-vector product q_i = A d_i."""


@runtime_checkable
class HasDirectionHistory(Protocol):
    """Protocol for states with CG direction and residual history tracking.

    Classes satisfying this protocol maintain sliding window histories of
    search directions and residuals for orthogonalization and diagnostics.

    Satisfied By:
        - CGState

    Mathematical Context (Notay 2000):
        Flexible CG uses explicit orthogonalization of search directions:
            d_i ⊥_A d_j for j in sliding window [i-m, i-1]

        This requires storing recent direction vectors (d_j) and their
        matrix products (q_j = A d_j) for computing orthogonalization
        coefficients: β_j = (w_i, q_j) / (d_j, q_j)

    Usage:
        >>> def orthogonalize(state: SolverState, w: NDArray) -> NDArray:
        >>>     if isinstance(state, HasDirectionHistory):
        >>>         history = state.direction_history
        >>>         d_vectors = history.d_vectors
        >>>         q_vectors = history.q_vectors
        >>> # Compute orthogonalization...
        >>>     return w  # No orthogonalization if no history

    Theory (Notay 2000):
        Truncated orthogonalization maintains approximate A-conjugacy:
            (d_i, A d_j) ≈ 0 for j in window [i-m, i-1]

        This prevents numerical breakdown when the preconditioner M_i is
        not symmetric positive definite (SPD). Full orthogonalization
        would require O(i) storage; truncation reduces to O(m).

    References:
        - Notay, Y. (2000). Flexible Conjugate Gradients.
        - Saad, Y. (2003). Section 9.4: Flexible Variants.
    """

    direction_history: DirectionHistory
    """Sliding window history of search directions (d_i, q_i) for orthogonalization."""

    residual_history: ResidualHistory
    """History of residual norms (absolute and relative) for diagnostics."""
