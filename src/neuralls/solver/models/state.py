"""Typed, immutable state classes for iterative solvers.

This module defines a hierarchy of immutable state dataclasses that eliminate
the need for dynamic attributes. All state is properly typed and validated.

Hierarchy:
    SolverState (base) - Common solver state
    └── KrylovState - Krylov-specific state (adds working vectors)
        └── CGState - CG-specific state (adds direction history)

Design Principles:
    - Single Responsibility: Each class represents one level of abstraction
    - Immutable (frozen=True): Prevents accidental mutation
    - Typed: All fields properly typed, no dynamic attributes
    - Open/Closed: Easy to extend for new solver types

Theory:
    Krylov methods maintain working vectors (x, r, z, p, q) that are updated
    each iteration. Previously these were added dynamically to IterationState,
    causing type safety issues. Now all fields are explicit and typed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .history import DirectionHistory, ResidualHistory


@dataclass(frozen=True)
class SolverState:
    """Base solver state common to all iterative solvers.

    Contains diagnostic information and counters shared by all solver types.
    Immutable (frozen=True) to prevent accidental mutation.

    Attributes:
        iteration: Current iteration number (0-indexed).
        converged: Whether convergence criterion has been satisfied.
        breakdown: Whether numerical breakdown occurred (NaN/Inf). Terminal condition.
        divergence: Whether divergence detected (||r|| > threshold).
        residual_norm: Current residual norm ||r_k||_2.
        rhs_norm: Right-hand side norm ||b||_2 (constant).
        num_restarts: Count of restart events.
        num_residual_replacements: Count of true residual recomputations.

    Example:
        >>> state = SolverState(
        ...     iteration=0,
        ...     converged=False,
        ...     breakdown=False,
        ...     divergence=False,
        ...     residual_norm=1.0,
        ...     rhs_norm=1.0,
        ... )
        >>> state.converged
        False
    """

    # Required fields (no defaults)
    iteration: int
    """Current iteration number (0-indexed)."""

    converged: bool
    """Whether convergence criterion satisfied: ||r|| <= max(rtol*||b||, atol)."""

    breakdown: bool
    """Whether numerical breakdown occurred (NaN/Inf). Terminal condition."""

    divergence: bool
    """Whether divergence detected. Triggers residual recomputation."""

    residual_norm: float
    """Current residual norm ||r_k||_2."""

    rhs_norm: float
    """Right-hand side norm ||b||_2 (constant throughout solve)."""

    # Optional fields (with defaults) - must come after required fields
    num_restarts: int = 0
    """Count of restart events (for non-SPD preconditioner recovery)."""

    num_residual_replacements: int = 0
    """Count of true residual recomputation events (r = b - A*x)."""

    def __post_init__(self) -> None:
        """Validate state invariants.

        Raises:
            ValueError: If state has inconsistent fields.
        """
        if self.iteration < 0:
            raise ValueError(f"iteration must be >= 0, got {self.iteration}")
        if self.residual_norm < 0:
            raise ValueError(f"residual_norm must be >= 0, got {self.residual_norm}")
        if self.rhs_norm < 0:
            raise ValueError(f"rhs_norm must be >= 0, got {self.rhs_norm}")


@dataclass(frozen=True, kw_only=True)
class KrylovState(SolverState):
    """Krylov solver state with working vectors.

    Extends SolverState with the working vectors (x, r, z, p, q) that
    Krylov methods maintain. All vectors are properly typed, no dynamic attributes.

    Note: kw_only=True ensures all fields must be passed as keyword arguments,
    preventing positional argument confusion with parent class fields.

    Attributes:
        x: Current solution vector x_k.
        r: Current residual vector r_k = b - A*x_k.
        z: Preconditioned residual z_k = M^{-1} r_k.
        p: Search direction p_k.
        q: Matrix-vector product q_k = A p_k.

    Theory:
        Krylov methods iteratively update these vectors:
        1. Preconditioning: z_k = M^{-1} r_k
        2. Direction: p_k = f(z_k, history)
        3. Matrix-vector: q_k = A p_k
        4. Step length: α_k = (p_k, r_k) / (p_k, q_k)
        5. Solution: x_{k+1} = x_k + α_k p_k
        6. Residual: r_{k+1} = r_k - α_k q_k

    Example:
        >>> import numpy as np
        >>> state = KrylovState(
        ...     iteration=0,
        ...     converged=False,
        ...     breakdown=False,
        ...     divergence=False,
        ...     residual_norm=1.0,
        ...     rhs_norm=1.0,
        ...     x=np.zeros(10),
        ...     r=np.ones(10),
        ...     z=np.ones(10),
        ...     p=np.ones(10),
        ...     q=np.ones(10),
        ... )
        >>> state.x.shape
        (10,)
    """

    x: NDArray
    """Current solution vector x_k."""

    r: NDArray
    """Current residual vector r_k = b - A*x_k."""

    z: NDArray
    """Preconditioned residual z_k = M^{-1} r_k."""

    p: NDArray
    """Search direction p_k."""

    q: NDArray
    """Matrix-vector product q_k = A p_k."""


@dataclass(frozen=True, kw_only=True)
class CGState(KrylovState):
    """Conjugate Gradient solver state with history tracking.

    Extends KrylovState with CG-specific history buffers for direction
    orthogonalization and residual tracking.

    Attributes:
        direction_history: Sliding window of search directions and products.
            Used by FCG for truncated Gram-Schmidt orthogonalization.

        residual_history: Residual norms (absolute and relative).
            Used for convergence monitoring and post-analysis.

    Theory:
        FCG maintains a window of previous search directions to orthogonalize
        the new direction against:
            p_i = z_i - Σ_{k=i-m}^{i-1} [(z_i, q_k) / (p_k, q_k)] p_k

        The direction_history stores the last m (p_k, q_k) pairs.

    Example:
        >>> import numpy as np
        >>> from neuralls.solver.models.history import DirectionHistory, ResidualHistory
        >>> state = CGState(
        ...     iteration=0,
        ...     converged=False,
        ...     breakdown=False,
        ...     divergence=False,
        ...     residual_norm=1.0,
        ...     rhs_norm=1.0,
        ...     x=np.zeros(10),
        ...     r=np.ones(10),
        ...     z=np.ones(10),
        ...     p=np.ones(10),
        ...     q=np.ones(10),
        ...     direction_history=DirectionHistory.empty(max_size=10),
        ...     residual_history=ResidualHistory.empty(),
        ... )
        >>> len(state.direction_history)
        0
    """

    direction_history: DirectionHistory
    """Sliding window of search directions and matrix-vector products."""

    residual_history: ResidualHistory
    """Residual norms (absolute and relative) across iterations."""

    @classmethod
    def create_initial(
        cls,
        x: NDArray,
        r: NDArray,
        z: NDArray,
        p: NDArray,
        q: NDArray,
        residual_norm: float,
        rhs_norm: float,
        max_history: int = 10,
    ) -> CGState:
        """Create initial CG state with empty histories.

        Args:
            x: Initial solution x_0.
            r: Initial residual r_0 = b - A*x_0.
            z: Initial preconditioned residual z_0 = M^{-1} r_0.
            p: Initial search direction p_0 = z_0.
            q: Initial matrix-vector product q_0 = A p_0.
            residual_norm: Initial residual norm ||r_0||_2.
            rhs_norm: RHS norm ||b||_2.
            max_history: Maximum direction history size.

        Returns:
            New CGState at iteration 0 with empty histories.

        Example:
            >>> import numpy as np
            >>> n = 10
            >>> b = np.ones(n)
            >>> state = CGState.create_initial(
            ...     x=np.zeros(n),
            ...     r=b.copy(),
            ...     z=b.copy(),
            ...     p=b.copy(),
            ...     q=np.ones(n),
            ...     residual_norm=float(np.linalg.norm(b)),
            ...     rhs_norm=float(np.linalg.norm(b)),
            ...     max_history=10,
            ... )
            >>> state.iteration
            0
        """
        from .history import DirectionHistory, ResidualHistory

        return cls(
            iteration=0,
            converged=False,
            breakdown=False,
            divergence=False,
            residual_norm=residual_norm,
            rhs_norm=rhs_norm,
            num_restarts=0,
            num_residual_replacements=0,
            x=x,
            r=r,
            z=z,
            p=p,
            q=q,
            direction_history=DirectionHistory.empty(max_size=max_history),
            residual_history=ResidualHistory.empty().add(
                norm_abs=residual_norm,
                norm_rel=residual_norm / rhs_norm if rhs_norm > 0 else residual_norm,
            ),
        )
