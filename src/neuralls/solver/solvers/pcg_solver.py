"""Preconditioned Conjugate Gradient solver.

This module implements the standard Preconditioned Conjugate Gradient (PCG)
algorithm using two-term recurrence without explicit orthogonalization.

Algorithm (PCG, using Notay 2000 notation):
    Initialize: u_0 = 0, r_0 = b
    For i = 0, 1, ..., until convergence:
      1. w_i = M^{-1}(r_i)                           # Apply preconditioner
      2. β_i = (r_i, w_i) / (r_{i-1}, w_{i-1})       # Compute beta
      3. d_i = w_i + β_i * d_{i-1}                   # Two-term recurrence
      4. q_i = A @ d_i                               # Matrix-vector product
      5. α_i = (r_i, w_i) / (d_i, q_i)               # Step length
      6. u_{i+1} = u_i + α_i * d_i                   # Update solution
      7. r_{i+1} = r_i - α_i * q_i                   # Update residual

Design Principles:
    - Template Method: Inherit from ConjugateGradientBase
    - Single Responsibility: Only implements two-term recurrence direction
    - Dependency Inversion: Depends on Preconditioner abstraction

References:
    - Hestenes & Stiefel (1952). Methods of Conjugate Gradients.
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, Chapter 9.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import aslinearoperator

from ...constants import DEFAULT_ATOL, DEFAULT_RTOL
from .cg_base import ConjugateGradientBase
from ..strategies.convergence import CombinedToleranceCriterion, IConvergenceCriterion
from ..models.state import KrylovState, SolverState
from ..models.result import SolverResult
from ..preconditioners import IdentityPreconditioner, Preconditioner
from ..monitoring.trace_recorder import TraceRecorder
from ..utils.numerics import stable_dot_product

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PreconditionedCGSolver(ConjugateGradientBase):
    """Preconditioned Conjugate Gradient solver.

    Standard PCG using two-term recurrence formula. This is the classic
    CG algorithm extended with preconditioning. Requires symmetric positive
    definite (SPD) matrix and SPD preconditioner.

    Key Features:
        - Two-term recurrence (memory efficient)
        - O(1) memory for direction storage
        - Optimal for SPD systems with fixed SPD preconditioners
        - A-conjugacy maintained implicitly (no explicit orthogonalization)

    Attributes:
        preconditioner: SPD preconditioner M (default: Identity)
        convergence_criterion: Convergence test strategy
        beta_formula: Formula for beta coefficient (default: fletcher_reeves)
        event_logger: Optional event recorder for diagnostics

    Example:
        >>> solver = PreconditionedCGSolver(preconditioner=jacobi_precond)
        >>> u, result = solver.solve(A, b, rtol=1e-6)
    """

    def __init__(
        self,
        preconditioner: Preconditioner | None = None,
        convergence_criterion: IConvergenceCriterion | None = None,
        beta_formula: str = "fletcher_reeves",
        event_logger: TraceRecorder | None = None,
    ) -> None:
        """Initialize PCG solver.

        Args:
            preconditioner: SPD preconditioner M (default: Identity)
            convergence_criterion: Convergence test (default: Combined rtol/atol)
            beta_formula: Beta formula ("fletcher_reeves", "polak_ribiere", "hestenes_stiefel")
            event_logger: Optional TraceRecorder for diagnostics
        """
        self.preconditioner = preconditioner or IdentityPreconditioner()
        self.convergence_criterion = convergence_criterion or CombinedToleranceCriterion(
            rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
        )
        self.beta_formula = beta_formula
        self.event_logger = event_logger

        # Store previous w for beta computation
        self._w_prev: NDArray | None = None
        self._r_prev: NDArray | None = None
        self._rw_prev: float = 0.0

    # Implement KrylovSolverBase abstract method

    def _update_direction(
        self,
        w: NDArray,
        state: KrylovState,
        **kwargs,
    ) -> NDArray:
        """Compute PCG search direction using two-term recurrence.

        Formula (Notay 2000):
            d_0 = w_0                           # First iteration
            d_i = w_i + β_i * d_{i-1}           # Subsequent iterations

        where β_i = (r_i, w_i) / (r_{i-1}, w_{i-1}) (Fletcher-Reeves)

        Args:
            w: Preconditioned residual w_i = M^{-1}(r_i) (Notay 2000)
            state: Current KrylovState with d_{i-1}
            **kwargs: Additional parameters

        Returns:
            Search direction d_i (Notay 2000)

        Theory:
            The two-term recurrence automatically maintains A-conjugacy:
                (d_i, A*d_j) = 0 for i ≠ j

            This is guaranteed for SPD A and M, eliminating the need for
            explicit orthogonalization.
        """
        # First iteration: steepest descent
        if state.iteration == 0:
            self._w_prev = w.copy()
            self._r_prev = state.r.copy()
            self._rw_prev = stable_dot_product(state.r, w, dtype=np.float64)
            return w.copy()

        # Compute beta coefficient
        rw_curr = stable_dot_product(state.r, w, dtype=np.float64)

        if self.beta_formula == "fletcher_reeves":
            beta = rw_curr / self._rw_prev if self._rw_prev != 0 else 0.0
        elif self.beta_formula == "polak_ribiere" and self._w_prev is not None:
            # Polak-Ribiere: β = (r_i, w_i - w_{i-1}) / (r_{i-1}, w_{i-1})
            w_diff = w - self._w_prev
            numerator = stable_dot_product(state.r, w_diff, dtype=np.float64)
            beta = numerator / self._rw_prev if self._rw_prev != 0 else 0.0
            # Polak-Ribiere-Plus: reset negative beta to 0
            beta = max(0.0, beta)
        else:
            # Default to Fletcher-Reeves
            beta = rw_curr / self._rw_prev if self._rw_prev != 0 else 0.0

        # Update stored values for next iteration
        self._w_prev = w.copy()
        self._r_prev = state.r.copy()
        self._rw_prev = rw_curr

        # Two-term recurrence (Notay 2000)
        d_new = w + beta * state.d

        return d_new

    # Implement IterativeSolverBase abstract methods

    def _validate_system(
        self,
        A: Callable[[NDArray], NDArray] | NDArray,
        b: NDArray,
        x0: NDArray | None,
        **kwargs,
    ) -> None:
        """Validate system inputs."""
        b_arr = np.asarray(b, dtype=np.float64)
        if b_arr.ndim != 1:
            raise ValueError(f"b must be 1D vector, got shape {b_arr.shape}")

        if x0 is not None:
            x0_arr = np.asarray(x0, dtype=np.float64)
            if x0_arr.shape != b_arr.shape:
                raise ValueError(f"x0 shape {x0_arr.shape} != b shape {b_arr.shape}")

    def _prepare_operator(
        self,
        A: Callable[[NDArray], NDArray] | NDArray,
        b: NDArray,
        **kwargs,
    ) -> Callable[[NDArray], NDArray]:
        """Prepare linear operator."""
        # If already callable, return as is
        if callable(A) and not isinstance(A, np.ndarray):
            return A  # type: ignore[return-value]

        linear_op = aslinearoperator(A)
        return linear_op.matvec

    def _normalize_tolerances(
        self,
        b: NDArray,
        **kwargs,
    ) -> tuple[float, float]:
        """Normalize tolerances."""
        rtol = kwargs.get("rtol", DEFAULT_RTOL)
        atol = kwargs.get("atol", DEFAULT_ATOL)
        return float(rtol), float(atol)

    def _initialize_state(
        self,
        linear_op: Callable[[NDArray], NDArray],
        b: NDArray,
        x0: NDArray | None,
        **kwargs,
    ) -> KrylovState:
        """Initialize PCG state."""
        b_arr = np.asarray(b, dtype=np.float64)
        n = b_arr.shape[0]

        # Initial guess u_0 (Notay 2000)
        u = np.asarray(x0, dtype=np.float64) if x0 is not None else np.zeros(n)

        # Initial residual r_0
        r = b_arr - linear_op(u)
        r_norm = float(norm(r))

        # Apply preconditioner w_0 = M^{-1}(r_0) (Notay 2000)
        w = self._apply_preconditioner(self.preconditioner, r)

        # Initial direction (steepest descent) d_0 (Notay 2000)
        d = w.copy()

        # Matrix-vector product q_0 = A d_0
        q = linear_op(d)

        # Reset previous values
        self._w_prev = None
        self._r_prev = None
        self._rw_prev = 0.0

        # Create initial state
        return KrylovState(
            iteration=0,
            converged=False,
            breakdown=False,
            divergence=False,
            residual_norm=r_norm,
            rhs_norm=float(self.convergence_criterion.norm(b_arr)),
            u=u,
            r=r,
            w=w,
            d=d,
            q=q,
        )

    def _check_stopping(
        self,
        state: SolverState,
        rtol: float,
        atol: float,
        maxiter: int,
        **kwargs,
    ) -> bool:
        """Check if solver should stop."""
        if state.converged:
            return True

        if state.breakdown:
            return True

        if state.divergence:
            return True

        if maxiter is not None and state.iteration >= maxiter:
            return True

        # Dependency injection: create criterion once, delegate to base class helper
        criterion = self._create_convergence_criterion(rtol, atol)
        return self._check_convergence_with_criterion(state, criterion)

    def _build_result(
        self,
        state: SolverState,
        rtol: float,
        atol: float,
        **kwargs,
    ) -> SolverResult:
        """Build final SolverResult."""
        # Dependency injection: create criterion once, delegate to base class helper
        criterion = self._create_convergence_criterion(rtol, atol)
        converged = self._check_convergence_with_criterion(state, criterion)

        # Extract histories from event log using base class helper
        event_log = self._get_event_logger(**kwargs)
        (
            residual_history_abs,
            residual_history_rel,
            residual_vectors,
            solution_vectors,
        ) = self._extract_histories_from_event_log(event_log, state.rhs_norm)

        return SolverResult(
            converged=converged,
            iterations=state.iteration,
            residual=state.residual_norm / state.rhs_norm if state.rhs_norm > 0 else 0.0,
            residual_abs=state.residual_norm,
            rhs_norm=state.rhs_norm,
            breakdown=state.breakdown,
            residual_history=residual_history_rel,
            residual_history_abs=residual_history_abs,
            tol=rtol,
            atol=atol,
            event_log=event_log,  # CRITICAL: Attach event_log to result
            residual_vectors=residual_vectors,
            solution_vectors=solution_vectors,
        )
