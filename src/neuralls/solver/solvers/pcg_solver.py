"""Preconditioned Conjugate Gradient solver with optional reorthogonalization.

This module implements the standard Preconditioned Conjugate Gradient (PCG)
algorithm using two-term recurrence. Optional reorthogonalization can be enabled
to correct accumulated rounding errors in finite precision arithmetic.

Algorithm (PCG, using Notay 2000 notation):
    Initialize: u_0 = 0, r_0 = b
    For i = 0, 1, ..., until convergence:
      1. w_i = M^{-1}(r_i)                           # Apply preconditioner
      2. β_i = (r_i, w_i) / (r_{i-1}, w_{i-1})       # Compute beta
      3. d_i = w_i + β_i * d_{i-1}                   # Two-term recurrence
      3b. (Optional) d_i := d_i - Σ[(d_i, A d_j)/(d_j, A d_j)] d_j  # Reorthogonalization
      4. q_i = A @ d_i                               # Matrix-vector product
      5. α_i = (r_i, w_i) / (d_i, q_i)               # Step length
      6. u_{i+1} = u_i + α_i * d_i                   # Update solution
      7. r_{i+1} = r_i - α_i * q_i                   # Update residual

Reorthogonalization (Optional):
    The two-term recurrence maintains A-conjugacy in exact arithmetic, but
    accumulated rounding errors can degrade conjugacy in finite precision.
    Optional reorthogonalization restores conjugacy by explicitly correcting
    the direction vector against previous directions.

    This is ADDITIVE: reorthogonalization happens AFTER the standard PCG
    two-term recurrence, preserving PCG's structure while correcting errors.

Design Principles:
    - Template Method: Inherit from ConjugateGradientBase
    - Single Responsibility: Only implements two-term recurrence direction
    - Dependency Inversion: Depends on Preconditioner and OrthogonalizationStrategy abstractions
    - Open/Closed: Reorthogonalization is optional via strategy pattern

References:
    - Hestenes & Stiefel (1952). Methods of Conjugate Gradients.
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, Chapter 9.
    - Notay, Y. (2000). Flexible Conjugate Gradients. SIAM J. Sci. Comput.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import aslinearoperator

from ...constants import DEFAULT_ATOL, DEFAULT_RTOL, DEFAULT_FCG_HISTORY_LIMIT
from .cg_base import ConjugateGradientBase
from ..strategies.convergence import CombinedToleranceCriterion, IConvergenceCriterion
from ..strategies.orthogonalization import OrthogonalizationStrategy
from ..models.state import KrylovState, CGState, SolverState
from ..models.result import SolverResult
from ..preconditioners import IdentityPreconditioner, Preconditioner
from ..monitoring.trace_recorder import TraceRecorder
from ..utils.numerics import stable_dot_product

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PreconditionedCGSolver(ConjugateGradientBase):
    """Preconditioned Conjugate Gradient solver with optional reorthogonalization.

    Standard PCG using two-term recurrence formula. This is the classic
    CG algorithm extended with preconditioning. Requires symmetric positive
    definite (SPD) matrix and SPD preconditioner.

    Optional reorthogonalization can be enabled to correct accumulated rounding
    errors, at the cost of increased memory (O(m*n) for m directions).

    Algorithm (without reorthogonalization):
        At each iteration k:
        1. Apply preconditioner: z_k = M^{-1} r_k
        2. Compute beta: β_k = (r_k, z_k) / (r_{k-1}, z_{k-1})
        3. Update direction: d_k = z_k + β_k * d_{k-1}
        4. Compute step length: α_k = (r_k, z_k) / (d_k, A d_k)
        5. Update solution: x_{k+1} = x_k + α_k * d_k
        6. Update residual: r_{k+1} = r_k - α_k * A d_k

    Algorithm (with reorthogonalization):
        Steps 1-3 as above, then:
        3b. Reorthogonalize: d_k := d_k - Σ[(d_k, A d_j)/(d_j, A d_j)] * d_j
        Steps 4-6 as above.

    Use reorthogonalization when:
        - Problem is ill-conditioned (large condition number)
        - Long iteration sequences expected
        - Numerical accuracy is critical
        - Standard PCG shows loss of conjugacy

    Key Features:
        - Two-term recurrence (memory efficient)
        - O(1) memory when reorthogonalization disabled (default)
        - O(m*n) memory when reorthogonalization enabled
        - Optimal for SPD systems with fixed SPD preconditioners
        - A-conjugacy maintained implicitly (or explicitly with reorthogonalization)

    Attributes:
        preconditioner: SPD preconditioner M (default: Identity)
        convergence_criterion: Convergence test strategy
        beta_formula: Formula for beta coefficient (default: fletcher_reeves)
        orthogonalization: Optional orthogonalization strategy (default: None)
        event_logger: Optional event recorder for diagnostics

    Examples:
        >>> # Standard PCG (most efficient)
        >>> solver = PreconditionedCGSolver(preconditioner=jacobi_precond)
        >>> u, result = solver.solve(A, b, rtol=1e-6)

        >>> # PCG with full reorthogonalization (ill-conditioned problems)
        >>> from neuralls.solver.strategies.orthogonalization import FullOrthogonalization
        >>> solver = PreconditionedCGSolver(
        ...     preconditioner=jacobi_precond,
        ...     orthogonalization=FullOrthogonalization()
        ... )
        >>> u, result = solver.solve(A, b, rtol=1e-6)

        >>> # PCG with truncated reorthogonalization (window size 20)
        >>> from neuralls.solver.strategies.orthogonalization import TruncatedGramSchmidt
        >>> solver = PreconditionedCGSolver(
        ...     preconditioner=jacobi_precond,
        ...     orthogonalization=TruncatedGramSchmidt(window_size=20)
        ... )
        >>> u, result = solver.solve(A, b, rtol=1e-6)
    """

    def __init__(
        self,
        preconditioner: Preconditioner | None = None,
        convergence_criterion: IConvergenceCriterion | None = None,
        beta_formula: str = "fletcher_reeves",
        orthogonalization: OrthogonalizationStrategy | None = None,
        event_log: TraceRecorder | None = None,
        trace_mode = None,
    ) -> None:
        """Initialize PCG solver.

        Args:
            preconditioner: SPD preconditioner M (default: Identity)
            convergence_criterion: Convergence test (default: Combined rtol/atol)
            beta_formula: Beta formula ("fletcher_reeves", "polak_ribiere", "hestenes_stiefel")
            orthogonalization: Optional orthogonalization strategy for correcting
                rounding errors (default: None, disabled). When enabled, applies
                reorthogonalization AFTER the standard two-term recurrence to
                restore A-conjugacy degraded by finite precision arithmetic.
            event_log: Optional TraceRecorder for diagnostics
            trace_mode: Logging granularity (DISABLED, MINIMAL, FULL)
        """
        from ..monitoring.trace_mode import TraceMode

        self.preconditioner = preconditioner or IdentityPreconditioner()
        self.convergence_criterion = convergence_criterion or CombinedToleranceCriterion(
            rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
        )
        self.beta_formula = beta_formula
        self.orthogonalization = orthogonalization

        # Store previous w for beta computation
        self._w_prev: NDArray | None = None
        self._r_prev: NDArray | None = None
        self._rw_prev: float = 0.0

        # Initialize base class with logging configuration
        trace_mode_enum = trace_mode if trace_mode is not None else TraceMode.MINIMAL
        super().__init__(event_log=event_log, trace_mode=trace_mode_enum)

    # Implement KrylovSolverBase abstract method

    def _update_direction(
        self,
        w: NDArray,
        state: KrylovState | CGState,
    ) -> NDArray:
        """Compute PCG search direction with optional reorthogonalization.

        Formula (without reorthogonalization):
            d_0 = w_0                           # First iteration
            d_i = w_i + β_i * d_{i-1}           # Subsequent iterations

        Formula (with reorthogonalization):
            d_i = w_i + β_i * d_{i-1}                                    # Two-term recurrence
            d_i := d_i - Σ_j [(d_i, A d_j) / (d_j, A d_j)] * d_j        # Reorthogonalization

        where β_i = (r_i, w_i) / (r_{i-1}, w_{i-1}) (Fletcher-Reeves)

        Args:
            w: Preconditioned residual w_i = M^{-1}(r_i) (Notay 2000)
            state: Current KrylovState or CGState

        Returns:
            Search direction d_i (with optional reorthogonalization)

        Theory:
            The two-term recurrence automatically maintains A-conjugacy in exact arithmetic.
            Reorthogonalization corrects accumulated rounding errors that degrade conjugacy
            in finite precision arithmetic.
        """
        # First iteration: steepest descent
        if state.iteration == 0:
            self._w_prev = w.copy()
            self._r_prev = state.r.copy()
            self._rw_prev = stable_dot_product(state.r, w, dtype=np.float64)
            return w.copy()

        # Phase 1: Standard two-term recurrence (UNCHANGED)
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

        # Phase 2: Optional reorthogonalization (NEW)
        if self.orthogonalization is not None:
            from ..models.protocols import HasDirectionHistory

            # Extract direction history from CGState
            if isinstance(state, HasDirectionHistory):
                history = state.direction_history
                d_vectors = list(history.d_vectors)
                q_vectors = list(history.q_vectors)

                # Reorthogonalize against previous directions
                # Formula: d_i := d_i - Σ[(d_i, A d_j)/(d_j, A d_j)] * d_j
                d_reorthog, _report = self.orthogonalization.orthogonalize(
                    vector=d_new,
                    d_vectors=d_vectors,
                    q_vectors=q_vectors,
                )

                return d_reorthog

            # Fallback: state doesn't have history (shouldn't happen if _initialize_state is correct)
            import warnings
            warnings.warn(
                "Orthogonalization enabled but state does not have direction history. "
                "Skipping reorthogonalization for this iteration.",
                RuntimeWarning,
                stacklevel=2,
            )

        return d_new

    # Implement IterativeSolverBase abstract methods

    def _initialize_state(
        self,
        linear_op: Callable[[NDArray], NDArray],
        b: NDArray,
        x0: NDArray | None,
        maxiter: int | None = None,
    ) -> CGState:
        """Initialize PCG state.

        Returns:
            CGState: Always returns CGState (CGState wraps KrylovState and adds history)
        """
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

        # Compute history size (0 if no orthogonalization, or computed size)
        max_history = 0
        if self.orthogonalization is not None:
            max_history = self._compute_max_history(n, maxiter)

        # Always return CGState (even when max_history=0 for consistency)
        return CGState.create_initial(
            u=u,
            r=r,
            w=w,
            d=d,
            q=q,
            residual_norm=r_norm,
            rhs_norm=float(self.convergence_criterion.norm(b_arr)),
            max_history=max_history,
        )

    def _check_stopping(
        self,
        state: SolverState,
        rtol: float,
        atol: float,
        maxiter: int,
        breakdown_tol: float | None = None,
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
        breakdown_tol: float | None = None,
    ) -> SolverResult:
        """Build final SolverResult."""
        # Dependency injection: create criterion once, delegate to base class helper
        criterion = self._create_convergence_criterion(rtol, atol)
        converged = self._check_convergence_with_criterion(state, criterion)

        # Extract histories from event log using base class helper
        (
            residual_history_abs,
            residual_history_rel,
            residual_vectors,
            solution_vectors,
        ) = self._extract_histories_from_event_log(self.event_log, state.rhs_norm)

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
            event_log=self.event_log,  # CRITICAL: Attach event_log to result
            residual_vectors=residual_vectors,
            solution_vectors=solution_vectors,
        )
