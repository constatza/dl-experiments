"""Conjugate Gradient solver family with proper SOLID separation.

This module implements the CG solver hierarchy:
    IterativeSolverBase (template method pattern)
        ↓ inherits
    ConjugateGradientSolver (unified CG logic)
        ↓ inherits
        ├─ PCGSolver (two-term recurrence)
        └─ FCGSolver (orthogonalization-based)

Architecture Benefits:
    - Single Responsibility: PCGSolver and FCGSolver have distinct algorithms
    - Liskov Substitution: convergence_criterion is always honored
    - Interface Segregation: Each solver exposes only relevant parameters
    - No duck typing: Type-safe composition at construction time
    - Clear semantics: Solver class name indicates algorithm variant

Algorithm (unified CG iteration):
    Initialize: u_0, r_0 = b - A*u_0
    For k = 0, 1, ..., until convergence:
        1. w_k = M^{-1}(r_k)           # Apply preconditioner
        2. d_k = strategy.compute_direction(w_k, state)  # STRATEGY
        3. q_k = A @ d_k               # Matrix-vector product
        4. α_k = (d_k, r_k) / (d_k, A d_k)  # Step length
        5. u_{k+1} = u_k + α_k * d_k   # Update solution
        6. r_{k+1} = r_k - α_k * q_k   # Update residual

Design Principles:
    - Strategy Pattern: Direction computation via DirectionStrategy
    - Template Method: Inherits from IterativeSolverBase
    - Single Responsibility: Each solver has one algorithm
    - Composition over Inheritance: Strategies injected, not inherited

References:
    - Hestenes & Stiefel (1952). Methods of Conjugate Gradients.
    - Notay, Y. (2000). Flexible Conjugate Gradients.
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np

from .models.state import CGState, KrylovState, SolverState
from ..constants import DEFAULT_ATOL, DEFAULT_RTOL
from .base import IterativeSolverBase
from .strategies.convergence import CombinedToleranceCriterion, IConvergenceCriterion
from .strategies.direction import (
    DirectionStrategy,
    TwoTermRecurrenceStrategy,
    OrthogonalizationDirectionStrategy,
    CompositeDirectionStrategy,
)
from .strategies.orthogonalization import OrthogonalizationStrategy
from .models.result import SolverResult
from .preconditioners import Identity, Preconditioner
from .monitoring.event_log import EventLog
from .monitoring.iteration_history import IterationHistory
from .monitoring.trace_mode import TraceMode
from .utils.numerics import stable_dot_product

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ConjugateGradientSolver(IterativeSolverBase[KrylovState]):
    """Concrete Conjugate Gradient solver implementing the Strategy pattern.

    Provides common CG iteration logic and delegates direction computation
    to the injected DirectionStrategy.

    Attributes:
        direction_strategy: Strategy for computing search directions (PCG/FCG).
        preconditioner: Preconditioner M (default: Identity).
        convergence_criterion: Convergence test strategy.
        iteration_history: Optional IterationHistory for continuous monitoring.
        event_log: Optional EventLog for discrete event logging.
        trace_mode: Logging granularity (DISABLED, MINIMAL, FULL).
    """

    def __init__(
        self,
        direction_strategy: DirectionStrategy,
        preconditioner: Preconditioner | None = None,
        convergence_criterion: IConvergenceCriterion | None = None,
        iteration_history: IterationHistory | None = None,
        event_log: EventLog | None = None,
        trace_mode: TraceMode = TraceMode.MINIMAL,
    ) -> None:
        """Initialize CG solver.

        Args:
            direction_strategy: Strategy for direction computation.
            preconditioner: Preconditioner M (default: Identity).
            convergence_criterion: Convergence test (default: Combined rtol/atol).
            iteration_history: Optional IterationHistory for continuous monitoring.
            event_log: Optional EventLog for discrete event logging.
            trace_mode: Logging granularity (default: MINIMAL).
        """
        self.direction_strategy = direction_strategy
        self.preconditioner = preconditioner or Identity()
        self.convergence_criterion = (
            convergence_criterion
            or CombinedToleranceCriterion(rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
        )

        super().__init__(
            iteration_history=iteration_history,
            event_log=event_log,
            trace_mode=trace_mode,
        )

    # Implement IterativeSolverBase abstract methods

    def _initialize_state(
        self,
        linear_op: Callable[[NDArray], NDArray],
        b: NDArray,
        x0: NDArray | None,
        maxiter: int | None = None,
    ) -> CGState:
        """Initialize CG state.

        Creates initial state with u_0, r_0, w_0, d_0, q_0 and empty histories.
        Allocates history based on direction_strategy requirements.

        Args:
            linear_op: Prepared linear operator.
            b: Right-hand side.
            x0: Initial guess.
            maxiter: Maximum iterations (used for history sizing).

        Returns:
            Initial CGState with iteration=0, all vectors initialized.
        """
        b_arr = np.asarray(b, dtype=np.float64)
        n = b_arr.shape[0]

        # Initial guess u_0 (Notay 2000)
        u = np.asarray(x0, dtype=np.float64) if x0 is not None else np.zeros(n)

        # Initial residual r_0 = b - A*u_0
        r = b_arr - linear_op(u)
        r_norm = float(np.linalg.norm(r))
        rhs_norm = float(self.convergence_criterion.norm(b_arr))

        # Create temporary initial state for contextual preconditioners
        from neuralls.solver.models.state import SolverState

        initial_state = SolverState(
            iteration=0,
            converged=False,
            breakdown=False,
            divergence=False,
            residual_norm=r_norm,
            rhs_norm=rhs_norm,
        )

        # Apply preconditioner w_0 = M^{-1}(r_0) (Notay 2000)
        w = self._apply_preconditioner(self.preconditioner, r, initial_state)

        # Initial direction (steepest descent) d_0 = w_0 (Notay 2000)
        d = w.copy()

        # Matrix-vector product q_0 = A @ d_0
        q = linear_op(d)

        # Determine history requirement from strategy
        req_history = self.direction_strategy.required_history_size
        if req_history is None:
            # Unlimited history (e.g., FCG(∞)) -> use maxiter or n
            max_history = maxiter if maxiter is not None else n
        else:
            max_history = req_history

        # Create initial state using factory method
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

    def _iterate_step(
        self,
        linear_op: Callable[[NDArray], NDArray],
        state: KrylovState,
        breakdown_tol: float | None = None,
    ) -> SolverState:
        """Execute single CG iteration.

        Unified CG iteration merging KrylovSolverBase and ConjugateGradientBase logic:
            1. Apply preconditioner: w_k = M^{-1}(r_k)
            2. Compute direction: d_k = strategy.compute_direction(w_k, state)
            3. Matrix-vector product: q_k = A @ d_k
            4. Compute curvature and step length
            5. Update solution: u_{k+1} = u_k + α_k * d_k
            6. Update residual: r_{k+1} = r_k - α_k * q_k

        Args:
            linear_op: Linear operator A
            state: Current CGState
            breakdown_tol: Breakdown detection tolerance

        Returns:
            Updated CGState for next iteration
        """
        from ..constants import DEFAULT_BREAKDOWN_TOL

        # Resolve breakdown tolerance
        breakdown_tol_eff = (
            breakdown_tol if breakdown_tol is not None else DEFAULT_BREAKDOWN_TOL
        )

        # Step 1: Apply preconditioner (Notay 2000: w_k = M^{-1}(r_k))
        w = self._apply_preconditioner(self.preconditioner, state.r, state)

        # Compute inner product for two-term recurrence (needed for beta)
        rw_curr = stable_dot_product(state.r, w)

        # Step 2: Compute direction using strategy (STRATEGY PATTERN)
        d = self.direction_strategy.compute_direction(w, state)

        # Step 3: Matrix-vector product (Notay 2000: q_k = A @ d_k)
        q = linear_op(d)

        # Step 4: Compute step length (matches scipy.cg lines 415-416)
        # scipy.cg: alpha = rho_cur / dotprod(p, q)
        # Notay 2000: α_k = (w_k, r_k) / (d_k, A d_k)
        numerator = rw_curr
        curvature = stable_dot_product(d, q)
        alpha = numerator / curvature

        # Step 5-6: Update solution and residual (matches scipy.cg lines 417-418)
        # scipy.cg: x += alpha*p; r -= alpha*q
        u_new = state.u + alpha * d
        r_new = state.r - alpha * q
        residual_norm_new = float(np.linalg.norm(r_new))

        # Update direction history (for strategies that need it)
        direction_history = getattr(state, "direction_history", None)
        if direction_history is not None and hasattr(direction_history, "add"):
            # Only add to history if strategy actually needs it (optimization)
            # A window size of 0 implies no history needed
            req_size = self.direction_strategy.required_history_size
            if req_size is None or req_size > 0:
                direction_history = direction_history.add(d, q)

        # Update residual history
        residual_history = getattr(state, "residual_history", None)
        if residual_history is not None and hasattr(residual_history, "add"):
            rel_residual = self._compute_relative_residual(
                residual_norm_new, state.rhs_norm
            )
            residual_history = residual_history.add(residual_norm_new, rel_residual)

        # Create new state with updated vectors
        update_dict = {
            "iteration": state.iteration + 1,
            "u": u_new,
            "r": r_new,
            "w": w,
            "d": d,
            "q": q,
            "residual_norm": residual_norm_new,
        }

        # Only add CGState-specific fields if they exist
        if hasattr(state, "direction_history"):
            update_dict["direction_history"] = direction_history
        if hasattr(state, "residual_history"):
            update_dict["residual_history"] = residual_history

        # Update two-term recurrence state (for stateless TwoTermRecurrenceStrategy)
        if hasattr(state, "w_prev"):
            update_dict["w_prev"] = w.copy()
        if hasattr(state, "r_prev"):
            update_dict["r_prev"] = state.r.copy()
        if hasattr(state, "rw_prev"):
            update_dict["rw_prev"] = rw_curr

        return replace(state, **update_dict)

    def _check_stopping(
        self,
        state: KrylovState,
        rtol: float,
        atol: float,
        maxiter: int,
        breakdown_tol: float | None = None,
    ) -> bool:
        """Check if solver should stop.

        Checks for:
            - Convergence: ||r|| <= max(rtol * ||b||, atol)
            - Breakdown: NaN/Inf detected
            - Divergence: residual increasing significantly
            - Max iterations reached

        Args:
            state: Current CGState
            rtol: Relative tolerance (ignored if convergence_criterion set in __init__)
            atol: Absolute tolerance (ignored if convergence_criterion set in __init__)
            maxiter: Maximum iterations
            breakdown_tol: Breakdown detection tolerance (unused)

        Returns:
            True if should stop, False if should continue
        """
        # Check convergence flag
        if state.converged:
            return True

        # Check breakdown flag
        if state.breakdown:
            return True

        # Check divergence flag
        if state.divergence:
            return True

        # Check max iterations
        if maxiter is not None and state.iteration >= maxiter:
            return True

        # Use self.convergence_criterion (honors injected criterion!)
        return self._check_convergence_with_criterion(state, self.convergence_criterion)

    def _build_result(
        self,
        state: SolverState,
        rtol: float,
        atol: float,
        breakdown_tol: float | None = None,
    ) -> SolverResult:
        """Build final SolverResult.

        Args:
            state: Final CGState
            rtol: Relative tolerance used (for result only)
            atol: Absolute tolerance used (for result only)
            breakdown_tol: Breakdown tolerance used (unused)

        Returns:
            SolverResult with convergence info, histories, diagnostics
        """
        # Check final solution validity and convergence
        from .utils.validation import check_solution_validity, record_breakdown_event

        if not check_solution_validity(state.u):
            record_breakdown_event(self.event_log, state.u, state.iteration)
            converged = False
        else:
            # Use self.convergence_criterion (honors injected criterion!)
            converged = self._check_convergence_with_criterion(
                state, self.convergence_criterion
            )

        # Extract histories from iteration history
        (
            residual_history_abs,
            residual_history_rel,
            residual_vectors,
            solution_vectors,
        ) = self._extract_histories_from_iteration_history(
            self.iteration_history, state.rhs_norm
        )

        # Fallback to CGState.residual_history if iteration_history unavailable
        from .models.protocols import HasDirectionHistory

        if residual_history_abs is None and isinstance(state, HasDirectionHistory):
            residual_history_abs = list(state.residual_history.norms_abs)
            # Compute relative residuals with consistent zero-RHS handling
            residual_history_rel = [
                self._compute_relative_residual(r, state.rhs_norm)
                for r in residual_history_abs
            ]

        return SolverResult(
            converged=converged,
            iterations=state.iteration,
            residual=self._compute_relative_residual(
                state.residual_norm, state.rhs_norm
            ),
            residual_abs=state.residual_norm,
            rhs_norm=state.rhs_norm,
            breakdown=state.breakdown,
            residual_history=residual_history_rel,
            residual_history_abs=residual_history_abs,
            tol=rtol,
            atol=atol,
            event_log=self.event_log,
            iteration_history=self.iteration_history,
            residual_vectors=residual_vectors,
            solution_vectors=solution_vectors,
        )

    # Protected helper methods

    def _compute_relative_residual(self, abs_residual: float, rhs_norm: float) -> float:
        """Compute relative residual with zero-RHS handling.

        Args:
            abs_residual: Absolute residual norm ||r||
            rhs_norm: Right-hand side norm ||b||

        Returns:
            Relative residual ||r|| / ||b|| if ||b|| > threshold, else ||r||

        Theory:
            When ||b|| is very small or zero, relative residual is ill-defined.
            In this case, we fall back to absolute residual comparison.
        """
        return abs_residual / rhs_norm if rhs_norm > 1e-14 else abs_residual


class PCGSolver(ConjugateGradientSolver):
    """Preconditioned Conjugate Gradient solver wrapper.

    Configures ConjugateGradientSolver with TwoTermRecurrenceStrategy,
    optionally combining with reorthogonalization.

    Attributes:
        reorthogonalization: Optional reorthogonalization strategy
        beta_formula: Beta coefficient formula (default: fletcher_reeves)
        preconditioner: Preconditioner M
        convergence_criterion: Convergence test
    """

    def __init__(
        self,
        preconditioner: Preconditioner | None = None,
        convergence_criterion: IConvergenceCriterion | None = None,
        reorthogonalization: OrthogonalizationStrategy | None = None,
        beta_formula: str = "fletcher_reeves",
        iteration_history: IterationHistory | None = None,
        event_log: EventLog | None = None,
        trace_mode: TraceMode = TraceMode.MINIMAL,
    ) -> None:
        """Initialize PCG solver.

        Args:
            preconditioner: Preconditioner M (default: Identity)
            convergence_criterion: Convergence test (default: Combined rtol/atol)
            reorthogonalization: Optional reorthogonalization strategy
            beta_formula: Beta coefficient formula (default: fletcher_reeves)
            iteration_history: Optional IterationHistory for continuous monitoring
            event_log: Optional EventLog for discrete event logging
            trace_mode: Logging granularity (default: MINIMAL)
        """
        self.reorthogonalization = reorthogonalization
        self.beta_formula = beta_formula

        # Build strategy
        base_strategy = TwoTermRecurrenceStrategy(beta_formula=beta_formula)
        if reorthogonalization is not None:
            direction_strategy = CompositeDirectionStrategy(
                base_strategy=base_strategy,
                reorthog_strategy=reorthogonalization,
            )
        else:
            direction_strategy = base_strategy

        # Initialize base solver with configured strategy
        super().__init__(
            direction_strategy=direction_strategy,
            preconditioner=preconditioner,
            convergence_criterion=convergence_criterion,
            iteration_history=iteration_history,
            event_log=event_log,
            trace_mode=trace_mode,
        )


class FCGSolver(ConjugateGradientSolver):
    """Flexible Conjugate Gradient solver wrapper.

    Configures ConjugateGradientSolver with OrthogonalizationDirectionStrategy.

    Attributes:
        orthogonalization: Orthogonalization strategy (REQUIRED)
        preconditioner: Preconditioner M
        convergence_criterion: Convergence test
    """

    def __init__(
        self,
        orthogonalization: OrthogonalizationStrategy,
        preconditioner: Preconditioner | None = None,
        convergence_criterion: IConvergenceCriterion | None = None,
        iteration_history: IterationHistory | None = None,
        event_log: EventLog | None = None,
        trace_mode: TraceMode = TraceMode.MINIMAL,
    ) -> None:
        """Initialize FCG solver.

        Args:
            orthogonalization: Orthogonalization strategy (REQUIRED)
            preconditioner: Preconditioner M (default: Identity)
            convergence_criterion: Convergence test (default: Combined rtol/atol)
            iteration_history: Optional IterationHistory for continuous monitoring
            event_log: Optional EventLog for discrete event logging
            trace_mode: Logging granularity (default: MINIMAL)
        """
        self.orthogonalization = orthogonalization

        # Build strategy
        direction_strategy = OrthogonalizationDirectionStrategy(orthogonalization)

        # Initialize base solver with configured strategy
        super().__init__(
            direction_strategy=direction_strategy,
            preconditioner=preconditioner,
            convergence_criterion=convergence_criterion,
            iteration_history=iteration_history,
            event_log=event_log,
            trace_mode=trace_mode,
        )