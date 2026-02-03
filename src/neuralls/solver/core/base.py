"""Base template for iterative linear system solvers.

This module provides the abstract base class for iterative solvers using the
Template Method pattern. It defines the lifecycle of solving a linear system
while allowing subclasses to customize specific steps.

Design Principles:
    - Template Method: solve() orchestrates the algorithm flow
    - Open/Closed: Open for extension (subclass hooks), closed for modification
    - Interface Segregation: Minimal public interface, rich protected hooks
    - Dependency Inversion: Depends on Preconditioner and ConvergenceCriterion abstractions

References:
    - Gamma et al. (1994): Design Patterns - Template Method
    - Martin, R. (2017): Clean Architecture - SOLID Principles
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np

from ..monitoring.trace_mode import TraceMode

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..strategies.convergence import IConvergenceCriterion
    from ..models.result import SolverResult
    from ..preconditioners import Preconditioner
    from ..models.state import SolverState
    from ..monitoring.trace_recorder import TraceRecorder


class IterativeSolverBase(ABC):
    """Abstract base class for iterative solvers using template method pattern.

    All subclasses MUST define a convergence_criterion attribute in their __init__ method.

    Attributes:
        convergence_criterion: Required convergence criterion for checking solution convergence.
            This must be set in the subclass __init__ method. The criterion defines
            the norm and tolerance logic for determining when the solver has converged.
        event_log: Optional TraceRecorder for logging iteration events
        trace_mode: Controls logging granularity (DISABLED, MINIMAL, FULL)

    This class implements the common lifecycle for iterative linear system solvers:
    1. Validate inputs
    2. Prepare operator
    3. Initialize state
    4. Iterate until convergence/breakdown/max iterations
    5. Build result

    Subclasses implement protected hooks to customize each step while the public
    solve() method orchestrates the overall flow.

    Public Methods:
        solve: Solve linear system Ax = b (orchestrates template)
        check_convergence: Check if convergence criterion satisfied

    Protected Hooks (Abstract - must implement):
        _initialize_state: Setup initial IterationState
        _iterate_step: Execute single solver iteration
        _check_stopping: Check convergence/breakdown/divergence
        _build_result: Assemble final SolverResult
        _log_state: Log iteration state via TraceRecorder

    Protected Hooks (Concrete - available to all):
        _validate_system: Validate inputs (dimensions, types)
        _prepare_operator: Wrap A in consistent callable interface

    Protected Helpers (Concrete - available to all):
        _apply_preconditioner: Apply preconditioner to residual

    Mathematical Background:
        Iterative solvers generate sequence x_0, x_1, ... converging to solution.
        Convergence when ||r_k|| = ||b - Ax_k|| <= max(rtol * ||b||, atol).
    """

    def __init__(
        self,
        *,
        event_log: TraceRecorder | None = None,
        trace_mode: TraceMode = TraceMode.MINIMAL,
    ) -> None:
        """Initialize base solver with logging configuration.

        Args:
            event_log: Optional TraceRecorder for logging iteration events
            trace_mode: Controls logging granularity (default: MINIMAL)
        """
        self.event_log = event_log
        self.trace_mode = trace_mode

    def solve(
        self,
        A: Callable[[NDArray], NDArray] | NDArray,
        b: NDArray,
        x0: NDArray | None = None,
        *,
        rtol: float | None = None,
        atol: float | None = None,
        maxiter: int | None = None,
        breakdown_tol: float | None = None,
    ) -> tuple[NDArray, SolverResult]:
        """Solve linear system Ax = b using iterative method.

        Template method that orchestrates the solver lifecycle. Subclasses
        customize behavior by implementing protected hook methods.

        Logging configuration (event_log, trace_mode) is set via constructor,
        not as parameters to solve().

        Args:
            A: Linear operator mapping vectors to vectors. Can be:
                - Matrix (will be wrapped)
                - Callable taking vector, returning vector
                - SciPy LinearOperator
            b: Right-hand side vector, shape (n,)
            x0: Initial guess, shape (n,). If None, uses zero vector.
            rtol: Relative tolerance (default: DEFAULT_RTOL from constants)
            atol: Absolute tolerance (default: DEFAULT_ATOL from constants)
            maxiter: Maximum iterations (default: 100)
            breakdown_tol: Breakdown detection tolerance (default: DEFAULT_BREAKDOWN_TOL)

        Returns:
            Tuple of (solution, result):
            - solution: Solution vector x, shape (n,)
            - result: SolverResult with convergence info, history, diagnostics

        Raises:
            ValueError: If inputs invalid (shape mismatch, etc.)

        Theory:
            This implements the general iterative solver template:
                Given: A, b, x_0, tolerance
                While not converged:
                    x_{k+1} = update(x_k, A, b)
                Return x_k, diagnostics
        """
        from ...constants import DEFAULT_RTOL, DEFAULT_ATOL, DEFAULT_BREAKDOWN_TOL

        # Step 1: Resolve defaults
        rtol_eff = rtol if rtol is not None else DEFAULT_RTOL
        atol_eff = atol if atol is not None else DEFAULT_ATOL
        maxiter_eff = maxiter if maxiter is not None else 100
        breakdown_tol_eff = breakdown_tol if breakdown_tol is not None else DEFAULT_BREAKDOWN_TOL

        # Step 2: Validate inputs
        self._validate_system(A, b, x0)

        # Step 3: Prepare operator (ensure consistent callable interface)
        linear_op = self._prepare_operator(A, b)

        # Step 4: Initialize iteration state
        state = self._initialize_state(linear_op, b, x0, maxiter=maxiter_eff)

        # Step 5: Log initial state (iteration 0)
        self._log_state(state, iteration=0)

        # Step 6: Iterate until stopping criterion met
        while not self._check_stopping(state, rtol_eff, atol_eff, maxiter_eff, breakdown_tol=breakdown_tol_eff):
            state = self._iterate_step(linear_op, state, breakdown_tol=breakdown_tol_eff)
            self._log_state(state, iteration=state.iteration)

        # Step 7: Build final result
        u_final = getattr(state, "u", None)
        if u_final is None:
            raise RuntimeError("Final solution not found in state")

        return u_final, self._build_result(state, rtol_eff, atol_eff, breakdown_tol=breakdown_tol_eff)

    def _create_convergence_criterion(
        self, rtol: float, atol: float
    ) -> IConvergenceCriterion:
        """Create convergence criterion (dependency injection helper).

        Args:
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            Convergence criterion instance
        """
        from ..strategies.convergence import CombinedToleranceCriterion

        return CombinedToleranceCriterion(rtol=rtol, atol=atol)

    def _check_convergence_with_criterion(
        self,
        state: SolverState,
        criterion: IConvergenceCriterion,
    ) -> bool:
        """Check convergence using injected criterion (internal helper).

        Args:
            state: Current solver state
            criterion: Convergence criterion to use

        Returns:
            True if converged, False otherwise
        """
        from ..models.protocols import HasVectors

        if isinstance(state, HasVectors):
            return self.check_convergence(state.r, state.rhs_norm, criterion)
        return False

    def check_convergence(
        self,
        residual: NDArray,
        rhs_norm: float,
        criterion: IConvergenceCriterion,
    ) -> bool:
        """Check convergence criterion (delegates to criterion strategy).

        Args:
            residual: Current residual vector r_k
            rhs_norm: Norm of right-hand side ||b||
            criterion: Convergence criterion implementation

        Returns:
            True if converged, False otherwise
        """
        return criterion.has_converged(residual, rhs_norm)

    # Protected abstract hooks

    def _validate_system(
        self,
        A: Callable[[NDArray], NDArray] | NDArray,
        b: NDArray,
        x0: NDArray | None,
    ) -> None:
        """Validate system inputs.

        Args:
            A: Linear operator
            b: Right-hand side
            x0: Initial guess

        Raises:
            ValueError: If validation fails
        """
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
    ) -> Callable[[NDArray], NDArray]:
        """Prepare linear operator A.

        Wraps A in consistent callable interface, handling:
        - Dense/sparse matrices
        - SciPy LinearOperator
        - User callables

        Args:
            A: Linear operator (various forms)
            b: Right-hand side (for shape inference)

        Returns:
            Callable taking vector, returning Ax
        """
        from scipy.sparse.linalg import aslinearoperator

        # If already callable, return as is
        if callable(A) and not isinstance(A, np.ndarray):
            return A  # type: ignore[return-value]

        # Convert to LinearOperator for consistency
        linear_op = aslinearoperator(A)
        return linear_op.matvec

    @abstractmethod
    def _initialize_state(
        self,
        linear_op: Callable[[NDArray], NDArray],
        b: NDArray,
        x0: NDArray | None,
        maxiter: int | None = None,
    ) -> SolverState:
        """Initialize solver state for iteration 0.

        Args:
            linear_op: Prepared linear operator
            b: Right-hand side
            x0: Initial guess (None means zero)
            maxiter: Maximum iterations (used for history sizing)

        Returns:
            Initial SolverState with:
                - x, r initialized
                - iteration = 0
                - converged = False
                - histories empty
        """
        ...

    @abstractmethod
    def _iterate_step(
        self,
        linear_op: Callable[[NDArray], NDArray],
        state: SolverState,
        breakdown_tol: float | None = None,
    ) -> SolverState:
        """Execute single solver iteration.

        Args:
            linear_op: Linear operator
            state: Current iteration state
            breakdown_tol: Breakdown detection tolerance

        Returns:
            Updated SolverState for next iteration
        """
        ...

    @abstractmethod
    def _check_stopping(
        self,
        state: SolverState,
        rtol: float,
        atol: float,
        maxiter: int,
        breakdown_tol: float | None = None,
    ) -> bool:
        """Check if solver should stop.

        Checks for:
        - Convergence: ||r|| <= max(rtol * ||b||, atol)
        - Breakdown: NaN/Inf detected
        - Max iterations reached

        Args:
            state: Current iteration state
            rtol: Relative tolerance
            atol: Absolute tolerance
            maxiter: Maximum iterations
            breakdown_tol: Breakdown detection tolerance

        Returns:
            True if should stop, False if should continue
        """
        ...

    @abstractmethod
    def _build_result(
        self,
        state: SolverState,
        rtol: float,
        atol: float,
        breakdown_tol: float | None = None,
    ) -> SolverResult:
        """Build final SolverResult.

        Args:
            state: Final iteration state
            rtol: Relative tolerance used
            atol: Absolute tolerance used
            breakdown_tol: Breakdown tolerance used

        Returns:
            SolverResult with convergence info, histories, diagnostics
        """
        ...

    def _log_state(
        self,
        state: SolverState,
        iteration: int,
    ) -> None:
        """Log iteration state via TraceRecorder (default implementation).

        This default implementation logs:
        - Scalars (always): iteration, residual_norm, converged, breakdown
        - Vectors (if FULL mode): residual, solution, direction, matrix_product, precond_residual

        Subclasses can override to add custom events (e.g., REORTHOGONALIZED in FCG)
        or to exclude specific vectors.

        Args:
            state: Current iteration state
            iteration: Iteration number (0-indexed)

        Theory:
            Separates logging concerns from solver algorithm. Event sourcing
            captures all state changes for post-hoc analysis without coupling
            to algorithm logic.
        """
        from ..monitoring.events import EventType
        from ..monitoring.trace_mode import TraceMode
        from ..models.protocols import HasVectors

        if self.event_log is None:
            return

        # Always log scalars (MINIMAL mode)
        self.event_log.log(EventType.ITERATION, iteration)
        self.event_log.log(EventType.RESIDUAL_NORM, state.residual_norm)
        self.event_log.log(EventType.CONVERGED, state.converged)
        self.event_log.log(EventType.BREAKDOWN, state.breakdown)

        # Log vectors if FULL trace mode
        if self.trace_mode == TraceMode.FULL and isinstance(state, HasVectors):
            self.event_log.log(EventType.RESIDUAL, state.r)
            self.event_log.log(EventType.SOLUTION, state.u)

            # Log Krylov vectors if available
            if hasattr(state, "p"):
                self.event_log.log(EventType.DIRECTION, state.p)
            if hasattr(state, "q"):
                self.event_log.log(EventType.MATRIX_PRODUCT, state.q)
            if hasattr(state, "z"):
                self.event_log.log(EventType.PRECOND_RESIDUAL, state.z)

    # Protected concrete helpers

    def _apply_preconditioner(
        self,
        preconditioner: Preconditioner,
        residual: NDArray,
    ) -> NDArray:
        """Apply general preconditioner to residual.

        This is a concrete helper available to ALL iterative solvers.
        Different methods (CG, GMRES, BiCGSTAB) all apply preconditioners,
        though the specific point in the iteration may vary.

        Can handle linear preconditioners (M^{-1}) or non-linear approximations
        (neural networks, adaptive strategies).

        Args:
            preconditioner: Preconditioner operator (linear or non-linear)
            residual: Residual vector to precondition

        Returns:
            Preconditioned residual z = M^{-1}r (or non-linear approximation)

        Raises:
            ValueError: If preconditioner output shape doesn't match input
        """
        z = preconditioner.apply(residual)

        # Validate shape/dtype
        if z.shape != residual.shape:
            raise ValueError(
                f"Preconditioner output shape {z.shape} != input {residual.shape}"
            )

        # Ensure float64 dtype
        return np.asarray(z, dtype=np.float64)


    def _extract_histories_from_event_log(
        self,
        event_log,
        rhs_norm: float,
    ) -> tuple[list[float] | None, list[float] | None, NDArray | None, NDArray | None]:
        """Extract history arrays from event_log for SolverResult.

        Args:
            event_log: TraceRecorder containing logged events
            rhs_norm: Right-hand side norm for computing relative residuals

        Returns:
            Tuple of:
            - residual_history_abs: Absolute residual norms ||r_k||
            - residual_history_rel: Relative residual norms ||r_k|| / ||b||
            - residual_vectors: Stacked residual vectors (if FULL mode)
            - solution_vectors: Stacked solution vectors (if FULL mode)

        Theory:
            Post-processes event log into structured history arrays for result.
            Computes relative residuals by dividing absolute norms by ||b||.
            Handles missing data gracefully (returns None).
        """
        from ..monitoring.events import EventType

        if event_log is None:
            return None, None, None, None

        # Extract scalar residual norms
        residual_history_abs = event_log.get_scalars(EventType.RESIDUAL_NORM)
        if not residual_history_abs:
            residual_history_abs = None

        # Compute relative residuals
        residual_history_rel = None
        if residual_history_abs is not None and rhs_norm > 0:
            residual_history_rel = [r / rhs_norm for r in residual_history_abs]
        elif residual_history_abs is not None:
            residual_history_rel = residual_history_abs

        # Extract vectors if logged (trace_mode=FULL)
        residual_vectors = None
        if event_log.has_event(EventType.RESIDUAL):
            residual_vectors = event_log.get_vectors(EventType.RESIDUAL)

        solution_vectors = None
        if event_log.has_event(EventType.SOLUTION):
            solution_vectors = event_log.get_vectors(EventType.SOLUTION)

        return (
            residual_history_abs,
            residual_history_rel,
            residual_vectors,
            solution_vectors,
        )

