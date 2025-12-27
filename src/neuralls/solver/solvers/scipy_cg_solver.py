"""SciPy CG solver with callback-based monitoring.

This module provides SciPyCGSolver, which wraps scipy.sparse.linalg.cg and uses
SciPyCallbackAdapter to capture iteration data via callbacks.

Design Principles:
    - Wrapper Pattern: Delegates to scipy.sparse.linalg.cg
    - Callback Integration: Uses SciPyCallbackAdapter for monitoring
    - Single Responsibility: Only wraps scipy CG, no custom algorithm
    - Composition: Uses HistoryTracker and TraceRecorder via dependency injection

Use Cases:
    - Comparison workflows requiring scipy CG baseline
    - Validation against scipy reference implementation
    - When scipy's CG implementation is preferred over custom

References:
    - scipy.sparse.linalg.cg documentation
    - Hestenes & Stiefel (1952). Methods of Conjugate Gradients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import LinearOperator, aslinearoperator, cg

from ..models.result import SolverResult
from ..monitoring.callbacks import InitialStateComputer, SciPyCallbackAdapter
from ..monitoring.events import EventType
from ..monitoring.history_tracker import HistoryTracker
from ..monitoring.trace_mode import TraceMode
from ..preconditioners import IdentityPreconditioner, Preconditioner

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..monitoring.trace_recorder import TraceRecorder


class SciPyCGSolver:
    """Wrapper for scipy.sparse.linalg.cg with callback monitoring.

    This solver delegates to scipy's CG implementation while capturing
    iteration data via callbacks. It provides the same interface as
    custom solvers but uses scipy under the hood.

    Attributes:
        preconditioner: Preconditioner strategy (M^{-1} operator).
        event_logger: Optional TraceRecorder for iteration logging.

    Example:
        >>> from neuralls.solver.preconditioners import IdentityPreconditioner
        >>> from neuralls.solver.monitoring.trace_recorder import TraceRecorder
        >>>
        >>> precond = IdentityPreconditioner()
        >>> logger = TraceRecorder()
        >>> solver = SciPyCGSolver(preconditioner=precond, event_logger=logger)
        >>>
        >>> x, result = solver.solve(A, b, rtol=1e-6)
        >>> print(f"Converged: {result.converged}, Iterations: {result.iterations}")
    """

    def __init__(
        self,
        preconditioner: Preconditioner | None = None,
        event_logger: TraceRecorder | None = None,
    ) -> None:
        """Initialize scipy CG solver.

        Args:
            preconditioner: Preconditioner M (default: Identity).
            event_logger: Optional TraceRecorder for iteration logging.

        Example:
            >>> solver = SciPyCGSolver()  # Identity preconditioner, no logging
            >>> solver = SciPyCGSolver(event_logger=TraceRecorder())  # With logging
        """
        self.preconditioner = preconditioner or IdentityPreconditioner()
        self.event_logger = event_logger

    def solve(
        self,
        A: NDArray | Callable[[NDArray], NDArray],
        b: NDArray,
        x0: NDArray | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-14,
        max_iterations: int = 100,
        trace_mode: TraceMode | str = TraceMode.MINIMAL,
        **kwargs,
    ) -> tuple[NDArray, SolverResult]:
        """Solve Ax = b using scipy.sparse.linalg.cg with callback monitoring.

        Args:
            A: System matrix or linear operator (n × n, SPD).
            b: Right-hand side vector (n,).
            x0: Initial guess (n,). If None, uses zero vector.
            rtol: Relative tolerance for convergence.
            atol: Absolute tolerance for convergence.
            max_iterations: Maximum number of iterations.
            trace_mode: Logging granularity (MINIMAL or FULL).
            **kwargs: Additional parameters (event_log override).

        Returns:
            Tuple (x, result):
                - x: Solution vector (n,).
                - result: SolverResult with convergence info and diagnostics.

        Example:
            >>> x, result = solver.solve(A, b, rtol=1e-6, max_iterations=100)
            >>> print(f"Residual: {result.residual}, Iterations: {result.iterations}")
        """
        # Convert trace_mode string to enum
        if isinstance(trace_mode, str):
            trace_mode = TraceMode(trace_mode)

        # Prepare linear operator
        A_op = aslinearoperator(A)
        b_arr = np.asarray(b, dtype=np.float64)

        # Prepare preconditioner as LinearOperator
        M_op = self._prepare_preconditioner(A_op.shape)

        # Get event logger from kwargs or instance
        event_log = kwargs.get("event_log", self.event_logger)

        # Create history tracker
        tracker = HistoryTracker()
        if trace_mode == TraceMode.FULL:
            tracker.enable_solution_tracking()

        # Compute initial residual and log iteration 0
        x_init, r_init, r0_norm = InitialStateComputer.compute_initial_residual(
            A_op, b_arr, x0
        )

        # Compute RHS norm for relative residuals
        rhs_norm = float(norm(b_arr))

        # Log initial state (iteration 0)
        r0_norm_rel = r0_norm / rhs_norm if rhs_norm > 0 else r0_norm
        tracker.record_residual(r0_norm, r0_norm_rel)

        if trace_mode == TraceMode.FULL:
            tracker.record_solution(x_init)

        if event_log is not None:
            event_log.log(EventType.ITERATION, 0)
            event_log.log(EventType.RESIDUAL_NORM, r0_norm)
            event_log.log(EventType.CONVERGED, False)
            event_log.log(EventType.BREAKDOWN, False)

            if trace_mode == TraceMode.FULL:
                event_log.log(EventType.RESIDUAL, r_init)
                event_log.log(EventType.SOLUTION, x_init)

        # Create callback adapter
        callback = SciPyCallbackAdapter(
            A=A_op,
            b=b_arr,
            callback_type="x",  # Capture solution vectors
            tracker=tracker,
            event_log=event_log,
        )

        # Solve using scipy CG
        # scipy convergence: ||r|| <= max(atol, rtol * ||b||)
        # which matches our convention exactly
        x_sol, info_code = cg(
            A=A_op,
            b=b_arr,
            x0=x_init,
            rtol=rtol,  # Relative tolerance
            atol=atol,  # Absolute tolerance
            maxiter=max_iterations,
            M=M_op,
            callback=callback,
        )

        # Build result
        result = self._build_result(
            x_sol=x_sol,
            info_code=info_code,
            tracker=tracker,
            event_log=event_log,
            rtol=rtol,
            atol=atol,
            rhs_norm=rhs_norm,
        )

        return x_sol, result

    def _prepare_preconditioner(
        self, shape: tuple[int, int]
    ) -> LinearOperator | None:
        """Convert preconditioner to scipy LinearOperator.

        Args:
            shape: Matrix shape (n, n).

        Returns:
            LinearOperator for preconditioner, or None for identity.

        Theory:
            scipy.sparse.linalg.cg expects preconditioner as LinearOperator
            with matvec(v) -> M^{-1} v operation.
        """
        if isinstance(self.preconditioner, IdentityPreconditioner):
            return None  # scipy handles identity natively

        def matvec(v: NDArray) -> NDArray:
            return self.preconditioner.apply(v)

        return LinearOperator(shape=shape, matvec=matvec, dtype=np.float64)

    def _build_result(
        self,
        x_sol: NDArray,
        info_code: int,
        tracker: HistoryTracker,
        event_log: TraceRecorder | None,
        rtol: float,
        atol: float,
        rhs_norm: float,
    ) -> SolverResult:
        """Build SolverResult from scipy output and tracked data.

        Args:
            x_sol: Final solution vector from scipy.
            info_code: scipy CG info code (0=success, >0=max iter, <0=error).
            tracker: HistoryTracker with recorded residuals.
            event_log: Optional TraceRecorder with logged events.
            rtol: Relative tolerance used.
            atol: Absolute tolerance used.
            rhs_norm: Norm of RHS vector ||b||.

        Returns:
            SolverResult with all diagnostic information.

        Theory:
            scipy CG info codes:
            - 0: Convergence achieved
            - >0: Max iterations reached (not converged)
            - <0: Illegal input or breakdown
        """
        # Extract residual histories from tracker
        residual_history_abs = tracker.residual_norms_abs
        residual_history_rel = tracker.residual_norms_rel

        # Extract vectors if logged (FULL mode)
        residual_vectors = None
        solution_vectors = None

        if event_log is not None:
            if event_log.has_event(EventType.RESIDUAL):
                residual_vectors = event_log.get_vectors(EventType.RESIDUAL)

            if event_log.has_event(EventType.SOLUTION):
                solution_vectors = event_log.get_vectors(EventType.SOLUTION)

        # Determine convergence
        converged = info_code == 0
        iterations = len(residual_history_abs) - 1  # Exclude iteration 0

        # Get final residuals
        final_residual_abs = residual_history_abs[-1] if residual_history_abs else 0.0
        final_residual_rel = residual_history_rel[-1] if residual_history_rel else 0.0

        return SolverResult(
            converged=converged,
            iterations=iterations,
            residual=final_residual_rel,
            residual_abs=final_residual_abs,
            rhs_norm=rhs_norm,
            breakdown=(info_code < 0),
            residual_history=residual_history_rel,
            residual_history_abs=residual_history_abs,
            tol=rtol,
            atol=atol,
            event_log=event_log,
            residual_vectors=residual_vectors,
            solution_vectors=solution_vectors,
            info=info_code,
            niter=iterations,  # scipy alias
            status=info_code,  # scipy alias
        )
