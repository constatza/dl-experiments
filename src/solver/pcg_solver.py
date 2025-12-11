"""SciPy-backed Conjugate Gradient solver with optional residual recording."""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
import scipy.sparse.linalg as spla

from .event_logging import VectorLogger
from .info import SolverCallbackLogger, SolverResult


def _as_linear_operator(matrix: Any) -> spla.LinearOperator:
    """Coerce dense, sparse, or LinearOperator input into a LinearOperator."""
    return spla.aslinearoperator(matrix)  # type: ignore[arg-type]


def _wrap_preconditioner(
    preconditioner: Callable[[np.ndarray], np.ndarray] | spla.LinearOperator | None,
    shape: tuple[int, int],
    dtype: np.dtype,
) -> spla.LinearOperator | None:
    """Convert a preconditioner callable into a SciPy LinearOperator."""
    if preconditioner is None:
        return None
    if isinstance(preconditioner, spla.LinearOperator):
        return preconditioner

    def matvec(vec: np.ndarray) -> np.ndarray:
        return np.asarray(preconditioner(np.asarray(vec, dtype=dtype, copy=False)), dtype=dtype, copy=False)

    return spla.LinearOperator(shape, matvec=matvec, dtype=dtype)


class ResidualRecorder(SolverCallbackLogger):
    """SciPy-compatible callback that records residual norms and solutions."""

    def __init__(
        self,
        A: spla.LinearOperator,
        b: np.ndarray,
        *,
        callback_type: Literal["x", "pr_norm", None] | None = "x",
        track_solution: bool = False,
        event_log: VectorLogger | None = None,
    ) -> None:
        self.A = A
        self.b = b
        self.callback_type = callback_type
        self.track_solution = track_solution
        self.event_log = event_log
        self.rhs_norm = float(np.linalg.norm(b))
        self.b_norm = self.rhs_norm  # Alias for compatibility
        self.history_abs: list[float] = []
        self.history_rel: list[float] = []
        self.solutions: list[np.ndarray] | None = [] if track_solution else None
        self._initial_logged = False

    def __call__(self, xk_or_norm: np.ndarray | float) -> None:
        if self.callback_type == "pr_norm":
            residual_abs = float(xk_or_norm)
        else:
            xk = np.asarray(xk_or_norm, dtype=np.float64)
            residual = self.b - self.A @ xk
            residual_abs = float(np.linalg.norm(residual))
            if self.solutions is not None:
                self.solutions.append(xk.copy())
            if self.event_log is not None:
                self.event_log.log("residual", residual)
                self.event_log.log("solution", xk)
        residual_rel = residual_abs / self.rhs_norm if self.rhs_norm > 0 else residual_abs
        self.history_abs.append(residual_abs)
        self.history_rel.append(residual_rel)
        if self.event_log is not None:
            self.event_log.log("residual_norm", residual_abs)

    def log_initial_state(self, x0: np.ndarray | None) -> None:
        """Log the initial residual and solution (iteration 0).

        This must be called before scipy CG starts iterating. Scipy's callback
        only fires during iterations (1, 2, 3, ...), so we manually capture
        the initial state here.

        Args:
            x0: Initial guess. If None, assumes zero vector (r0 = b).

        Theory:
            The initial residual is:
            - r0 = b             when x0 = 0 (None)
            - r0 = b - A @ x0    when x0 is provided

            This matches the initialization in FlexibleConjugateGradientSolver
            (fcg_solver.py:91-95) for consistency.

        Examples:
            >>> # Zero initial guess
            >>> recorder = ResidualRecorder(A, b)
            >>> recorder.log_initial_state(None)
            >>> assert recorder.history_abs[0] == np.linalg.norm(b)

            >>> # Non-zero initial guess
            >>> x0 = np.random.rand(n)
            >>> recorder.log_initial_state(x0)
            >>> assert recorder.history_abs[0] == np.linalg.norm(b - A @ x0)
        """
        if self._initial_logged:
            return  # Already logged, skip

        # Compute initial residual
        if x0 is None:
            r0 = self.b.copy()
        else:
            x0 = np.asarray(x0, dtype=np.float64, copy=False)
            r0 = self.b - self.A @ x0

        # Compute norms
        r0_norm = float(np.linalg.norm(r0))
        r0_norm_rel = r0_norm / self.b_norm if self.b_norm > 0 else r0_norm

        # Append to history (iteration 0 comes first)
        self.history_abs.append(r0_norm)
        self.history_rel.append(r0_norm_rel)

        # Log to event logger if available
        if self.event_log is not None:
            self.event_log.log("residual", r0)
            self.event_log.log("residual_norm", r0_norm)
            if x0 is not None:
                self.event_log.log("solution", x0.copy())

        # Track solution history if requested
        if self.track_solution and self.solutions is not None and x0 is not None:
            self.solutions.append(x0.copy())

        self._initial_logged = True


def cg_solve(
    A: spla.LinearOperator | np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    *,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    M: spla.LinearOperator | Callable[[np.ndarray], np.ndarray] | None = None,
    callback: Callable[[np.ndarray | float], None] | None = None,
    callback_type: Literal["x", "pr_norm", None] | None = None,
    record_history: bool = True,
    track_solution: bool = False,
    event_log: VectorLogger | None = None,
) -> tuple[np.ndarray, SolverResult]:
    """Thin wrapper around scipy.sparse.linalg.cg returning SolverResult."""
    A_op = _as_linear_operator(A)
    dtype = A_op.dtype or np.dtype(np.float64)
    M_op = _wrap_preconditioner(M, A_op.shape, dtype) if M is not None else None

    recorder: ResidualRecorder | None = None
    if record_history:
        recorder = ResidualRecorder(
            A_op,
            np.asarray(b, dtype=dtype, copy=False),
            callback_type=callback_type,
            track_solution=track_solution,
            event_log=event_log,
        )

    # Log initial state (iteration 0) before scipy CG starts
    if recorder is not None:
        recorder.log_initial_state(x0)

    def _composed_callback(xk_or_norm: np.ndarray | float) -> None:
        if callback is not None:
            callback(xk_or_norm)
        if recorder is not None:
            recorder(xk_or_norm)

    applied_callback = _composed_callback if (callback is not None or recorder is not None) else None

    x, info_code = spla.cg(
        A_op,
        np.asarray(b, dtype=dtype, copy=False),
        x0=None if x0 is None else np.asarray(x0, dtype=dtype, copy=False),
        rtol=tol,
        maxiter=maxiter,
        M=M_op,
        callback=applied_callback,
        atol=atol,
    )

    final_residual = np.asarray(b, dtype=dtype, copy=False) - A_op @ x
    residual_abs = float(np.linalg.norm(final_residual))
    rhs_norm = recorder.rhs_norm if recorder is not None else float(np.linalg.norm(b))
    residual_rel = residual_abs / rhs_norm if rhs_norm > 0 else residual_abs

    if recorder is not None and recorder.history_abs and (
        recorder.callback_type is None or recorder.callback_type == "pr_norm"
    ):
        # Ensure final residual is recorded for callback types that do not include it.
        recorder.history_abs.append(residual_abs)
        recorder.history_rel.append(residual_rel)
        if recorder.event_log is not None:
            recorder.event_log.log("residual_norm", residual_abs)

    # SciPy-style info code convention:
    # - info = 0: successfully converged
    # - info > 0: did not converge, equals number of iterations performed
    # SciPy never returns negative codes, so breakdown is always False here
    iterations = info_code if info_code > 0 else (len(recorder.history_abs) if recorder else 0)
    converged = info_code == 0

    result = SolverResult(
        converged=converged,
        iterations=iterations,
        info=info_code,
        niter=iterations,
        status=info_code,
        residual=residual_rel,
        residual_abs=residual_abs,
        residual_rel=residual_rel,
        residual_history=recorder.history_rel if recorder else None,
        residual_history_abs=recorder.history_abs if recorder else None,
        rhs_norm=rhs_norm,
        breakdown=False,  # SciPy CG never reports breakdown
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        callback_type=callback_type,
        logger=recorder,
        event_log=event_log,
    )

    return x, result
