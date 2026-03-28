"""Default collaborators for generation workflows."""

from __future__ import annotations

from typing import Any

import numpy as np

from neuralls.domain.generation.ports import TracingSolverPort
from neuralls.domain.solver import pcg
from neuralls.domain.solver.monitoring.iteration_history import IterationHistory
from neuralls.domain.solver.monitoring.trace_mode import TraceMode
from neuralls.domain.solver.scipy_wrapper import SciPyCGSolver


def make_residual_solver() -> TracingSolverPort:
    """Construct the default tracing solver for residual trace strategies."""

    def _solve(
        A: np.ndarray,
        b: np.ndarray,
        x0: np.ndarray,
        *,
        maxiter: int,
        rtol: float,
        atol: float,
    ) -> tuple[np.ndarray, Any]:
        history = IterationHistory(mode=TraceMode.FULL)
        solver = SciPyCGSolver(iteration_history=history)
        return solver.solve(
            A=A,
            b=b,
            x0=x0,
            maxiter=maxiter,
            rtol=rtol,
            atol=atol,
            trace_mode="full",
        )

    return _solve


def make_direction_solver() -> TracingSolverPort:
    """Construct the default tracing solver for search direction strategies."""

    def _solve(
        A: np.ndarray,
        b: np.ndarray,
        x0: np.ndarray,
        *,
        maxiter: int,
        rtol: float,
        atol: float,
    ) -> tuple[np.ndarray, Any]:
        return pcg(
            A=A,
            b=b,
            x0=x0,
            maxiter=maxiter,
            rtol=rtol,
            atol=atol,
            trace_mode=TraceMode.FULL,
        )

    return _solve
