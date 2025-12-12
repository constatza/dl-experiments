"""Dataclasses and contexts for solver interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

if TYPE_CHECKING:
    from .event_logging import VectorLogger


class SolverCallbackLogger(Protocol):
    """Common interface for SciPy-compatible callbacks that also store history."""

    history_abs: list[float]
    history_rel: list[float]
    solutions: list[np.ndarray] | None

    def __call__(self, xk_or_norm: np.ndarray | float) -> None: ...


@dataclass
class SolverResult:
    """Execution result for CG-family solvers with SciPy-aligned metadata."""

    converged: bool
    iterations: int
    residual: float
    residual_abs: float
    rhs_norm: float
    breakdown: bool
    info: int = 0
    niter: int | None = None
    status: int | None = None
    residual_rel: float | None = None
    residual_history: list[float] | None = None
    residual_history_abs: list[float] | None = None
    tol: float | None = None
    atol: float | None = None
    maxiter: int | None = None
    callback_type: Literal["x", "pr_norm", None] | None = None
    logger: SolverCallbackLogger | None = None
    event_log: VectorLogger | None = None
    stopping_criterion: str | None = None
    residual_vectors: np.ndarray | None = None
    solution_vectors: np.ndarray | None = None

    def __post_init__(self) -> None:
        # Keep residual and residual_rel in sync for callers expecting either.
        if self.residual_rel is None:
            self.residual_rel = self.residual
        if self.niter is None:
            self.niter = self.iterations
        if self.status is None:
            self.status = self.info


@dataclass
class CGComparisonResult:
    """Result container for CG comparison sweeps."""

    x: np.ndarray
    converged: bool
    iterations: int
    residual: float
    residual_abs: float
    residual_history: list[float]
    residual_history_abs: list[float]
    warm_start: str
    preconditioner: str
    step_helper: str
    initial_guess: np.ndarray
    exact_error: float | None
    rhs_norm: float
    breakdown: bool
    helper_iterations: list[int]
    helper_norms: list[float]
    error: str | None = None


@dataclass(frozen=True)
class IterationContext:
    """Context passed to flexible preconditioners and helpers."""

    iteration: int
    residual: np.ndarray
    solution: np.ndarray
    matrix: np.ndarray
    rhs: np.ndarray
