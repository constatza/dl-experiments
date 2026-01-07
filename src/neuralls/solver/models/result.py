"""Result dataclasses for solver execution.

This module defines result containers returned by solvers after execution.
Includes diagnostic information, convergence status, and optional traces.

Design:
    - SolverResult: General solver result with SciPy-aligned metadata
    - CGComparisonResult: Extended result for comparison experiments
    - IterationContext: Context passed to flexible preconditioners

Theory:
    Results contain both numerical outcomes (solution, residual) and diagnostic
    information (iterations, convergence status, breakdown flags). This separation
    enables post-analysis and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..monitoring.trace_recorder import TraceRecorder


@dataclass
class SolverResult:
    """Execution result for iterative solvers with diagnostic metadata.

    This dataclass contains all information about a solver run:
    - Convergence status and final residual
    - Iteration counts and history
    - Diagnostic flags (breakdown, divergence)
    - Optional traces (residual vectors, solution vectors)

    Attributes:
        converged: Whether convergence criterion satisfied.
        iterations: Number of iterations executed.
        residual: Relative residual norm ||r||/||b|| at termination.
        residual_abs: Absolute residual norm ||r|| at termination.
        rhs_norm: Right-hand side norm ||b||.
        breakdown: Whether numerical breakdown occurred.
        info: Status code (0=success, >0=warning, <0=error).
        niter: Alias for iterations (SciPy compatibility).
        status: Alias for info (SciPy compatibility).
        residual_rel: Alias for residual (explicit naming).
        residual_history: Relative residual norms across iterations.
        residual_history_abs: Absolute residual norms across iterations.
        tol: Tolerance used (for diagnostics).
        atol: Absolute tolerance used.
        maxiter: Maximum iterations allowed.
        event_log: Optional TraceRecorder with detailed iteration traces.
        stopping_criterion: Description of why solver stopped.
        residual_vectors: Optional full residual vectors (shape: iterations x n).
        solution_vectors: Optional full solution vectors (shape: iterations x n).

    Example:
        >>> result = SolverResult(
        ...     converged=True,
        ...     iterations=10,
        ...     residual=1e-8,
        ...     residual_abs=1e-8,
        ...     rhs_norm=1.0,
        ...     breakdown=False,
        ... )
        >>> result.converged
        True
    """

    converged: bool
    """Whether convergence criterion satisfied: ||r|| <= max(rtol*||b||, atol)."""

    iterations: int
    """Number of iterations executed."""

    residual: float
    """Relative residual norm ||r||/||b|| at termination."""

    residual_abs: float
    """Absolute residual norm ||r|| at termination."""

    rhs_norm: float
    """Right-hand side norm ||b||."""

    breakdown: bool
    """Whether numerical breakdown occurred (NaN/Inf). Terminal condition."""

    info: int = 0
    """Status code: 0=success, >0=warning (max iter), <0=error (breakdown)."""

    niter: int | None = None
    """Alias for iterations (SciPy compatibility)."""

    status: int | None = None
    """Alias for info (SciPy compatibility)."""

    residual_rel: float | None = None
    """Alias for residual (explicit naming)."""

    residual_history: list[float] | None = None
    """Relative residual norms ||r_k||/||b|| across iterations."""

    residual_history_abs: list[float] | None = None
    """Absolute residual norms ||r_k|| across iterations."""

    tol: float | None = None
    """Relative tolerance used (for diagnostics)."""

    atol: float | None = None
    """Absolute tolerance used (for diagnostics)."""

    maxiter: int | None = None
    """Maximum iterations allowed."""

    event_log: TraceRecorder | None = None
    """Optional TraceRecorder with detailed iteration traces."""

    stopping_criterion: str | None = None
    """Description of why solver stopped (e.g., 'converged', 'max_iter', 'breakdown')."""

    residual_vectors: np.ndarray | None = None
    """Optional full residual vectors (shape: iterations x n). Heavy, use sparingly."""

    solution_vectors: np.ndarray | None = None
    """Optional full solution vectors (shape: iterations x n). Heavy, use sparingly."""

    def __post_init__(self) -> None:
        """Synchronize aliased fields for backward compatibility.

        Ensures that niter, status, and residual_rel are set if not provided.
        This maintains compatibility with code expecting SciPy-style fields.
        """
        if self.residual_rel is None:
            self.residual_rel = self.residual
        if self.niter is None:
            self.niter = self.iterations
        if self.status is None:
            self.status = self.info


@dataclass
class CGComparisonResult:
    """Result container for CG comparison experiments.

    Extended result class for comparison workflows that test multiple
    warm start and preconditioner combinations.

    Attributes:
        x: Final solution vector.
        converged: Whether convergence criterion satisfied.
        iterations: Number of iterations executed.
        residual: Relative residual norm at termination.
        residual_abs: Absolute residual norm at termination.
        residual_history: Relative residual norms across iterations.
        residual_history_abs: Absolute residual norms across iterations.
        warm_start: Warm start strategy name.
        preconditioner: Preconditioner strategy name.
        step_helper: Step helper strategy name (e.g., 'none', 'neural').
        initial_guess: Initial guess vector x_0.
        exact_error: Error ||x - x_exact|| if exact solution known.
        rhs_norm: Right-hand side norm ||b||.
        breakdown: Whether numerical breakdown occurred.
        helper_iterations: Iterations where step helper was invoked.
        helper_norms: Residual norms after step helper application.
        error: Error message if solver failed.

    Example:
        >>> result = CGComparisonResult(
        ...     x=np.zeros(10),
        ...     converged=True,
        ...     iterations=5,
        ...     residual=1e-8,
        ...     residual_abs=1e-8,
        ...     residual_history=[1.0, 0.1, 0.01, 0.001, 1e-8],
        ...     residual_history_abs=[1.0, 0.1, 0.01, 0.001, 1e-8],
        ...     warm_start='neural',
        ...     preconditioner='jacobi',
        ...     step_helper='none',
        ...     initial_guess=np.zeros(10),
        ...     exact_error=None,
        ...     rhs_norm=1.0,
        ...     breakdown=False,
        ...     helper_iterations=[],
        ...     helper_norms=[],
        ... )
        >>> result.warm_start
        'neural'
    """

    x: np.ndarray
    """Final solution vector."""

    converged: bool
    """Whether convergence criterion satisfied."""

    iterations: int
    """Number of iterations executed."""

    residual: float
    """Relative residual norm at termination."""

    residual_abs: float
    """Absolute residual norm at termination."""

    residual_history: list[float]
    """Relative residual norms across iterations."""

    residual_history_abs: list[float]
    """Absolute residual norms across iterations."""

    warm_start: str
    """Warm start strategy name (e.g., 'none', 'neural', 'precond')."""

    preconditioner: str
    """Preconditioner strategy name (e.g., 'none', 'jacobi', 'neural')."""

    step_helper: str
    """Step helper strategy name (e.g., 'none', 'neural')."""

    initial_guess: np.ndarray
    """Initial guess vector x_0."""

    exact_error: float | None
    """Error ||x - x_exact|| if exact solution known."""

    rhs_norm: float
    """Right-hand side norm ||b||."""

    breakdown: bool
    """Whether numerical breakdown occurred."""

    helper_iterations: list[int]
    """Iterations where step helper was invoked."""

    helper_norms: list[float]
    """Residual norms after step helper application."""

    error: str | None = None
    """Error message if solver failed."""


@dataclass(frozen=True)
class IterationContext:
    """Context passed to flexible preconditioners and step helpers.

    Provides iteration-specific information to preconditioners and helpers
    that need access to current solver state beyond just the residual.

    Attributes:
        iteration: Current iteration number (0-indexed).
        residual: Current residual vector r_k.
        solution: Current solution vector x_k.
        matrix: System matrix A (optional, may be expensive to store).
        rhs: Right-hand side vector b.

    Example:
        >>> import numpy as np
        >>> ctx = IterationContext(
        ...     iteration=5,
        ...     residual=np.ones(10),
        ...     solution=np.zeros(10),
        ...     matrix=np.eye(10),
        ...     rhs=np.ones(10),
        ... )
        >>> ctx.iteration
        5

    Usage:
        Context-aware preconditioners can use iteration info for adaptive behavior:

        >>> def adaptive_precond(r: np.ndarray, ctx: IterationContext) -> np.ndarray:
        ...     if ctx.iteration < 10:
        ...         # Use simple preconditioner early
        ...         return r / np.diag(ctx.matrix)
        ...     else:
        ...         # Use expensive preconditioner later
        ...         return neural_network(r, ctx.solution)
    """

    iteration: int
    """Current iteration number (0-indexed)."""

    residual: np.ndarray
    """Current residual vector r_k = b - A*x_k."""

    solution: np.ndarray
    """Current solution vector x_k."""

    matrix: np.ndarray
    """System matrix A (optional, may be expensive to store)."""

    rhs: np.ndarray
    """Right-hand side vector b."""
