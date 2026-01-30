"""Factory functions for creating CG solver variants.

This module provides simple, high-level factory functions for constructing
different Conjugate Gradient solver configurations. Each factory encapsulates
the appropriate strategy choices and sensible defaults for common use cases.

Design Pattern: Factory Pattern
    Instead of requiring users to manually assemble strategies and solvers,
    these factories provide convenient entry points with clear semantic meaning:

    1. flexible_cg() - FCG with Gram-Schmidt orthogonalization
    2. preconditioned_cg() - PCG with two-term recurrence

Mathematical Context:
    flexible_cg:
        Best for neural preconditioners or highly non-SPD M_k.
        Uses truncated Gram-Schmidt to maintain approximate A-conjugacy.

    preconditioned_cg:
        Best for classical preconditioners (Jacobi, ILU, etc.).
        Uses two-term recurrence with minimal memory footprint.

References:
    - Notay (2000): "Flexible Conjugate Gradients"
    - Saad (2003): "Iterative Methods for Sparse Linear Systems"
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from scipy.sparse.linalg import LinearOperator

from .monitoring.trace_recorder import TraceRecorder
from .monitoring.trace_mode import TraceMode
from .solvers.fcg_solver import FlexibleCGSolver
from .solvers.pcg_solver import PreconditionedCGSolver
from .models.result import SolverResult
from .strategies.orthogonalization import create_fcg_orthogonalization
from .strategies.convergence import CombinedToleranceCriterion
from .strategies.norms import Norm, euclidean_norm
from .preconditioners import (
    FunctionPreconditioner,
    IdentityPreconditioner,
    LinearOperatorPreconditioner,
)
from ..constants import (
    DEFAULT_ATOL,
    DEFAULT_M_MAX,
    DEFAULT_RTOL,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def flexible_cg(
    A: NDArray,
    b: NDArray,
    x0: NDArray | None = None,
    *,
    preconditioner: Callable[..., NDArray] | LinearOperator | None = None,
    m_max: int = DEFAULT_M_MAX,
    norm: Norm = euclidean_norm,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    maxiter: int | None = None,
    trace_mode: TraceMode = TraceMode.MINIMAL,
) -> tuple[NDArray, SolverResult]:
    """Solve linear system using Flexible CG with truncated orthogonalization.

    This is the target algorithm for neural preconditioning and non-SPD
    preconditioners. It uses explicit Gram-Schmidt orthogonalization against
    a sliding window of previous search directions.

    Mathematical Method:
        At each iteration k:
        1. Apply preconditioner: z_k = M_k^{-1} r_k
        2. Orthogonalize: p_k = z_k - Σ[(z_k, q_j)/(p_j, q_j)] p_j
        3. Update solution: x_{k+1} = x_k + α_k p_k
        4. Update residual: r_{k+1} = r_k - α_k q_k

    Args:
        A: System matrix, shape (n, n). Must be square and SPD.
        b: Right-hand side vector, shape (n,).
        x0: Initial guess, shape (n,). If None, uses zero vector.
        preconditioner: Callable implementing M^{-1} operation.
            If None, uses identity (no preconditioning).
        m_max: Maximum history length for orthogonalization. Default: 10.
        norm: Norm function for convergence checking. Default: euclidean_norm.
            Use energy_norm(A) for A-norm convergence criterion.
        rtol: Relative tolerance for convergence. Default: 1e-6.
        atol: Absolute tolerance for convergence. Default: 1e-14.
        tol: Alternative name for rtol (overrides rtol if provided).
        maxiter: Maximum number of iterations. Default: 100.
        trace_mode: Tracing granularity level. Default: MINIMAL.
            MINIMAL: Log scalars only (iteration, residual_norm, flags)
            FULL: Log scalars + vectors (residual, solution, direction)
            DISABLED: No logging (event_log will be None)

    Returns:
        Tuple of (x, result):
        - x (ndarray): Solution vector, shape (n,)
        - result (SolverResult): Convergence information

    Examples:
        >>> # Neural preconditioner
        >>> x, result = flexible_cg(A, b, preconditioner=neural_precond)

        >>> # High-accuracy solve
        >>> x, result = flexible_cg(A, b, rtol=1e-10, m_max=20)

        >>> # A-norm convergence criterion
        >>> from neuralls.solver.strategies.norms import energy_norm
        >>> x, result = flexible_cg(A, b, norm=energy_norm(A), rtol=1e-6)
    """
    # Create preconditioner strategy
    if preconditioner is None:
        precond_strategy = IdentityPreconditioner()
    elif isinstance(preconditioner, LinearOperator):
        precond_strategy = LinearOperatorPreconditioner(preconditioner)
    else:
        precond_strategy = FunctionPreconditioner(preconditioner)

    # Create orthogonalization strategy via factory
    orthog_strategy = create_fcg_orthogonalization(m_max=m_max)

    # Create event logger based on trace mode
    # Convert string to TraceMode if needed
    trace_mode_enum = (
        TraceMode(trace_mode) if isinstance(trace_mode, str) else trace_mode
    )
    logger = TraceRecorder() if trace_mode_enum != TraceMode.DISABLED else None

    # Resolve rtol before constructing criterion

    # Create convergence criterion with injected norm
    convergence_criterion = CombinedToleranceCriterion(
        rtol=rtol,
        atol=atol,
        norm=norm,
    )

    # Construct solver
    solver = FlexibleCGSolver(
        preconditioner=precond_strategy,
        orthogonalization=orthog_strategy,
        convergence_criterion=convergence_criterion,
        event_logger=logger,
    )

    # Solve system
    # Type ignore needed because solve() accepts both NDArray and Callable
    x, result = solver.solve(  # type: ignore[arg-type]
        A,
        b,
        x0,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        trace_mode=trace_mode_enum,
        event_log=logger,
    )

    return x, result


def preconditioned_cg(
    A: NDArray,
    b: NDArray,
    x0: NDArray | None = None,
    *,
    preconditioner: Callable[..., NDArray] | LinearOperator | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    tol: float | None = None,
    maxiter: int | None = None,
    beta_formula: str = "fletcher_reeves",
    trace_mode: TraceMode = TraceMode.MINIMAL,
) -> tuple[NDArray, SolverResult]:
    """Solve linear system using Preconditioned CG with two-term recurrence.

    Standard PCG using two-term recurrence formula. Memory-efficient with
    O(1) storage for directions. Best for classical SPD preconditioners.

    Mathematical Method:
        At each iteration k:
        1. Apply preconditioner: z_k = M^{-1} r_k
        2. Compute beta: β_k = (r_k, z_k) / (r_{k-1}, z_{k-1})
        3. Update direction: p_k = z_k + β_k p_{k-1}
        4. Update solution: x_{k+1} = x_k + α_k p_k
        5. Update residual: r_{k+1} = r_k - α_k q_k

    Args:
        A: System matrix, shape (n, n). Must be square and SPD.
        b: Right-hand side vector, shape (n,).
        x0: Initial guess, shape (n,). If None, uses zero vector.
        preconditioner: Callable implementing M^{-1} operation.
            If None, uses identity (standard CG).
        rtol: Relative tolerance for convergence. Default: 1e-6.
        atol: Absolute tolerance for convergence. Default: 1e-14.
        tol: Alternative name for rtol (overrides rtol if provided).
        maxiter: Maximum number of iterations. Default: 100.
        beta_formula: Beta formula ("fletcher_reeves", "polak_ribiere"). Default: "fletcher_reeves".
        trace_mode: Tracing granularity level. Default: MINIMAL.
            MINIMAL: Log scalars only (iteration, residual_norm, flags)
            FULL: Log scalars + vectors (residual, solution, direction)
            DISABLED: No logging (event_log will be None)

    Returns:
        Tuple of (x, result):
        - x (ndarray): Solution vector, shape (n,)
        - result (SolverResult): Convergence information

    Examples:
        >>> # Jacobi preconditioner
        >>> x, result = preconditioned_cg(A, b, preconditioner=jacobi_precond)

        >>> # Standard CG (no preconditioning)
        >>> x, result = preconditioned_cg(A, b)
    """
    # Create preconditioner strategy
    if preconditioner is None:
        precond_strategy = IdentityPreconditioner()
    elif isinstance(preconditioner, LinearOperator):
        precond_strategy = LinearOperatorPreconditioner(preconditioner)
    else:
        precond_strategy = FunctionPreconditioner(preconditioner)

    # Create event logger based on trace mode
    # Convert string to TraceMode if needed
    trace_mode_enum = (
        TraceMode(trace_mode) if isinstance(trace_mode, str) else trace_mode
    )
    logger = TraceRecorder() if trace_mode_enum != TraceMode.DISABLED else None

    # Construct solver
    solver = PreconditionedCGSolver(
        preconditioner=precond_strategy,
        beta_formula=beta_formula,
        event_logger=logger,
    )

    # Solve system
    # Type ignore needed because solve() accepts both NDArray and Callable
    x, result = solver.solve(  # type: ignore[arg-type]
        A,
        b,
        x0,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        trace_mode=trace_mode_enum,
        event_log=logger,
    )

    return x, result


def scipy_cg(
    A: NDArray,
    b: NDArray,
    x0: NDArray | None = None,
    *,
    preconditioner: Callable[..., NDArray] | LinearOperator | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    tol: float | None = None,
    maxiter: int | None = None,
    trace_mode: TraceMode = TraceMode.MINIMAL,
) -> tuple[NDArray, SolverResult]:
    """Solve linear system using scipy.sparse.linalg.cg with callback monitoring.

    This solver wraps scipy's CG implementation while capturing iteration
    data via callbacks. Useful for comparison against scipy baseline.

    Mathematical Method:
        Delegates to scipy.sparse.linalg.cg, which implements standard
        preconditioned conjugate gradient with two-term recurrence.

    Args:
        A: System matrix, shape (n, n). Must be square and SPD.
        b: Right-hand side vector, shape (n,).
        x0: Initial guess. If None, uses zero vector.
        preconditioner: Callable implementing M^{-1} or LinearOperator.
        rtol: Relative tolerance for convergence. Default: 1e-6.
        atol: Absolute tolerance for convergence. Default: 1e-14.
        tol: Alternative name for rtol (overrides rtol if provided).
        maxiter: Maximum number of iterations. Default: 100.
        trace_mode: Tracing granularity level. Default: MINIMAL.
            MINIMAL: Log scalars only (iteration, residual_norm, flags)
            FULL: Log scalars + vectors (residual, solution)
            DISABLED: No logging (event_log will be None)

    Returns:
        Tuple of (x, result):
        - x (ndarray): Solution vector, shape (n,)
        - result (SolverResult): Convergence information

    Examples:
        >>> # Basic usage
        >>> x, result = scipy_cg(A, b)

        >>> # With preconditioner and full logging
        >>> x, result = scipy_cg(A, b, preconditioner=jacobi, trace_mode="full")

    References:
        - scipy.sparse.linalg.cg documentation
        - Hestenes & Stiefel (1952). Methods of Conjugate Gradients.
    """
    from .solvers.scipy_cg_solver import SciPyCGSolver

    # Create preconditioner strategy
    if preconditioner is None:
        precond_strategy = IdentityPreconditioner()
    elif isinstance(preconditioner, LinearOperator):
        precond_strategy = LinearOperatorPreconditioner(preconditioner)
    else:
        precond_strategy = FunctionPreconditioner(preconditioner)

    # Create event logger based on trace mode
    trace_mode_enum = (
        TraceMode(trace_mode) if isinstance(trace_mode, str) else trace_mode
    )
    logger = TraceRecorder() if trace_mode_enum != TraceMode.DISABLED else None

    # Construct solver
    solver = SciPyCGSolver(
        preconditioner=precond_strategy,
        event_logger=logger,
    )

    # Resolve rtol alias
    rtol_value = tol if tol is not None else rtol

    # Solve system
    x, result = solver.solve(
        A,
        b,
        x0,
        rtol=rtol_value,
        atol=atol,
        maxiter=maxiter,
        trace_mode=trace_mode_enum,
    )

    return x, result
