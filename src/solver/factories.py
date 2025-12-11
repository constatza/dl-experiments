"""Factory functions for creating CG solver variants.

This module provides simple, high-level factory functions for constructing
different Conjugate Gradient solver configurations. Each factory encapsulates
the appropriate strategy choices and sensible defaults for common use cases.

Design Pattern: Factory Pattern
    Instead of requiring users to manually assemble strategies and solvers,
    these factories provide convenient entry points with clear semantic meaning:

    1. flexible_cg() - FCG with Gram-Schmidt orthogonalization (target algorithm)
    2. preconditioned_cg() - PCG with flexible beta updates (includes baseline when preconditioner=None)

    Each factory handles:
    - Strategy instantiation with appropriate parameters
    - Solver construction with injected strategies
    - Parameter forwarding to solver.solve()

Mathematical Context:
    These factories create different solver variants for different scenarios:

    flexible_cg:
        Best for neural preconditioners or highly non-SPD M_k.
        Uses truncated Gram-Schmidt to maintain approximate A-conjugacy
        despite preconditioner variation.

    preconditioned_cg:
        Best for classical preconditioners (Jacobi, ILU, etc.) or baseline (preconditioner=None).
        Uses two-term recurrence with restart mechanism.

References:
    - Notay (2000): "Flexible Conjugate Gradients"
    - Saad (2003): "Iterative Methods for Sparse Linear Systems"
    - Algorithm.md: Detailed algorithm specification
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from scipy.sparse.linalg import LinearOperator

from .event_logging import VectorLogger
from .fcg_solver import FlexibleConjugateGradientSolver
from .pcg_solver import cg_solve
from .info import SolverResult
from .preconditioners import (
    CallablePreconditioner,
    IdentityPreconditioner,
    LinearOperatorPreconditioner,
)
from ..constants import (
    DEFAULT_ATOL,
    DEFAULT_BETA_MAX,
    DEFAULT_CURVATURE_EPSILON,
    DEFAULT_DIVERGENCE_FACTOR,
    DEFAULT_FCG_HISTORY_LIMIT,
    DEFAULT_M_MAX,
    DEFAULT_RESIDUAL_REPLACEMENT_FREQ,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from .reorthogonalization import ReorthogonalizationStrategy


def flexible_cg(
    A: NDArray,
    b: NDArray,
    x0: NDArray | None = None,
    *,
    preconditioner: Callable[..., NDArray] | LinearOperator | None = None,
    m_max: int = DEFAULT_M_MAX,
    history_limit: int = DEFAULT_FCG_HISTORY_LIMIT,
    rtol: float = 1e-6,
    atol: float = DEFAULT_ATOL,
    tol: float | None = None,
    max_iterations: int | None = None,
    max_iter: int | None = None,
    eps_curv: float = DEFAULT_CURVATURE_EPSILON,
    eps_breakdown: float = 1e-14,
    breakdown_tol: float | None = None,
    m_replacement: int = DEFAULT_RESIDUAL_REPLACEMENT_FREQ,
    gamma_div: float = DEFAULT_DIVERGENCE_FACTOR,
    stopping_criterion: Literal["residual_norm", "tolerance", "fixed_iterations"] = "residual_norm",
    reorthogonalize: ReorthogonalizationStrategy | None = None,
    capture_traces: bool | None = None,
    capture_search_directions: bool | None = None,
    helper: Callable[..., NDArray] | None = None,
) -> tuple[NDArray, SolverResult]:
    """Solve linear system using Flexible CG with truncated orthogonalization.

    This is the target algorithm for neural preconditioning and non-SPD
    preconditioners. It uses explicit Gram-Schmidt orthogonalization against
    a sliding window of previous search directions to maintain approximate
    A-conjugacy despite preconditioner variation.

    Mathematical Method:
        At each iteration k:
        1. Apply preconditioner: z_k = M_k^{-1} r_k
        2. Orthogonalize against last m_k directions:
           p_k = z_k - Σ_{i=k-m_k}^{k-1} [(z_k, q_i) / (p_i, q_i)] p_i
        3. Update solution: x_{k+1} = x_k + α_k p_k
        4. Update residual: r_{k+1} = r_k - α_k A p_k

        where m_k = min(k, max(1, mod(k, m_max+1))) implements periodic reset.

    When to Use:
        - Neural network preconditioners (time-varying, non-symmetric)
        - Severely non-SPD preconditioners
        - Cases where PCG shows frequent restarts
        - Memory allows O(m_max * n) storage

    Args:
        A: System matrix, shape (n, n). Must be square and SPD.
        b: Right-hand side vector, shape (n,).
        x0: Initial guess, shape (n,). If None, uses zero vector.

    Preconditioner Parameters:
        preconditioner: Callable implementing M^{-1} operation.
            Signature: f(r) -> z or f(r, ctx) -> z.
            If None, uses identity (no preconditioning).
        m_max: Maximum history length for orthogonalization. Default: 10.
            Larger values increase cost but improve stability.
            Typical range: 5-50.
        history_limit: Maximum search directions retained for orthogonalization
            and optional reorthogonalization. Increase to mitigate loss of
            orthogonality on long runs. Default: 200.

    Tolerance Parameters:
        rtol: Relative tolerance for convergence. Default: 1e-6.
            Convergence when ||r_k|| <= rtol * ||b||.
        atol: Absolute tolerance for convergence. Default: 1e-14.
            Convergence when ||r_k|| <= atol (handles b=0 case).
        tol: Alternative name for rtol (overrides rtol if provided).

    Iteration Control:
        max_iterations: Maximum number of iterations. Default: 100.
        max_iter: Alternative name for max_iterations.
        stopping_criterion: When to stop iterations:
            - "residual_norm": Stop when ||r_k|| <= max(rtol*||b||, atol)
            - "tolerance": Same as residual_norm
            - "fixed_iterations": Run exactly max_iterations regardless

    Numerical Safety Parameters:
        eps_curv: Curvature threshold for restart. Default: 1e-14.
            Restart if d_k <= eps_curv * ||p_k||^2 or d_k <= 0.
        eps_breakdown: Minimum safe denominator in step length. Default: 1e-14.
        breakdown_tol: Alternative name for eps_breakdown.
        m_replacement: Frequency of true residual recomputation. Default: 50.
            Every m_replacement iterations, recompute r = b - A*x.
        gamma_div: Divergence threshold multiplier. Default: 1e10.
            Trigger divergence if ||r_k|| > gamma_div * ||b||.

    Diagnostics:
        capture_traces: If True, record residual and solution vectors at each
            iteration for post-hoc analysis. Default: False.
        capture_search_directions: If True, also record search directions p_k
            and products q_k. Requires capture_traces=True. Default: False.
        reorthogonalize: Optional reorthogonalization strategy for numerical
            stability. Applied after direction update. Default: None.

    Returns:
        Tuple of (x, info):
        - x (ndarray): Solution vector, shape (n,)
        - info (FlexiblePCGInfo): Detailed convergence information

    Raises:
        ValueError: If A is not square or b has incompatible shape.

    Theory:
        The truncated orthogonalization strategy enforces approximate A-conjugacy
        even when the preconditioner M_k varies with iteration. This is essential
        for neural preconditioners, which are inherently time-varying and non-symmetric.

        The periodic reset mechanism (via m_max) balances:
        - Numerical stability (prevents error accumulation)
        - Memory usage (O(m_max * n) storage)
        - Convergence rate (more history → better orthogonality)

        For SPD preconditioners, this reduces to standard PCG when m_max is large.
        For non-SPD preconditioners, the restart mechanism provides recovery.

    Performance:
        - Memory: O(m_max * n) for direction history
        - Cost per iteration: O(m_k * n) for orthogonalization, m_k ≤ m_max
        - Typical overhead: 5-10x vs unpreconditioned CG for m_max=10

    Examples:
        >>> # Neural preconditioner with default parameters
        >>> def neural_precond(r, ctx):
        ...     features = extract_features(r, ctx.solution)
        ...     return model.predict(features)
        >>> x, info = flexible_cg(A, b, preconditioner=neural_precond)

        >>> # High-accuracy solve with tight tolerance
        >>> x, info = flexible_cg(A, b, preconditioner=neural_precond,
        ...                       rtol=1e-10, m_max=20)

        >>> # Diagnostic run with trace capture
        >>> x, info = flexible_cg(A, b, preconditioner=neural_precond,
        ...                       capture_traces=True,
        ...                       capture_search_directions=True)
        >>> import matplotlib.pyplot as plt
        >>> plt.semilogy(info.residual_history)

    See Also:
        preconditioned_cg: PCG with two-term recurrence (lower cost)
    """

    # Create preconditioner strategy
    if preconditioner is None:
        precond_strategy = None  # Will use IdentityPreconditioner by default
    elif isinstance(preconditioner, LinearOperator):
        precond_strategy = LinearOperatorPreconditioner(preconditioner)
    else:
        precond_strategy = CallablePreconditioner(preconditioner)

    # Create event logger for iteration history
    logger = VectorLogger()

    # Construct new FCG solver
    solver = FlexibleConjugateGradientSolver(
        preconditioner=precond_strategy,
        m_max=m_max,
        history_limit=history_limit,
        event_logger=logger,
    )

    # Resolve parameter aliases
    max_iters = (
        max_iterations
        if max_iterations is not None
        else (max_iter if max_iter is not None else 100)
    )
    rtol_value = tol if tol is not None else rtol
    eps_break = breakdown_tol if breakdown_tol is not None else eps_breakdown

    # Map tolerance to residual_norm for backwards compatibility
    criterion = stopping_criterion if stopping_criterion != "tolerance" else "residual_norm"

    # Solve system
    x, result = solver.solve(
        A,
        b,
        x0,
        rtol=rtol_value,
        atol=atol,
        max_iterations=max_iters,
        eps_curv=eps_curv,
        eps_breakdown=eps_break,
        m_replacement=m_replacement,
        gamma_div=gamma_div,
        stopping_criterion=criterion,
        reorthogonalize=reorthogonalize,
        capture_traces=bool(capture_traces),
        capture_search_directions=bool(capture_search_directions),
        helper=helper,
    )

    return x, result


def preconditioned_cg(
    A: NDArray,
    b: NDArray,
    x0: NDArray | None = None,
    *,
    preconditioner: Callable[..., NDArray] | None = None,
    beta_max: float = DEFAULT_BETA_MAX,
    rtol: float = 1e-6,
    atol: float = DEFAULT_ATOL,
    tol: float | None = None,
    max_iterations: int | None = None,
    max_iter: int | None = None,
    eps_curv: float = DEFAULT_CURVATURE_EPSILON,
    eps_breakdown: float = 1e-14,
    breakdown_tol: float | None = None,
    m_replacement: int = DEFAULT_RESIDUAL_REPLACEMENT_FREQ,
    gamma_div: float = DEFAULT_DIVERGENCE_FACTOR,
    stopping_criterion: Literal["residual_norm", "tolerance", "fixed_iterations"] = "residual_norm",
) -> tuple[NDArray, SolverResult]:
    """Solve linear system using SciPy's CG with optional preconditioning."""
    logger = VectorLogger()
    max_iters = (
        max_iterations if max_iterations is not None else (max_iter if max_iter is not None else 100)
    )
    rtol_value = tol if tol is not None else rtol
    x, result = cg_solve(
        A,
        b,
        x0=x0,
        tol=rtol_value,
        atol=atol,
        maxiter=max_iters,
        M=preconditioner,
        callback_type="x",
        record_history=True,
        track_solution=False,
        event_log=logger,
    )
    # Preserve legacy stopping_criterion field for compatibility
    result.stopping_criterion = stopping_criterion
    return x, result
