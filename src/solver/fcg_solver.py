"""Flexible Conjugate Gradient solver with event sourcing and SOLID design.

This module implements the Flexible Conjugate Gradient (FCG) algorithm
following the IIterativeSolver interface. It uses truncated Gram-Schmidt
orthogonalization for robustness with non-SPD preconditioners.

Algorithm (from user specification):
    Initialize: x_0 = 0, r_0 = b
    For i = 0, 1, ..., iter-1:
      1. z_i = B(r_i)                                      # Apply preconditioner
      2. m_i = min(i, max(1, mod(i, m_max+1)))             # Compute history length
      3. p_i = z_i - Σ_{k=i-m_i}^{i-1} [(z_i, q_k) / (p_k, q_k)] p_k  # Orthogonalize
      4. q_i = A p_i                                       # Matrix-vector product
      5. α_i = (p_i, r_i) / (p_i, q_i)                    # Step length
      6. x_{i+1} = x_i + α_i p_i                          # Update solution
      7. r_{i+1} = r_i - α_i q_i                          # Update residual

Design Principles:
    - Single Responsibility: Solver only orchestrates algorithm
    - Dependency Inversion: Depends on Preconditioner abstraction
    - Interface Segregation: Implements minimal IIterativeSolver interface
    - Trace Recording: Optional TraceRecorder for iteration history

References:
    - Notay (2000): "Flexible Conjugate Gradients"
    - Saad (2003): "Iterative Methods for Sparse Linear Systems"
    - Algorithm.md: Detailed algorithm documentation
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal
from collections.abc import Callable

import numpy as np
from scipy.linalg import norm

from ..constants import (
    DEFAULT_ATOL,
    DEFAULT_CURVATURE_EPSILON,
    DEFAULT_DIVERGENCE_FACTOR,
    DEFAULT_FCG_HISTORY_LIMIT,
    DEFAULT_M_MAX,
    DEFAULT_RESIDUAL_REPLACEMENT_FREQ,
)
from .convergence import CombinedToleranceCriterion, IConvergenceCriterion
from .direction_strategies import TruncatedOrthogonalDirection
from .helpers import (
    convergence_check,
    curvature,
    initialize_state,
    residual_management,
    step_length,
)
from .info import IterationContext, SolverResult
from .interfaces import IIterativeSolver
from .preconditioners import IdentityPreconditioner, Preconditioner
from .reorthogonalization import (
    ReorthogonalizationReport,
    ReorthogonalizationStrategy,
    apply_reorthogonalization,
)
from .state import DirectionHistory, IterationState

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .trace_recorder import TraceRecorder


def _ensure_vector(
    vector: np.ndarray | None, template: np.ndarray, *, name: str
) -> np.ndarray:
    """Validate and reshape vector to match template shape.

    Args:
        vector: Array to validate (may be None or misshapen).
        template: Reference array defining expected shape.
        name: Descriptive name for error messages.

    Returns:
        Validated array with shape matching template.

    Raises:
        ValueError: If vector is None.
    """
    if vector is None:
        raise ValueError(f"{name} returned None; expected ndarray")
    arr = np.asarray(vector, dtype=np.float64)
    if arr.shape != template.shape:
        arr = arr.reshape(template.shape)
    return arr


def _relative(value: float, b_norm: float) -> float:
    """Compute relative value with respect to RHS norm.

    Args:
        value: Absolute value (e.g., residual norm).
        b_norm: Reference norm (e.g., ||b||).

    Returns:
        Relative value: value / b_norm if b_norm > 0, else value.
    """
    return value / b_norm if b_norm > 0 else value


class FlexibleConjugateGradientSolver(IIterativeSolver):
    """Flexible Conjugate Gradient solver with trace recording.

    Implements the FCG algorithm using truncated Gram-Schmidt orthogonalization.
    This solver is specifically designed for systems with non-SPD or time-varying
    preconditioners (e.g., neural network preconditioners).

    Algorithm:
        The solver implements the exact algorithm specified:
        1. Apply preconditioner: z_i = B(r_i)
        2. Compute history length: m_i = min(i, max(1, mod(i, m_max+1)))
        3. Gram-Schmidt orthogonalization: p_i = z_i - Σ [(z_i,q_k)/(p_k,q_k)] p_k
        4. Matrix-vector product: q_i = A p_i
        5. Step length: α_i = (p_i, r_i) / (p_i, q_i)
        6. Update solution: x_{i+1} = x_i + α_i p_i
        7. Update residual: r_{i+1} = r_i - α_i q_i

    Design:
        - Uses TruncatedOrthogonalDirection for step 3 (Gram-Schmidt)
        - Optional TraceRecorder for iteration history
        - Implements IIterativeSolver interface
        - Convergence checking via check_convergence() method

    Attributes:
        preconditioner: Preconditioner strategy (default: IdentityPreconditioner)
        direction_strategy: TruncatedOrthogonalDirection with m_max parameter
        event_logger: Optional TraceRecorder for event sourcing

    Mathematical Properties:
        - Handles non-SPD preconditioners via restart mechanism
        - Periodic reset of orthogonalization history (m_i calculation)
        - Numerical safety: curvature checks, breakdown detection
        - Residual management for long iterations

    Examples:
        >>> # Basic solve with identity preconditioner
        >>> solver = FlexibleConjugateGradientSolver()
        >>> x, result = solver.solve(A, b, rtol=1e-6)
        >>> print(f"Converged: {result.converged}, Iterations: {result.iterations}")
        >>>
        >>> # With event logging
        >>> logger = TraceRecorder()
        >>> solver = FlexibleConjugateGradientSolver(event_logger=logger)
        >>> x, result = solver.solve(A, b)
        >>> residual_history = logger.get_history("residual_norm")
        >>>
        >>> # With Jacobi preconditioner
        >>> jacobi = make_jacobi_preconditioner(A)
        >>> precond = CallablePreconditioner(jacobi)
        >>> solver = FlexibleConjugateGradientSolver(preconditioner=precond, m_max=10)
        >>> x, result = solver.solve(A, b, rtol=1e-10)
    """

    def __init__(
        self,
        preconditioner: Preconditioner | None = None,
        m_max: int = DEFAULT_M_MAX,
        history_limit: int = DEFAULT_FCG_HISTORY_LIMIT,
        event_logger: TraceRecorder | None = None,
    ) -> None:
        """Initialize FCG solver.

        Args:
            preconditioner: Preconditioner strategy. If None, uses identity
                preconditioner (no preconditioning). For custom preconditioners,
                wrap with CallablePreconditioner.
            m_max: Maximum orthogonalization window size. Controls memory
                usage (O(m_max * n)) and orthogonalization cost (O(m_max * n)
                per iteration). Default: 10 (good balance). Typical range: 5-50.
            history_limit: Maximum number of search directions retained for
                orthogonalization/reorthogonalization. Increase if residuals
                oscillate from loss of orthogonality. Default: 200.
            event_logger: Optional TraceRecorder for iteration history. If provided,
                logs residual, solution, search_direction, and residual_norm at
                each iteration. Default: None (no logging, zero overhead).

        Theory:
            The m_max parameter controls the trade-off between:
            - Numerical stability (larger m_max → better orthogonality)
            - Memory usage (O(m_max * n))
            - Computational cost (O(m_max * n) per iteration)

            For most problems, m_max=10 provides good stability without
            excessive cost. Increase for severely ill-conditioned systems.
        """
        self.preconditioner = preconditioner or IdentityPreconditioner()
        self.direction_strategy = TruncatedOrthogonalDirection(m_max=m_max)
        self.history_limit = history_limit
        self.event_logger = event_logger

    def check_convergence(
        self,
        residual: NDArray,
        rhs_norm: float,
        criterion: IConvergenceCriterion,
    ) -> bool:
        """Check convergence: ||r|| <= max(rtol*||b||, atol).

        Implements the standard convergence criterion for iterative solvers.

        Args:
            residual: Current residual vector r_k = b - A x_k.
            rhs_norm: Precomputed norm ||b|| of right-hand side.
            criterion: Convergence criterion (default matches SciPy).

        Returns:
            True if converged, False otherwise.

        Theory:
            The combined relative/absolute tolerance ensures:
            - Relative tolerance: Solution accurate for typical problems
            - Absolute tolerance: Prevents false convergence when ||b|| small
        """
        return criterion.has_converged(residual, rhs_norm)

    def solve(
        self,
        A: NDArray,
        b: NDArray,
        x0: NDArray | None = None,
        *,
        rtol: float = 1e-6,
        atol: float = DEFAULT_ATOL,
        max_iterations: int = 100,
        eps_curv: float = DEFAULT_CURVATURE_EPSILON,
        eps_breakdown: float = 1e-14,
        m_replacement: int = DEFAULT_RESIDUAL_REPLACEMENT_FREQ,
        gamma_div: float = DEFAULT_DIVERGENCE_FACTOR,
        stopping_criterion: Literal[
            "residual_norm", "fixed_iterations"
        ] = "residual_norm",
        reorthogonalize: ReorthogonalizationStrategy | None = None,
        capture_traces: bool = False,
        capture_search_directions: bool = False,
        helper: Callable[[NDArray], NDArray] | None = None,
        convergence_criterion: IConvergenceCriterion | None = None,
        **kwargs: object,
    ) -> tuple[NDArray, SolverResult]:  # type: ignore[override]
        """Solve linear system Ax = b using FCG algorithm.

        This method implements the FCG algorithm exactly as specified:
        - Step 1: Apply preconditioner z_i = B(r_i)
        - Step 2: Compute m_i for periodic reset
        - Step 3: Gram-Schmidt orthogonalization (via direction_strategy)
        - Step 4: Compute q_i = A p_i
        - Step 5: Compute step length α_i
        - Step 6: Update solution x_{i+1}
        - Step 7: Update residual r_{i+1}

        If event_logger is provided, logs at each iteration:
        - "residual": r_i vector
        - "solution": x_i vector
        - "search_direction": p_i vector
        - "residual_norm": ||r_i|| scalar

        Args:
            A: System matrix, shape (n, n). Must be square.
            b: Right-hand side vector, shape (n,).
            x0: Initial guess, shape (n,). If None, uses zero vector.
            rtol: Relative tolerance for convergence. Default: 1e-6.
            atol: Absolute tolerance for convergence. Default: 1e-14.
            max_iterations: Maximum number of iterations. Default: 100.
            eps_curv: Curvature threshold for restart. Default: 1e-14.
            eps_breakdown: Minimum safe denominator. Default: 1e-14.
            m_replacement: Frequency of true residual recomputation. Default: 50.
            gamma_div: Divergence threshold multiplier. Default: 1e10.
            stopping_criterion: When to stop ("residual_norm" or "fixed_iterations").
            reorthogonalize: Optional strategy to reorthogonalize search directions.
            **kwargs: Additional arguments (ignored for compatibility).

        Returns:
            Tuple of (solution, solver_result):
            - solution (ndarray): Solution vector x, shape (n,)
            - solver_result (SolverResult): Convergence information

        Raises:
            ValueError: If A is not square or b has incompatible shape.

        Theory:
            The FCG algorithm maintains approximate A-conjugacy via explicit
            Gram-Schmidt orthogonalization. Unlike standard CG, it works with
            non-SPD preconditioners by:
            1. Orthogonalizing against A p_k products (not p_k directly)
            2. Periodic reset of history (m_i calculation)
            3. Numerical safety checks (curvature, breakdown)
        """
        # Validate inputs
        n = A.shape[0]
        if A.shape != (n, n):
            raise ValueError(f"A must be square, got shape {A.shape}")
        if b.shape != (n,):
            raise ValueError(f"b must have shape ({n},), got {b.shape}")

        # Precompute RHS norm for convergence checks
        b_norm = norm(b)

        # Build convergence criterion (SciPy-style by default)
        criterion = convergence_criterion or CombinedToleranceCriterion(
            rtol=rtol, atol=atol
        )

        # Initialize solver state
        x, r, z, p, state, history = self._initialize_state(A, b, x0, b_norm)

        logger: TraceRecorder | None = self.event_logger
        if capture_traces and logger is None:
            logger = TraceRecorder()
            self.event_logger = logger

        # Log initial state once through the active logger
        if logger is not None:
            logger.log("residual", r)
            logger.log("solution", x)
            logger.log("search_direction", p)
            logger.log("residual_norm", state.residual_history[0])

        # Main iteration loop
        q_managed: NDArray | None = None
        for iteration in range(max_iterations):
            # Track residual replacements to detect if restart occurred
            num_residual_replacements_before = state.num_residual_replacements

            # Step 1: Check convergence
            state = convergence_check(r, b_norm, criterion, state)
            if state.converged and stopping_criterion != "fixed_iterations":
                break

            # Step 4: Compute matrix-vector product q = A @ p
            q = A @ p

            # Check curvature d = p^T q
            d, state = curvature(p, q, z, eps_curv, state)
            if state.restart:
                # Restart: replace p with z and recompute q
                p = z.copy()
                q = A @ p
                d = float(np.dot(p, q))
                state = replace(state, restart=False)

            # Step 5: Compute step length α = (p^T r) / d (FCG formula)
            alpha, state = step_length(p, r, z, d, eps_breakdown, state)
            if state.breakdown:
                # Breakdown detected, terminate
                break

            # Step 6: Update solution x_{i+1} = x_i + α p_i
            x = x + alpha * p

            # Step 7: Update residual r_{i+1} = r_i - α q_i
            r_new = r - alpha * q

            # Step 1 (next iteration): Apply preconditioner z_{i+1} = M^{-1} r_{i+1}
            context = IterationContext(
                iteration=iteration + 1,
                residual=r_new.copy(),
                solution=x.copy(),
                matrix=A,
                rhs=b,
            )
            z_new = _ensure_vector(
                self.preconditioner.apply(r_new, context),
                r_new,
                name="preconditioner",
            )

            # Step 3: Compute new direction p_{i+1} via Gram-Schmidt orthogonalization
            # The direction_strategy (TruncatedOrthogonalDirection) implements:
            # m_i = min(i, max(1, mod(i, m_max+1)))
            # p_i = z_i - Σ_{k=i-m_i}^{i-1} [(z_i, q_k) / (p_k, q_k)] p_k
            p_new, state = self.direction_strategy.compute_direction(
                r_new, z_new, state, history, r_prev=r
            )

            # Residual management (periodic true residual recomputation)
            def precond_with_iteration(residual_vec: NDArray, iter_idx: int) -> NDArray:
                """Wrapper for residual_management helper."""
                ctx = IterationContext(
                    iteration=iter_idx,
                    residual=residual_vec,
                    solution=x.copy(),
                    matrix=A,
                    rhs=b,
                )
                return _ensure_vector(
                    self.preconditioner.apply(residual_vec, ctx),
                    residual_vec,
                    name="preconditioner",
                )

            r_managed, z_managed, p_managed, state = residual_management(
                A,
                precond_with_iteration,
                b,
                x,
                r_new,
                z_new,
                p_new,
                state,
                iteration + 1,
                m_replacement,
                gamma_div,
                lambda i: IterationContext(
                    iteration=i,
                    residual=r_new.copy(),
                    solution=x.copy(),
                    matrix=A,
                    rhs=b,
                ),
            )

            # Optional reorthogonalization hook shared across solvers
            reorth_info: ReorthogonalizationReport | None = None
            if reorthogonalize is not None:
                p_managed, reorth_info = apply_reorthogonalization(
                    reorthogonalize,
                    p_managed,
                    history.p_vectors,
                    iteration + 1,
                )
                if reorth_info.breakdown:
                    state = replace(state, breakdown=True)
                    q = A @ p_managed
            reorth_applied = bool(reorth_info) and not reorth_info.skipped

            # Update iteration state
            current_res_norm = norm(r_managed)
            threshold = criterion.threshold(b_norm)
            is_converged_now = current_res_norm <= threshold

            # Update state counters
            # SciPy convention: converged=True if residual meets tolerance,
            # regardless of breakdown status. Breakdown is a separate diagnostic.
            state = replace(
                state,
                iterations=iteration + 1,
                converged=is_converged_now,
            )

            # Log iteration values through the active logger
            if logger is not None:
                logger.log("residual", r_managed)
                logger.log("solution", x)
                logger.log("search_direction", p_managed)
                logger.log("residual_norm", current_res_norm)
                if capture_traces and capture_search_directions:
                    q_to_log = q_managed if q_managed is not None else q
                    logger.log("search_direction_product", q_to_log)

            # Update history for next iteration
            # If residual_management did a restart, p_managed is the new steepest descent
            if state.num_residual_replacements > num_residual_replacements_before:
                # Residual replacement occurred, p was reset to z
                q_managed = A @ p_managed
            else:
                q_managed = (
                    A @ p_managed if reorth_applied else q
                )  # Track reorthogonalized direction

            history = history.update(
                p_managed, q_managed, r, z, m_max=self.history_limit
            )

            # Update state vectors for next iteration
            r = r_managed
            z = z_managed
            p = p_managed

            # Check for convergence and exit if appropriate
            if state.breakdown:
                break
            if is_converged_now and stopping_criterion != "fixed_iterations":
                break

        # Compute final residual
        final_residual_norm = norm(b - A @ x)
        final_relative_residual = _relative(final_residual_norm, b_norm)

        # SciPy-style info code convention:
        # - info = 0: successfully converged (residual met tolerance)
        # - info = iterations: did not converge (max iterations or breakdown)
        # Keep breakdown flag separate for diagnostics
        info_code = 0 if state.converged else state.iterations

        active_logger = logger

        residual_history_abs = (
            active_logger.get_history("residual_norm")
            if active_logger is not None
            else state.residual_history
        )

        # Convert absolute residuals to relative for residual_history
        residual_history_rel = None
        if residual_history_abs:
            residual_history_rel = [
                float(r / b_norm) if b_norm > 0 else float(r)
                for r in residual_history_abs
            ]

        # Assemble return info
        residual_vectors = None
        solution_vectors = None
        if capture_traces and active_logger is not None:
            residual_vectors = active_logger.get_history("residual")
            solution_vectors = active_logger.get_history("solution")

        result = SolverResult(
            converged=state.converged,
            iterations=state.iterations,
            info=info_code,
            niter=state.iterations,
            status=info_code,
            residual=final_relative_residual,
            residual_abs=final_residual_norm,
            residual_rel=final_relative_residual,
            residual_history=residual_history_rel,
            residual_history_abs=[float(r) for r in residual_history_abs]
            if residual_history_abs
            else None,
            rhs_norm=b_norm,
            breakdown=state.breakdown,
            stopping_criterion=stopping_criterion,
            event_log=active_logger,
            tol=rtol,
            atol=atol,
            maxiter=max_iterations,
            residual_vectors=residual_vectors,
            solution_vectors=solution_vectors,
        )

        return x, result

    def _initialize_state(
        self,
        A: NDArray,
        b: NDArray,
        x0: NDArray | None,
        b_norm: float,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, IterationState, DirectionHistory]:
        """Initialize solver state vectors and iteration state.

        Args:
            A: System matrix, shape (n, n).
            b: Right-hand side vector, shape (n,).
            x0: Initial guess, shape (n,). If None, uses zero vector.
            b_norm: Precomputed norm of b.

        Returns:
            Tuple of (x, r, z, p, state, history):
            - x: Initial solution vector
            - r: Initial residual r_0 = b - A*x_0
            - z: Initial preconditioned residual z_0 = M^{-1} r_0
            - p: Initial search direction p_0 = z_0
            - state: Initial IterationState
            - history: Empty DirectionHistory

        Theory:
            Initialization follows standard CG setup:
            1. Start with initial guess x_0 (zero if not provided)
            2. Compute initial residual r_0 = b - A*x_0
            3. Apply preconditioner z_0 = M^{-1} r_0
            4. Set initial direction p_0 = z_0 (steepest descent)
            5. Initialize state with ||r_0||
        """
        # Initialize solution vector
        if x0 is None:
            x = np.zeros(A.shape[0], dtype=np.float64)
        else:
            x = np.asarray(x0, dtype=np.float64).copy()

        # Compute initial residual r_0 = b - A*x_0
        r = b - A @ x

        # Apply preconditioner z_0 = M^{-1} r_0
        context = IterationContext(
            iteration=0,
            residual=r.copy(),
            solution=x.copy(),
            matrix=A,
            rhs=b,
        )
        z = _ensure_vector(
            self.preconditioner.apply(r, context),
            r,
            name="preconditioner",
        )

        # Initialize search direction p_0 = z_0
        p = z.copy()

        # Initialize iteration state
        state = initialize_state(r, b_norm)

        # Initialize empty history
        history = DirectionHistory()

        return x, r, z, p, state, history
