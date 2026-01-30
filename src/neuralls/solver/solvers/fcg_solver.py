"""Flexible Conjugate Gradient solver.

This module implements the Flexible Conjugate Gradient (FCG) algorithm
from Notay (2000) using ConjugateGradientBase and strategy pattern.

Algorithm (FCG from Notay 2000, Algorithm 2.1):
    Initialize: u_0 = 0, r_0 = b, m_0 = 0
    For i = 0, 1, ..., until convergence:
      1. w_i = M^{-1}(r_i)                           # Apply (variable) preconditioner
      2. Compute m_i = max(1, i mod (m_max + 1))     # Periodic restart formula
      3. d_i = orthogonalize(w_i, D, AD)             # Orthogonalize against last m_i directions
      4. α_i = (d_i, r_i) / (d_i, A d_i)             # Step length
      5. u_{i+1} = u_i + α_i * d_i                   # Update solution
      6. r_{i+1} = r_i - α_i * A d_i                 # Update residual

Orthogonalization (from Notay 2000, Algorithm 2.1):
    d_i = w_i - Σ_{j=i-m_i}^{i-1} [(w_i, A d_j) / (d_j, A d_j)] d_j

This enforces approximate A-conjugacy: d_i^T A d_j ≈ 0 for i ≠ j.

Design Principles:
    - Strategy Pattern: Inject OrthogonalizationStrategy for flexibility
    - Template Method: Inherit from ConjugateGradientBase
    - Single Responsibility: Only implements direction computation
    - Composition over Inheritance: Strategies injected, not inherited

References:
    - Notay, Y. (2000). Flexible Conjugate Gradients. SIAM Journal on Scientific Computing,
      22(4), 1444-1460. doi:10.1137/S1064827599362314
      * Section 2, Algorithm 2.1 - FCG with periodic restart
      * Section 5.1 - FCG(∞) for ill-conditioned problems
      * Table 1 - Benchmark comparisons of truncation strategies
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems (2nd ed.). SIAM.
      * Chapter 9, Section 9.5 - Flexible variants of CG
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import aslinearoperator

from ...constants import (
    DEFAULT_ATOL,
    DEFAULT_M_MAX,
    DEFAULT_RTOL,
)
from .cg_base import ConjugateGradientBase
from ..strategies.convergence import CombinedToleranceCriterion, IConvergenceCriterion
from ..strategies.orthogonalization import (
    OrthogonalizationStrategy,
    PeriodicRestartOrthogonalization,
)
from ..models.state import CGState, KrylovState, SolverState
from ..models.result import SolverResult
from ..preconditioners import IdentityPreconditioner, Preconditioner
from ..monitoring.trace_recorder import TraceRecorder

if TYPE_CHECKING:
    from numpy.typing import NDArray


class FlexibleCGSolver(ConjugateGradientBase):
    """Flexible Conjugate Gradient solver (Notay 2000).

    Implements the FCG algorithm from Notay (2000) Algorithm 2.1 with periodic restart.
    FCG extends standard PCG to handle variable and non-SPD preconditioners through
    explicit orthogonalization of search directions.

    The key innovation is maintaining approximate A-conjugacy via truncated Gram-Schmidt
    orthogonalization with periodic restart, enabling convergence even when the
    preconditioner changes between iterations or is non-symmetric.

    Key Features:
        - Handles variable preconditioners (neural networks, adaptive methods)
        - Truncated orthogonalization with periodic restart (Notay 2000)
        - Compatible with non-SPD preconditioners
        - Configurable orthogonalization strategies (FCG(m), FCG(∞), Tr-FCG)

    Attributes:
        preconditioner: Preconditioner M to apply (can be variable, default: Identity)
        orthogonalization: Strategy for orthogonalizing directions (default: PeriodicRestartOrthogonalization with m_max=10)
        convergence_criterion: Convergence test strategy
        event_logger: Optional event recorder for diagnostics

    Example:
        >>> from neuralls.solver.strategies.orthogonalization import PeriodicRestartOrthogonalization
        >>> # FCG(10) with periodic restart (Notay 2000)
        >>> orthog = PeriodicRestartOrthogonalization(m_max=10)
        >>> solver = FlexibleCGSolver(
        ...     preconditioner=my_neural_preconditioner,
        ...     orthogonalization=orthog,
        ... )
        >>> u, result = solver.solve(A, b, rtol=1e-6)

    References:
        Notay, Y. (2000). Flexible Conjugate Gradients. SIAM Journal on Scientific Computing,
        22(4), 1444-1460. doi:10.1137/S1064827599362314
    """

    def __init__(
        self,
        preconditioner: Preconditioner | None = None,
        orthogonalization: OrthogonalizationStrategy | None = None,
        convergence_criterion: IConvergenceCriterion | None = None,
        event_logger: TraceRecorder | None = None,
    ) -> None:
        """Initialize Flexible CG solver.

        Args:
            preconditioner: Preconditioner M (can be variable, default: Identity)
            orthogonalization: Orthogonalization strategy (default: PeriodicRestartOrthogonalization
                with m_max=10, implementing FCG(10) from Notay 2000)
            convergence_criterion: Convergence test (default: Combined rtol/atol)
            event_logger: Optional TraceRecorder for diagnostics

        Note:
            The default orthogonalization implements Notay (2000) Algorithm 2.1 with
            m_max=10, using the periodic restart formula m_i = max(1, i mod (m_max + 1)).
        """
        # Set defaults
        self.preconditioner = preconditioner or IdentityPreconditioner()
        self.orthogonalization = orthogonalization or PeriodicRestartOrthogonalization(
            m_max=DEFAULT_M_MAX
        )
        self.convergence_criterion = (
            convergence_criterion
            or CombinedToleranceCriterion(rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
        )
        self.event_logger = event_logger

    # Implement KrylovSolverBase abstract method

    def _update_direction(
        self,
        w: NDArray,
        state: KrylovState,
        **kwargs,
    ) -> NDArray:
        """Compute FCG search direction using orthogonalization strategy.

        This is the core FCG operation from Notay (2000) Algorithm 2.1:
        orthogonalize the preconditioned residual against previous directions
        to maintain approximate A-conjugacy.

        Formula (Notay 2000, Algorithm 2.1):
            d_i = w_i - Σ_{j=i-m_i}^{i-1} [(w_i, A d_j) / (d_j, A d_j)] * d_j

        where:
            - w_i = M^{-1}(r_i) is the preconditioned residual
            - m_i = max(1, i mod (m_max + 1)) is the orthogonalization window
            - The sum is over the last m_i directions (truncated Gram-Schmidt)

        Args:
            w: Preconditioned residual w_i (Notay 2000)
            state: Current CGState with direction_history containing (d_j, A d_j) vectors
            **kwargs: Additional parameters

        Returns:
            Orthogonalized search direction d_i (Notay 2000)

        Theory:
            Orthogonalization maintains approximate A-conjugacy:
                d_i^T A d_j ≈ 0 for i ≠ j

            This ensures the search directions span the Krylov subspace without
            numerical linear dependence, enabling convergence even with variable
            or non-SPD preconditioners.

        References:
            Notay (2000) Section 2, Algorithm 2.1
        """
        # Access direction_history from CGState via Protocol
        from ..models.protocols import HasDirectionHistory

        if isinstance(state, HasDirectionHistory):
            history = state.direction_history
        else:
            # For base KrylovState, use empty history
            from ..models.history import DirectionHistory

            history = DirectionHistory.empty(max_size=0)
        d_vectors = list(history.d_vectors)  # Convert tuple to list
        q_vectors = list(history.q_vectors)  # Convert tuple to list

        # Apply primary orthogonalization
        d_orthog, _ = self.orthogonalization.orthogonalize(
            vector=w,
            d_vectors=d_vectors,
            q_vectors=q_vectors,
        )

        return d_orthog

    # Implement IterativeSolverBase abstract methods

    def _validate_system(
        self,
        A: Callable[[NDArray], NDArray] | NDArray,
        b: NDArray,
        x0: NDArray | None,
        **kwargs,
    ) -> None:
        """Validate system inputs.

        Args:
            A: Linear operator
            b: Right-hand side
            x0: Initial guess
            **kwargs: Solver parameters

        Raises:
            ValueError: If inputs invalid
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
        **kwargs,
    ) -> Callable[[NDArray], NDArray]:
        """Prepare linear operator.

        Args:
            A: Linear operator (matrix or callable)
            b: Right-hand side
            **kwargs: Solver parameters

        Returns:
            Callable linear operator
        """
        # If already callable, return as is
        if callable(A) and not isinstance(A, np.ndarray):
            return A  # type: ignore[return-value]

        # Convert to LinearOperator for consistency
        linear_op = aslinearoperator(A)
        return linear_op.matvec

    def _normalize_tolerances(
        self,
        b: NDArray,
        **kwargs,
    ) -> tuple[float, float]:
        """Normalize tolerances.

        Args:
            b: Right-hand side
            **kwargs: Must contain 'rtol' and 'atol'

        Returns:
            Tuple (rtol, atol)
        """
        rtol = kwargs.get("rtol", DEFAULT_RTOL)
        atol = kwargs.get("atol", DEFAULT_ATOL)
        return float(rtol), float(atol)

    def _initialize_state(
        self,
        linear_op: Callable[[NDArray], NDArray],
        b: NDArray,
        x0: NDArray | None,
        **kwargs,
    ) -> CGState:
        """Initialize FCG state.

        Args:
            linear_op: Prepared linear operator
            b: Right-hand side
            x0: Initial guess
            **kwargs: Solver parameters (may contain max_history)

        Returns:
            Initial CGState with u_0, r_0, and empty histories
        """
        b_arr = np.asarray(b, dtype=np.float64)
        n = b_arr.shape[0]

        # Initial guess u_0 (Notay 2000)
        u = np.asarray(x0, dtype=np.float64) if x0 is not None else np.zeros(n)

        # Initial residual r_0
        r = b_arr - linear_op(u)
        r_norm = float(norm(r))

        # Apply preconditioner w_0 = M^{-1}(r_0) (Notay 2000)
        w = self._apply_preconditioner(self.preconditioner, r)

        # Initial direction (steepest descent) d_0 (Notay 2000)
        d = w.copy()

        # Matrix-vector product q_0 = A d_0
        q = linear_op(d)

        # Get window size from orthogonalization strategy
        max_history = getattr(self.orthogonalization, "window_size", DEFAULT_M_MAX)

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

    def _check_stopping(
        self,
        state: SolverState,
        rtol: float,
        atol: float,
        maxiter: int,
        **kwargs,
    ) -> bool:
        """Check if solver should stop.

        Args:
            state: Current CGState
            rtol: Relative tolerance
            atol: Absolute tolerance
            maxiter: Maximum iterations

        Returns:
            True if should stop, False otherwise
        """
        # Check convergence
        if state.converged:
            return True

        # Check breakdown
        if state.breakdown:
            return True

        # Check divergence
        if state.divergence:
            return True

        # Check max iterations
        if maxiter is not None and state.iteration >= maxiter:
            return True

        # Dependency injection: create criterion once, delegate to base class helper
        criterion = self._create_convergence_criterion(rtol, atol)
        if self._check_convergence_with_criterion(state, criterion):
            return True

        return False

    def _build_result(
        self,
        state: SolverState,
        rtol: float,
        atol: float,
        **kwargs,
    ) -> SolverResult:
        """Build final SolverResult.

        Args:
            state: Final SolverState
            rtol: Relative tolerance
            atol: Absolute tolerance
            **kwargs: Solver parameters

        Returns:
            SolverResult with convergence info and diagnostics
        """
        # Dependency injection: create criterion once, delegate to base class helper
        from ..models.protocols import HasDirectionHistory

        criterion = self._create_convergence_criterion(rtol, atol)
        converged = self._check_convergence_with_criterion(state, criterion)

        # Extract histories from event log using base class helper
        event_log = self._get_event_logger(**kwargs)
        (
            residual_history_abs,
            residual_history_rel,
            residual_vectors,
            solution_vectors,
        ) = self._extract_histories_from_event_log(event_log, state.rhs_norm)

        # Fallback to CGState.residual_history if event_log unavailable (backward compatibility)
        if residual_history_abs is None and isinstance(state, HasDirectionHistory):
            residual_history_abs = list(state.residual_history.norms_abs)
            # Compute relative residuals
            if state.rhs_norm > 0:
                residual_history_rel = [
                    r / state.rhs_norm for r in residual_history_abs
                ]
            else:
                residual_history_rel = residual_history_abs

        return SolverResult(
            converged=converged,
            iterations=state.iteration,
            residual=state.residual_norm / state.rhs_norm
            if state.rhs_norm > 0
            else 0.0,
            residual_abs=state.residual_norm,
            rhs_norm=state.rhs_norm,
            breakdown=state.breakdown,
            residual_history=residual_history_rel,  # Return relative residuals
            residual_history_abs=residual_history_abs,  # Also provide absolute
            tol=rtol,
            atol=atol,
            event_log=event_log,  # Attach the event log
            residual_vectors=residual_vectors,  # Vector traces if FULL mode
            solution_vectors=solution_vectors,  # Vector traces if FULL mode
        )
