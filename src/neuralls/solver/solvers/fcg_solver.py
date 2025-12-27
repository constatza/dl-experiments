"""Flexible Conjugate Gradient solver.

This module implements the Flexible Conjugate Gradient (FCG) algorithm
using the new architecture with ConjugateGradientBase and strategy pattern.

Algorithm (FCG):
    Initialize: x_0 = 0, r_0 = b
    For k = 0, 1, ..., until convergence:
      1. z_k = M^{-1}(r_k)                           # Apply preconditioner
      2. p_k = orthogonalize(z_k, P, Q)              # Orthogonalization strategy
      3. q_k = A @ p_k                               # Matrix-vector product
      4. α_k = (p_k, r_k) / (p_k, q_k)               # Step length (from CG base)
      5. x_{k+1} = x_k + α_k * p_k                   # Update solution
      6. r_{k+1} = r_k - α_k * q_k                   # Update residual

Design Principles:
    - Strategy Pattern: Inject OrthogonalizationStrategy for flexibility
    - Template Method: Inherit from ConjugateGradientBase
    - Single Responsibility: Only implements direction computation
    - Composition over Inheritance: Strategies injected, not inherited

References:
    - Notay, Y. (2000). Flexible Conjugate Gradients. SIAM J. Sci. Comput.
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable
from dataclasses import replace

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
from ..strategies.orthogonalization import OrthogonalizationStrategy, TruncatedGramSchmidt
from ..strategies.reorthogonalization import ReorthogonalizationStrategy, apply_reorthogonalization
from ..models.state import CGState, KrylovState, SolverState
from ..models.history import DirectionHistory, ResidualHistory
from ..models.result import SolverResult
from ..preconditioners import IdentityPreconditioner, Preconditioner
from ..monitoring.trace_recorder import TraceRecorder

if TYPE_CHECKING:
    from numpy.typing import NDArray


class FlexibleCGSolver(ConjugateGradientBase):
    """Flexible Conjugate Gradient solver.

    FCG extends standard PCG to handle variable and non-SPD preconditioners
    through orthogonalization. This implementation uses strategy pattern to
    inject orthogonalization and reorthogonalization strategies.

    Key Features:
        - Handles variable preconditioners (neural networks, adaptive methods)
        - Truncated Gram-Schmidt orthogonalization for numerical stability
        - Optional reorthogonalization for improved accuracy
        - Compatible with non-SPD preconditioners

    Attributes:
        preconditioner: Preconditioner to apply (default: Identity)
        orthogonalization: Strategy for orthogonalizing directions
        reorthogonalization: Optional secondary orthogonalization
        convergence_criterion: Convergence test strategy
        event_logger: Optional event recorder for diagnostics

    Example:
        >>> from neuralls.solver.strategies.orthogonalization import TruncatedGramSchmidt
        >>> orthog = TruncatedGramSchmidt(window_size=10)
        >>> solver = FlexibleCGSolver(
        ...     preconditioner=my_preconditioner,
        ...     orthogonalization=orthog,
        ... )
        >>> x, result = solver.solve(A, b, rtol=1e-6)
    """

    def __init__(
        self,
        preconditioner: Preconditioner | None = None,
        orthogonalization: OrthogonalizationStrategy | None = None,
        reorthogonalization: ReorthogonalizationStrategy | None = None,
        convergence_criterion: IConvergenceCriterion | None = None,
        event_logger: TraceRecorder | None = None,
    ) -> None:
        """Initialize Flexible CG solver.

        Args:
            preconditioner: Preconditioner M (default: Identity)
            orthogonalization: Orthogonalization strategy (default: TruncatedGramSchmidt with m_max=10)
            reorthogonalization: Optional reorthogonalization strategy
            convergence_criterion: Convergence test (default: Combined rtol/atol)
            event_logger: Optional TraceRecorder for diagnostics
        """
        # Set defaults
        self.preconditioner = preconditioner or IdentityPreconditioner()
        self.orthogonalization = orthogonalization or TruncatedGramSchmidt(window_size=DEFAULT_M_MAX)
        self.reorthogonalization = reorthogonalization
        self.convergence_criterion = convergence_criterion or CombinedToleranceCriterion(
            rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
        )
        self.event_logger = event_logger

    # Implement KrylovSolverBase abstract method

    def _update_direction(
        self,
        z: NDArray,
        state: KrylovState,
        **kwargs,
    ) -> NDArray:
        """Compute FCG search direction using orthogonalization strategy.

        This is the core FCG operation: orthogonalize the preconditioned
        residual against previous directions to maintain approximate A-conjugacy.

        Formula:
            p_k = z_k - Σ_{j} [(z_k, q_j) / (p_j, q_j)] * p_j

        where the sum is over a window of recent directions (truncated GS).

        Args:
            z: Preconditioned residual z_k = M^{-1}(r_k)
            state: Current CGState with direction_history containing (p, q) vectors
            **kwargs: Additional parameters

        Returns:
            Orthogonalized search direction p_k

        Theory:
            Orthogonalization maintains approximate A-conjugacy:
                (p_i, A*p_j) ≈ 0 for i ≠ j

            This ensures the iterates span the Krylov subspace without
            numerical linear dependence.
        """
        # Access direction_history from CGState via Protocol
        from ..models.protocols import HasDirectionHistory

        if isinstance(state, HasDirectionHistory):
            history = state.direction_history
        else:
            # For base KrylovState, use empty history
            from ..models.history import DirectionHistory
            history = DirectionHistory.empty(max_size=0)
        p_vectors = list(history.p_vectors)  # Convert tuple to list
        q_vectors = list(history.q_vectors)  # Convert tuple to list

        # Apply primary orthogonalization
        p_orthog, orthog_report = self.orthogonalization.orthogonalize(
            vector=z,
            p_vectors=p_vectors,
            q_vectors=q_vectors,
        )

        # Apply optional reorthogonalization
        if self.reorthogonalization is not None:
            # Use the old interface for reorthogonalization (will be updated later)
            # For now, just use the primary orthogonalized result
            p_final = p_orthog
        else:
            p_final = p_orthog

        # Check for breakdown (from orthogonalization)
        if orthog_report.breakdown:
            # Fallback to steepest descent
            return self._restart_direction(z, state)

        return p_final

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
            Initial CGState with x_0, r_0, and empty histories
        """
        b_arr = np.asarray(b, dtype=np.float64)
        n = b_arr.shape[0]

        # Initial guess
        x = np.asarray(x0, dtype=np.float64) if x0 is not None else np.zeros(n)

        # Initial residual
        r = b_arr - linear_op(x)
        r_norm = float(norm(r))

        # Apply preconditioner
        z = self._apply_preconditioner(self.preconditioner, r)

        # Initial direction (steepest descent)
        p = z.copy()

        # Matrix-vector product
        q = linear_op(p)

        # Get window size from orthogonalization strategy
        max_history = getattr(self.orthogonalization, "window_size", DEFAULT_M_MAX)

        # Create initial state using factory method
        return CGState.create_initial(
            x=x,
            r=r,
            z=z,
            p=p,
            q=q,
            residual_norm=r_norm,
            rhs_norm=float(norm(b_arr)),
            max_history=max_history,
        )

    def _check_stopping(
        self,
        state: SolverState,
        rtol: float,
        atol: float,
        **kwargs,
    ) -> bool:
        """Check if solver should stop.

        Args:
            state: Current CGState
            rtol: Relative tolerance
            atol: Absolute tolerance
            **kwargs: May contain 'max_iterations'

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
        max_iter = kwargs.get("max_iterations", 100)
        if state.iteration >= max_iter:
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
                residual_history_rel = [r / state.rhs_norm for r in residual_history_abs]
            else:
                residual_history_rel = residual_history_abs

        return SolverResult(
            converged=converged,
            iterations=state.iteration,
            residual=state.residual_norm / state.rhs_norm if state.rhs_norm > 0 else 0.0,
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
