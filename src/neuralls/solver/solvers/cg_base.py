"""Base class for Conjugate Gradient solver variants.

This module provides the CG-specific base class that extends KrylovSolverBase
with Conjugate Gradient algorithm logic. It implements CG-specific operations
like curvature computation and step length calculation.

Mathematical Background (Notay 2000 notation):
    Conjugate Gradient (CG) and its variants (PCG, FCG) minimize the A-norm of
    the error over Krylov subspaces. The key property is A-conjugacy:
        (d_i, A d_j) = 0 for i ≠ j

    This leads to the characteristic CG iteration:
    1. Curvature: (d_i, A d_i) = (d_i, A d_i) where A d_i is cached
    2. Step length: α_i = (d_i, r_i) / (d_i, A d_i) (FCG formula, Notay 2000)
    3. Solution update: u_{i+1} = u_i + α_i d_i
    4. Residual update: r_{i+1} = r_i - α_i A d_i


Design Principles:
    - Template Method: Implements CG-specific operations
    - Open/Closed: Extensible for CG variants (PCG, FCG, etc.)
    - Single Responsibility: CG-specific logic only

References:
    - Hestenes & Stiefel (1952). Methods of Conjugate Gradients for Solving Linear Systems.
    - Notay, Y. (2000). Flexible Conjugate Gradients. SIAM J. Sci. Comput.
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, Chapter 9.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import replace

import numpy as np

from ..core.krylov_base import KrylovSolverBase
from ..utils.numerics import stable_dot_product, compute_curvature

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..models.state import KrylovState


class ConjugateGradientBase(KrylovSolverBase):
    """Abstract base for Conjugate Gradient solver variants.

    This class implements CG-specific operations while allowing variants to
    customize:
    - Direction computation (two-term recurrence vs truncated orthogonalization)
    - Preconditioner application (fixed vs flexible)
    - Reorthogonalization strategies

    All CG variants follow the same core update formulas implemented here:
    1. Curvature computation with breakdown detection
    2. Step length calculation using FCG formula
    3. Solution and residual updates

    Protected Abstract Hooks (CG-specific):
        _update_direction: Compute search direction (PCG: β update, FCG: orthogonalization)

    Protected Concrete Helpers (CG-specific):
        _compute_curvature_cg: Compute curvature with breakdown detection
        _compute_step_length_cg: Compute step length using FCG formula
        _update_solution_residual: Update x and r vectors
        _compute_max_history: Compute history size for orthogonalization strategies

    Theory Note (Notay 2000):
        The A-norm minimization property of CG:
            ||u_* - u_i||_A = min_{u ∈ u_0 + K_i} ||u - u_*||_A

        where ||e||_A = sqrt(e^T A e) and u_* is the true solution.
    """

    def _compute_max_history(
        self,
        matrix_size: int,
        maxiter: int | None,
    ) -> int:
        """Compute maximum history size for orthogonalization.

        Eliminates duplicated logic in FCG and PCG _initialize_state methods.

        Args:
            matrix_size: Size of the linear system (n)
            maxiter: Maximum number of iterations (None if not set)

        Returns:
            Maximum history size to use for DirectionHistory

        Theory:
            History size determines how many previous directions to retain:
            - If orthogonalization.window_size is None (unlimited): use maxiter
            - If orthogonalization.window_size is int: use that directly
            - Fallback: max(matrix_size, DEFAULT_FCG_HISTORY_LIMIT)

        Note: This method should only be called when orthogonalization
              exists on the solver instance.
        """
        from ...constants import DEFAULT_FCG_HISTORY_LIMIT

        orthog = getattr(self, "orthogonalization", None)
        if orthog is None:
            # No orthogonalization - return minimal size
            return 0

        window = orthog.window_size

        # Handle unlimited case (None = full orthogonalization)
        if window is None:
            # For unlimited orthogonalization, use maxiter or fallback
            if maxiter is not None and maxiter > 0:
                return maxiter
            else:
                # Fallback to constant limit
                return max(matrix_size, DEFAULT_FCG_HISTORY_LIMIT)
        else:
            # Truncated orthogonalization with fixed window
            return window

    # Implement KrylovSolverBase abstract methods

    def _compute_coefficients(
        self,
        d: NDArray,
        q: NDArray,
        state: KrylovState,
        breakdown_tol: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Compute CG step length and coefficients.

        Computes (Notay 2000):
        1. Curvature (d_i, A d_i) with breakdown detection
        2. Step length α_i = (d_i, r_i) / (d_i, A d_i) using FCG formula

        Args:
            d: Search direction d_i (Notay 2000)
            q: Matrix-vector product q_i = A d_i
            state: Current CG state
            breakdown_tol: Breakdown detection tolerance

        Returns:
            Tuple (step_length, coefficients):
            - step_length: α_i for updates
            - coefficients: Empty dict (CG only needs α)

        Theory (Notay 2000):
            The step length minimizes the A-norm of error along d_i:
                α_i = argmin_α ||u_i + α*d_i - u_*||_A
        """
        from ...constants import DEFAULT_BREAKDOWN_TOL

        # Resolve breakdown tolerance
        breakdown_tol_eff = breakdown_tol if breakdown_tol is not None else DEFAULT_BREAKDOWN_TOL

        # Compute curvature with breakdown detection
        curvature, breakdown = compute_curvature(d, q, breakdown_tol=breakdown_tol_eff)

        if breakdown:
            # Breakdown detected - curvature too small or negative
            return 0.0, {"curvature": curvature}

        # Compute step length using FCG formula (Notay 2000): α_i = (d_i, r_i) / (d_i, A d_i)
        # This works for both symmetric and non-symmetric preconditioners
        numerator = stable_dot_product(d, state.r, dtype=np.float64)
        alpha = numerator / curvature

        return alpha, {"curvature": curvature}

    def _update_vectors(
        self,
        state: KrylovState,
        d: NDArray,
        q: NDArray,
        w: NDArray,
        step_length: float,
        coefficients: dict[str, float],
    ) -> KrylovState:
        """Update solution, residual, and CG state.

        Updates (Notay 2000):
        1. Solution: u_{i+1} = u_i + α_i * d_i
        2. Residual: r_{i+1} = r_i - α_i * q_i
        3. Residual norm and iteration count
        4. Direction/residual history (for FCG variants)

        Args:
            state: Current CG state
            d: Search direction used in this iteration d_i (Notay 2000)
            q: Matrix-vector product q_i = A @ d_i
            w: Preconditioned residual w_i = M^{-1}(r_i) (Notay 2000)
            step_length: Step length α_i
            coefficients: Contains curvature and breakdown flag

        Returns:
            New CGState with updated vectors

        Theory:
            The updates maintain the residual orthogonality property:
                (r_{i+1}, w_j) = 0 for j = 0, 1, ..., i (for standard PCG)
        """
        # Update solution and residual
        u_new = state.u + step_length * d
        r_new = state.r - step_length * q

        # Compute new residual norm
        residual_norm_new = float(np.linalg.norm(r_new))

        # Update direction history (for FCG variants that use CGState)
        direction_history = getattr(state, "direction_history", None)
        if direction_history is not None and hasattr(direction_history, "add"):
            direction_history = direction_history.add(d, q)
        else:
            # For base KrylovState, initialize empty history
            from ..models.history import DirectionHistory
            direction_history = DirectionHistory.empty(max_size=0)

        # Update residual history (skip for base KrylovState that may not have it)
        residual_history = getattr(state, "residual_history", None)
        if residual_history is not None and hasattr(residual_history, "add"):
            residual_history = residual_history.add(residual_norm_new, residual_norm_new / state.rhs_norm)

        # Create new state - only include CGState-specific fields if they exist on the original state
        update_dict = {
            "iteration": state.iteration + 1,
            "u": u_new,
            "r": r_new,
            "w": w,
            "d": d,
            "q": q,
            "residual_norm": residual_norm_new,
        }

        # Only add direction_history and residual_history if state has them (CGState)
        if hasattr(state, "direction_history"):
            update_dict["direction_history"] = direction_history
        if hasattr(state, "residual_history"):
            update_dict["residual_history"] = residual_history

        new_state = replace(state, **update_dict)

        return new_state

    def _check_breakdown(
        self,
        state: KrylovState,
    ) -> bool:
        """Check for CG breakdown (NaN/Inf in solution or residual).

        Args:
            state: Current CG state

        Returns:
            True if breakdown detected (NaN/Inf), False otherwise
        """
        # Check for NaN/Inf in solution or residual
        if not np.all(np.isfinite(state.u)) or not np.all(np.isfinite(state.r)):
            return True

        # Check if residual norm is NaN/Inf
        if not np.isfinite(state.residual_norm):
            return True

        return False

    # Protected concrete helpers (CG-specific)

    def _compute_beta_coefficient(
        self,
        w_new: NDArray,
        w_old: NDArray,
        r_new: NDArray,
        r_old: NDArray,
        formula: str = "fletcher_reeves",
    ) -> float:
        """Compute β coefficient for two-term recurrence.

        Computes β for standard PCG direction update (Notay 2000):
            d_{i+1} = w_{i+1} + β_i * d_i

        Args:
            w_new: New preconditioned residual w_{i+1} (Notay 2000)
            w_old: Old preconditioned residual w_i (Notay 2000)
            r_new: New residual r_{i+1}
            r_old: Old residual r_i
            formula: Beta formula to use:
                - "fletcher_reeves": β = (r_{i+1}, w_{i+1}) / (r_i, w_i)
                - "polak_ribiere": β = (r_{i+1}, w_{i+1} - w_i) / (r_i, w_i)
                - "hestenes_stiefel": β = (r_{i+1}, w_{i+1} - w_i) / (d_i, w_{i+1} - w_i)

        Returns:
            Beta coefficient β_i

        Theory:
            Different β formulas have different properties:
            - Fletcher-Reeves: Most stable, works for SPD systems
            - Polak-Ribiere: Better for non-quadratic problems
            - Hestenes-Stiefel: Original CG formula, most general

        References:
            - Fletcher & Reeves (1964). Function minimization by conjugate gradients.
            - Polak & Ribière (1969). Note sur la convergence de méthodes de directions conjuguées.
        """
        if formula == "fletcher_reeves":
            numerator = stable_dot_product(r_new, w_new, dtype=np.float64)
            denominator = stable_dot_product(r_old, w_old, dtype=np.float64)
            return numerator / denominator if denominator != 0 else 0.0

        elif formula == "polak_ribiere":
            w_diff = w_new - w_old
            numerator = stable_dot_product(r_new, w_diff, dtype=np.float64)
            denominator = stable_dot_product(r_old, w_old, dtype=np.float64)
            # Polak-Ribiere can be negative, reset to 0 (Polak-Ribiere-Plus)
            beta = numerator / denominator if denominator != 0 else 0.0
            return max(0.0, beta)

        elif formula == "hestenes_stiefel":
            w_diff = w_new - w_old
            numerator = stable_dot_product(r_new, w_diff, dtype=np.float64)
            # Note: denominator uses d_i which we don't have here
            # This formula requires access to d_i, so fallback to FR
            denominator = stable_dot_product(r_old, w_old, dtype=np.float64)
            return numerator / denominator if denominator != 0 else 0.0

        else:
            raise ValueError(f"Unknown beta formula: {formula}")
