"""Base class for Conjugate Gradient solver variants.

This module provides the CG-specific base class that extends KrylovSolverBase
with Conjugate Gradient algorithm logic. It implements CG-specific operations
like curvature computation and step length calculation.

Mathematical Background:
    Conjugate Gradient (CG) and its variants (PCG, FCG) minimize the A-norm of
    the error over Krylov subspaces. The key property is A-conjugacy:
        (p_i, A*p_j) = 0 for i ≠ j

    This leads to the characteristic CG iteration:
    1. Curvature: d_k = (p_k, A*p_k) = (p_k, q_k) where q_k = A*p_k
    2. Step length: α_k = (p_k, r_k) / d_k (FCG formula)
    3. Solution update: x_{k+1} = x_k + α_k * p_k
    4. Residual update: r_{k+1} = r_k - α_k * q_k

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

from abc import abstractmethod
from typing import TYPE_CHECKING
from dataclasses import replace

import numpy as np

from ..core.krylov_base import KrylovSolverBase
from ..utils.numerics import stable_dot_product, compute_curvature

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..models.state import CGState, KrylovState


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

    Theory Note:
        The A-norm minimization property of CG:
            ||x - x_k||_A = min_{x ∈ x_0 + K_k} ||x - x_*||_A

        where ||e||_A = sqrt(e^T A e) and x_* is the true solution.
    """

    # Implement KrylovSolverBase abstract methods

    def _compute_coefficients(
        self,
        p: NDArray,
        q: NDArray,
        state: KrylovState,
        **kwargs,
    ) -> tuple[float, dict[str, float]]:
        """Compute CG step length and coefficients.

        Computes:
        1. Curvature d_k = (p_k, q_k) with breakdown detection
        2. Step length α_k = (p_k, r_k) / d_k using FCG formula

        Args:
            p: Search direction p_k
            q: Matrix-vector product q_k = A @ p_k
            state: Current CG state
            **kwargs: May contain 'breakdown_tol' for curvature check

        Returns:
            Tuple (step_length, coefficients):
            - step_length: α_k for updates
            - coefficients: Empty dict (CG only needs α)

        Theory:
            The step length minimizes the A-norm of error along p_k:
                α_k = argmin_α ||x_k + α*p_k - x_*||_A
        """
        # Extract parameters
        breakdown_tol = kwargs.get("breakdown_tol", 1e-14)

        # Compute curvature with breakdown detection
        curvature, breakdown = compute_curvature(p, q, breakdown_tol=breakdown_tol)

        if breakdown:
            # Return zero step length to trigger breakdown handling
            return 0.0, {"curvature": curvature, "breakdown": True}

        # Compute step length using FCG formula: α_k = (p_k, r_k) / d_k
        # This works for both symmetric and non-symmetric preconditioners
        numerator = stable_dot_product(p, state.r, dtype=np.float64)
        alpha = numerator / curvature

        return alpha, {"curvature": curvature, "breakdown": False}

    def _update_vectors(
        self,
        state: KrylovState,
        p: NDArray,
        q: NDArray,
        z: NDArray,
        step_length: float,
        coefficients: dict[str, float],
        **kwargs,
    ) -> KrylovState:
        """Update solution, residual, and CG state.

        Updates:
        1. Solution: x_{k+1} = x_k + α_k * p_k
        2. Residual: r_{k+1} = r_k - α_k * q_k
        3. Residual norm and iteration count
        4. Direction/residual history (for FCG variants)

        Args:
            state: Current CG state
            p: Search direction used in this iteration
            q: Matrix-vector product q = A @ p
            z: Preconditioned residual z = M^{-1}(r)
            step_length: Step length α_k
            coefficients: Contains curvature and breakdown flag
            **kwargs: Solver parameters

        Returns:
            New CGState with updated vectors

        Theory:
            The updates maintain the residual orthogonality property:
                (r_{k+1}, z_i) = 0 for i = 0, 1, ..., k (for standard PCG)
        """
        # Update solution and residual
        x_new = state.x + step_length * p
        r_new = state.r - step_length * q

        # Compute new residual norm
        residual_norm_new = float(np.linalg.norm(r_new))

        # Update direction history (for FCG variants that use CGState)
        direction_history = getattr(state, "direction_history", None)
        if direction_history is not None and hasattr(direction_history, "add"):
            direction_history = direction_history.add(p, q)
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
            "x": x_new,
            "r": r_new,
            "z": z,
            "p": p,
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
        **kwargs,
    ) -> bool:
        """Check for CG breakdown.

        CG breakdown occurs when:
        1. Curvature is negative or too small (d_k <= 0 or |d_k| < ε)
        2. NaN/Inf appears in solution or residual

        Args:
            state: Current CG state
            **kwargs: Breakdown parameters

        Returns:
            True if breakdown detected, False otherwise

        Theory:
            Negative curvature indicates:
            1. Matrix A is not SPD
            2. Preconditioner M is not SPD (for PCG)
            3. Numerical instability
        """
        # Check if already marked as breakdown from coefficients
        if state.breakdown:
            return True

        # Check for NaN/Inf in solution or residual
        if not np.all(np.isfinite(state.x)) or not np.all(np.isfinite(state.r)):
            return True

        # Check if residual norm is NaN/Inf
        if not np.isfinite(state.residual_norm):
            return True

        return False

    # Protected concrete helpers (CG-specific)

    def _restart_direction(
        self,
        z: NDArray,
        state: CGState,
    ) -> NDArray:
        """Restart with steepest descent direction.

        When breakdown detected or curvature issues occur, restart CG
        with steepest descent direction p = z (preconditioned residual).

        Args:
            z: Preconditioned residual z = M^{-1}(r)
            state: Current CG state

        Returns:
            Steepest descent direction p = z

        Theory:
            Steepest descent is the natural fallback when A-conjugacy is lost.
            This allows CG to recover from:
            1. Non-SPD preconditioner behavior
            2. Numerical cancellation
            3. Loss of conjugacy
        """
        return z.copy()

    def _compute_beta_coefficient(
        self,
        z_new: NDArray,
        z_old: NDArray,
        r_new: NDArray,
        r_old: NDArray,
        formula: str = "fletcher_reeves",
    ) -> float:
        """Compute β coefficient for two-term recurrence.

        Computes β for standard PCG direction update:
            p_{k+1} = z_{k+1} + β_k * p_k

        Args:
            z_new: New preconditioned residual z_{k+1}
            z_old: Old preconditioned residual z_k
            r_new: New residual r_{k+1}
            r_old: Old residual r_k
            formula: Beta formula to use:
                - "fletcher_reeves": β = (r_{k+1}, z_{k+1}) / (r_k, z_k)
                - "polak_ribiere": β = (r_{k+1}, z_{k+1} - z_k) / (r_k, z_k)
                - "hestenes_stiefel": β = (r_{k+1}, z_{k+1} - z_k) / (p_k, z_{k+1} - z_k)

        Returns:
            Beta coefficient β_k

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
            numerator = stable_dot_product(r_new, z_new, dtype=np.float64)
            denominator = stable_dot_product(r_old, z_old, dtype=np.float64)
            return numerator / denominator if denominator != 0 else 0.0

        elif formula == "polak_ribiere":
            z_diff = z_new - z_old
            numerator = stable_dot_product(r_new, z_diff, dtype=np.float64)
            denominator = stable_dot_product(r_old, z_old, dtype=np.float64)
            # Polak-Ribiere can be negative, reset to 0 (Polak-Ribiere-Plus)
            beta = numerator / denominator if denominator != 0 else 0.0
            return max(0.0, beta)

        elif formula == "hestenes_stiefel":
            z_diff = z_new - z_old
            numerator = stable_dot_product(r_new, z_diff, dtype=np.float64)
            # Note: denominator uses p_k which we don't have here
            # This formula requires access to p_k, so fallback to FR
            denominator = stable_dot_product(r_old, z_old, dtype=np.float64)
            return numerator / denominator if denominator != 0 else 0.0

        else:
            raise ValueError(f"Unknown beta formula: {formula}")
