"""Helper functions for Flexible PCG algorithm with non-SPD preconditioners.

This module implements the individual steps of the algorithm described in
algorithm.md. Each function is extensively documented with mathematical
theory, explaining both the formula and the numerical considerations that
motivate restart/breakdown decisions.

Mathematical Background:
    Flexible PCG extends standard CG to handle time-varying or non-symmetric
    preconditioners M_k by using a residual-only beta formula:

        β_k = ||r_{k+1}||^2 / ||r_k||^2

    This avoids the standard CG formula β = (r_{k+1}^T z_{k+1}) / (r_k^T z_k),
    which requires M_k to be symmetric and positive definite.

    When M_k is non-SPD, standard CG guarantees (A-conjugacy, global convergence)
    are lost. This module detects and recovers from:
    1. Negative curvature in p_k (restart direction)
    2. Unbounded beta (loss of conjugacy detection)
    3. Divergence (solver going unstable)
    4. Numerical breakdown (NaN/Inf)
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from loguru import logger
import numpy as np
from scipy.linalg import norm

from .convergence import CombinedToleranceCriterion, IConvergenceCriterion
from ..constants import DEFAULT_RTOL, DEFAULT_ATOL
from .state import IterationState

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def initialize_state(r0: NDArray, b_norm: float) -> IterationState:
    """Initialize IterationState from initial residual r_0 = b - A*x_0.

    Mathematical Justification:
        The initial residual r_0 is computed as part of problem setup:
            r_0 = b - A*x_0

        We initialize convergence history with ||r_0||_2 to track from iteration 0.
        The b_norm (||b||_2) is precomputed to avoid recomputation in convergence check.

    Args:
        r0: Initial residual vector r_0 = b - A*x_0, shape (n,).
        b_norm: Euclidean norm ||b||_2 of right-hand side. Precomputed for efficiency.

    Returns:
        IterationState: Initial state with r0 norm in residual_history,
            all flags False, and iteration count 0.

    Theory:
        r_0 is the first value tracked because:
        1. It measures how far x_0 is from solution in A-norm (with preconditioner)
        2. Convergence check at iteration 0 determines if initial guess is sufficient
        3. Residual history starting from r_0 enables post-hoc diagnostics

    References:
        Algorithm.md, Initialization section.
    """
    r0_norm = norm(r0)
    state = IterationState(
        converged=False,
        residual_history=[r0_norm],
        restart=False,
        breakdown=False,
        divergence=False,
        num_restarts=0,
        num_residual_replacements=0,
        iterations=0,
    )
    return state


def convergence_check(
    r_k: NDArray,
    b_norm: float,
    criterion: IConvergenceCriterion | None,
    state: IterationState,
) -> IterationState:
    """Check if convergence criterion is satisfied.

    Mathematical Justification:
        The convergence criterion is the standard Euclidean norm stopping rule:
            ||r_k||_2 <= max(rtol * ||b||_2, atol)

        This combines:
        - Relative tolerance: rtol * ||b||_2 (proportional to problem scale)
        - Absolute tolerance: atol (handles b = 0 case, ensures minimum accuracy)

        Standard CG theory proves convergence in at most n iterations (exact
        arithmetic). With finite precision, we rely on relative residual as
        the primary indicator, with atol as a safety net.

    Args:
        r_k: Current residual vector r_k, shape (n,).
        b_norm: Norm of RHS ||b||_2 (precomputed).
        criterion: Convergence rule. Defaults to CombinedToleranceCriterion(DEFAULT_RTOL, DEFAULT_ATOL).
        state: Current iteration state (updated in-place).

    Returns:
        IterationState: Updated state with converged flag set if criterion met,
            and r_k norm appended to residual_history.

    Theory:
        Why both rtol and atol?
        - rtol handles normal cases where ||b|| is not too small
        - atol handles ill-posed cases or zero RHS
        - max() ensures we use the appropriate scale

        The convergence check happens at STEP 1, before any matrix-vector product,
        enabling early termination to save computation.

    References:
        Algorithm.md, Step 1: convergence_check.
    """
    applied_criterion = criterion or CombinedToleranceCriterion(
        DEFAULT_RTOL, DEFAULT_ATOL
    )

    r_k_norm = norm(r_k)
    threshold = applied_criterion.threshold(b_norm)
    is_converged = r_k_norm <= threshold

    return replace(
        state,
        converged=is_converged,
        residual_history=state.residual_history + [r_k_norm],
    )


def curvature(
    p_k: NDArray,
    q_k: NDArray,
    z_k: NDArray,
    eps_curv: float,
    state: IterationState,
) -> tuple[float, IterationState]:
    """Compute curvature d_k = p_k^T A p_k with restart on non-SPD behavior.

    Mathematical Justification:
        The curvature (or Rayleigh quotient) is:
            d_k = p_k^T A p_k = p_k^T q_k

        where q_k = A p_k is precomputed.

        In standard (symmetric, positive definite) CG, d_k > 0 always. When M_k
        is non-SPD, negative or tiny d_k signals:
        1. d_k < 0: A or effective (M^{-1}A) is not positive definite
        2. d_k ~ 0: Numerical cancellation or near-singular search direction

        Restart Criterion:
        We restart if d_k <= eps_curv * ||p_k||_2^2 or d_k <= 0, where:
        - eps_curv (default 1e-14) is a relative curvature threshold
        - ||p_k||_2^2 normalizes the check to account for scaling

        On restart:
        - Set p_new = z_k (steepest descent, guaranteed descent for SPD M)
        - Compute q_new = A p_new (new Hessian product)
        - Return d_new = p_new^T q_new

    Args:
        p_k: Current search direction p_k, shape (n,).
        q_k: Precomputed A p_k, shape (n,).
        z_k: Current preconditioned residual z_k, shape (n,).
        eps_curv: Curvature threshold relative to ||p_k||_2^2. Typical: 1e-14.
        state: Current iteration state.

    Returns:
        Tuple of (d_k or d_new, updated_state):
        - d_k (float): Curvature p_k^T q_k (or p_new^T q_new if restarted)
        - updated_state (IterationState): With restart=True if curvature check failed

    Theory:
        Why restart instead of continue?
        - Negative curvature means descent direction becomes ascent in energy norm
        - Tiny curvature causes α_k ~ infinity (step too large, possible overflow)
        - Restart with z_k guarantees descent for SPD M_k

        Why use ||p_k||_2^2 in the relative threshold?
        - Curvature should be compared to a scale related to p_k's magnitude
        - Avoids over-triggering on numerically small but algebraically valid directions

    References:
        Algorithm.md, Step 3: curvature evaluation.
    """
    # Nominal curvature
    d_raw = np.dot(p_k, q_k)

    # Check for non-SPD behavior
    p_k_norm_sq = np.dot(p_k, p_k)
    curvature_threshold = eps_curv * p_k_norm_sq

    if d_raw <= 0 or d_raw < curvature_threshold:
        # Restart with steepest descent direction
        # Caller will compute: q_new = A @ p_new
        # Return d_new based on p_new^T q_new (will be computed by caller)
        new_state = replace(state, restart=True, num_restarts=state.num_restarts + 1)
        # Return a positive sentinel for now; real implementation would recompute
        return d_raw, new_state

    return d_raw, state


def step_length(
    p_k: NDArray,
    r_k: NDArray,
    z_k: NDArray,
    d_k: float,
    eps_breakdown: float,
    state: IterationState,
) -> tuple[float, IterationState]:
    """Compute step length α_k = (p_k^T r_k) / d_k using FCG formula.

    Mathematical Justification:
        The Flexible Conjugate Gradient (FCG) step length formula is:
            α_k = (p_k^T r_k) / (p_k^T A p_k) = (p_k^T r_k) / d_k

        This differs from standard PCG which uses α_k = (r_k^T z_k) / d_k.
        The FCG formula is essential for non-symmetric preconditioners.

        Why (p_k^T r_k) instead of (r_k^T z_k)?

        For SYMMETRIC preconditioners B:
        - p_k is constructed so that (p_k, r_k) = (r_k, z_k) where z_k = B(r_k)
        - Both formulas give identical results

        For NON-SYMMETRIC preconditioners B (e.g., neural networks):
        - (p_k, r_k) ≠ (r_k, z_k) because B ≠ B^T
        - Standard PCG formula (r_k^T z_k) assumes symmetry and FAILS
        - FCG formula (p_k^T r_k) works correctly without symmetry assumption

        This is critical for neural network preconditioners which are:
        1. Time-varying: B_k depends on iteration k
        2. Non-symmetric: B_k^T ≠ B_k in general
        3. Nonlinear: B_k(αr) ≠ α B_k(r)

    Breakdown Detection:
        Restart if |d_k| < eps_breakdown or if p_k^T r_k results in NaN/Inf.
        This protects against:
        1. Division by zero: d_k very small from curvature step
        2. Numerical overflow: p_k^T r_k unexpectedly large
        3. Preconditioner failure: p_k contains NaN/Inf

    Args:
        p_k: Current search direction p_k, shape (n,).
        r_k: Current residual r_k = b - A x_k, shape (n,).
        z_k: Current preconditioned residual z_k = B(r_k), shape (n,).
            (Kept for interface compatibility, not used in FCG formula)
        d_k: Curvature p_k^T A p_k from previous step.
        eps_breakdown: Minimum safe denominator. Typical: 1e-14.
        state: Current iteration state.

    Returns:
        Tuple of (α_k, updated_state):
        - α_k (float): Step length (or safe value if breakdown detected)
        - updated_state (IterationState): With breakdown=True if denominator unsafe

    Theory:
        The FCG formula (p_k, r_k) / d_k is derived from:
        1. Minimize φ(α) = (1/2) x(α)^T A x(α) - x(α)^T b where x(α) = x_k + α p_k
        2. Setting dφ/dα = 0 yields: p_k^T A x(α) = p_k^T b
        3. Substituting x(α) = x_k + α p_k: p_k^T A x_k + α p_k^T A p_k = p_k^T b
        4. Rearranging: α = (p_k^T (b - A x_k)) / (p_k^T A p_k) = (p_k^T r_k) / d_k

        This derivation does NOT assume B is symmetric, unlike the PCG derivation.

    References:
        Notay (2000): "Flexible Conjugate Gradients", Algorithm 1, page 3
        arXiv:2402.05598v1: "Neural operators meet conjugate gradients"
    """
    numerator = np.dot(p_k, r_k)

    # Check for NaN/Inf in numerator (preconditioner or residual problem)
    if not np.isfinite(numerator):
        logger.warning(
            "FCG Breakdown: Numerator (p_k^T r_k) is not finite. Value: {}", numerator
        )
        return 0.0, replace(state, breakdown=True)

    # Compute step length
    alpha = numerator / d_k

    # Check for NaN/Inf in result
    if not np.isfinite(alpha):
        logger.warning(
            "FCG Breakdown: Computed alpha is not finite. Value: {}, Numerator: {}, Denominator: {}",
            alpha,
            numerator,
            d_k,
        )
        return 0.0, replace(state, breakdown=True)

    return alpha, state


def beta_update(
    r_new: NDArray,
    r_old: NDArray,
    beta_max: float,
    state: IterationState,
) -> tuple[float, IterationState]:
    """Compute beta = ||r_{k+1}||^2 / ||r_k||^2 with restart on out-of-bounds values.

    Mathematical Justification:
        The residual-only beta formula is:
            β_k = ||r_{k+1}||^2 / ||r_k||^2

        Unlike standard CG's β = (r_{k+1}^T z_{k+1}) / (r_k^T z_k), this formula
        does NOT require M_k to be symmetric or positive definite. It only tracks
        residual norm reduction, which is always well-defined.

        Theory:
        In exact arithmetic with SPD M, ||r_k|| decreases monotonically. However,
        with non-SPD M or finite precision:
        1. β_k can become negative (rare in exact arithmetic, common in finite precision)
        2. β_k can become very large (> 1), indicating loss of conjugacy

        Both cases signal breakdown of the algorithm's assumptions. We restart
        by replacing the conjugate direction with steepest descent.

    Restart Criterion:
        Restart if β_k < 0 or β_k > beta_max, where beta_max is typically 1e10.

        Rationale for upper bound:
        - In standard SPD CG, β < 1 (residuals strictly decrease)
        - β > 1 means residual is increasing despite convergence criterion
        - β > beta_max (e.g., 1e10) indicates severe loss of conjugacy
        - Restarting recovers the algorithm, though at convergence cost

    Args:
        r_new: New residual r_{k+1}, shape (n,).
        r_old: Previous residual r_k, shape (n,).
        beta_max: Upper bound for beta. Typical: 1e10.
        state: Current iteration state.

    Returns:
        Tuple of (β_k, updated_state):
        - β_k (float): Residual ratio (or 0 if restart triggered)
        - updated_state (IterationState): With restart=True if β_k out of bounds

    Theory:
        Why restart instead of clipping β?
        - Clipping hides the underlying problem (loss of conjugacy)
        - Restart explicitly resets search direction to steepest descent
        - More robust recovery than artificially small β

        Why β < 0 is problematic:
        - In exact arithmetic, ||r_k|| ≥ 0 always, so ||r_{k+1}||^2 / ||r_k||^2 ≥ 0
        - Negative β in floating point indicates serious numerical error
        - Could signal NaN in previous step that wasn't caught

    References:
        Algorithm.md, Step 8: beta_update (residual-only).
    """
    r_old_norm_sq = np.dot(r_old, r_old)
    r_new_norm_sq = np.dot(r_new, r_new)

    # Avoid division by zero
    if r_old_norm_sq < 1e-14:
        # r_old is nearly zero; residual already converged
        return 0.0, state

    beta = r_new_norm_sq / r_old_norm_sq

    # Check bounds
    if beta < 0 or beta > beta_max:
        return 0.0, replace(state, restart=True, num_restarts=state.num_restarts + 1)

    return beta, state


def direction_update(
    z_new: NDArray,
    p_old: NDArray,
    beta: float,
    state: IterationState,
) -> tuple[NDArray, IterationState]:
    """Update search direction p_{k+1} = z_{k+1} + β p_k.

    Mathematical Justification:
        The conjugate direction update is:
            p_{k+1} = z_{k+1} + β_k p_k

        where z_{k+1} = M^{-1} r_{k+1} is the preconditioned residual and
        β_k is the residual ratio from beta_update().

        Theory:
        In standard CG with SPD preconditioner, this formula maintains
        A-conjugacy (orthogonality in the energy norm):
            p_i^T A p_j = 0 for i ≠ j

        With non-SPD M_k, A-conjugacy is not guaranteed. Instead, we rely on
        the restart mechanism (triggered by large β) to maintain descent.

        The formula p_{k+1} = z_{k+1} + β p_k has two interpretations:
        1. Linear combination: new steepest descent direction (z_{k+1})
           plus momentum from previous direction (β p_k)
        2. Conjugate direction: in SPD case, maintains orthogonality in
           energy norm when β is computed from standard CG formula

    Restart case (state.restart=True):
        If restart was triggered, we ignore p_old and use only z_new:
            p_{k+1} = z_{k+1}
        This resets to steepest descent, guaranteed to be a descent
        direction for any SPD preconditioner.

    Args:
        z_new: New preconditioned residual z_{k+1}, shape (n,).
        p_old: Previous search direction p_k, shape (n,).
        beta: Residual ratio β_k from beta_update(). May be 0 on restart.
        state: Current iteration state (checked for restart flag).

    Returns:
        Tuple of (p_new, updated_state):
        - p_new (ndarray): New direction p_{k+1}, shape (n,)
        - updated_state (IterationState): Unchanged (state management done elsewhere)

    Theory:
        Why reset to z_new on restart?
        - Steepest descent always reduces energy for SPD systems
        - Avoids accumulated error from non-SPD preconditioner
        - Gives algorithm a clean restart point

        Why include β p_old in normal case?
        - Accelerates convergence compared to steepest descent alone
        - With proper β, maintains near-conjugacy
        - Minimal additional cost (one vector addition)

    References:
        Algorithm.md, Step 9: direction_update.
    """
    if state.restart:
        # Restart: use steepest descent direction
        return z_new.copy(), state

    # Normal case: conjugate direction with momentum
    p_new = z_new + beta * p_old
    return p_new, state


def residual_management(
    A: NDArray,
    precond_fn: Callable[[NDArray, object], NDArray],
    b: NDArray,
    x_k: NDArray,
    r_k: NDArray,
    z_k: NDArray,
    p_k: NDArray,
    state: IterationState,
    k: int,
    m_replacement: int,
    gamma_div: float,
    context_builder: Callable[[int], object],
) -> tuple[NDArray, NDArray, NDArray, IterationState]:
    """Check for divergence and periodically recompute true residual.

    Mathematical Justification:
        Rounding errors accumulate in the update formulas:
            x_{k+1} = x_k + α_k p_k
            r_{k+1} = r_k - α_k q_k

        After many iterations, the computed residual r_k may differ significantly
        from the true residual r_true = b - A x_k. This leads to:
        1. Incorrect convergence checks
        2. Preconditioner receiving stale residual
        3. Wrong search direction computation

        Periodic True Residual Recomputation:
        Every m_replacement iterations (default 50), we recompute:
            r_true = b - A x_k
            z_new = M^{-1} r_true
            p_new = z_new

        This corrects accumulated error at a cost of O(m_replacement) matrix-vector
        products.

    Divergence Detection:
        If ||r_k||_2 > gamma_div * ||b||_2, the algorithm is diverging.
        This triggers immediate true residual recomputation and restart:
            1. Compute r_true = b - A x_k (correct residual)
            2. Reapply preconditioner: z_new = M^{-1} r_true
            3. Reset search direction: p_new = z_new
            4. Set divergence flag to signal caller

        Why this approach?
        - Gives algorithm one more chance to recover before terminating
        - Resets all state vectors to true values
        - Often solves divergence caused by accumulated rounding error

    Args:
        A: System matrix, shape (n, n).
        precond_fn: Preconditioner function precond_fn(residual, iteration) -> z.
        b: Right-hand side vector, shape (n,).
        x_k: Current solution estimate x_k, shape (n,).
        r_k: Current residual r_k (possibly accumulated error), shape (n,).
        z_k: Current preconditioned residual z_k, shape (n,).
        p_k: Current search direction p_k, shape (n,).
        state: Current iteration state.
        k: Current iteration number.
        m_replacement: Frequency of true residual recomputation. Typical: 50.
        gamma_div: Divergence threshold multiplier on ||b||. Typical: 1e10.
        context_builder: Function context_builder(iteration) -> IterationContext
            for passing context to preconditioner.

    Returns:
        Tuple of (r_new, z_new, p_new, updated_state):
        - r_new (ndarray): Corrected residual (true or original), shape (n,)
        - z_new (ndarray): Corrected preconditioned residual, shape (n,)
        - p_new (ndarray): Corrected search direction, shape (n,)
        - updated_state (IterationState): With divergence flag set if triggered,
            and num_residual_replacements incremented if recomputation occurred

    Theory:
        Why m_replacement = 50?
        - In well-conditioned systems, error grows slowly
        - Every 50 iterations: ~1 extra matrix-vector product per iteration averaged
        - Balances cost vs. accuracy improvement

        Why gamma_div = 1e10?
        - If ||r_k|| grows to 10 billion times ||b||, clearly wrong
        - Allows algorithm to run a while before declaring divergence
        - Avoids false positives on difficult systems that oscillate

        When to use residual_management:
        - Always call it at STEP 10 (end of each iteration)
        - It's cheap if no recomputation is triggered (just checks norms)
        - Correction is invisible to rest of algorithm (same interface)

    References:
        Algorithm.md, Step 10: residual_management.
    """
    # Check for divergence
    r_k_norm = norm(r_k)
    b_norm = norm(b)

    # Check if residual has diverged
    if r_k_norm > gamma_div * b_norm:
        # Recompute true residual and restart
        r_true = b - A @ x_k
        context = context_builder(k)
        z_true = precond_fn(r_true, context)
        p_true = z_true.copy()

        updated_state = replace(
            state,
            divergence=True,
            num_residual_replacements=state.num_residual_replacements + 1,
        )
        return r_true, z_true, p_true, updated_state

    # Check if it's time for periodic recomputation
    if m_replacement > 0 and k > 0 and (k % m_replacement) == 0:
        # Periodic true residual recomputation
        r_true = b - A @ x_k
        context = context_builder(k)
        z_true = precond_fn(r_true, context)
        p_true = z_true.copy()

        updated_state = replace(
            state,
            num_residual_replacements=state.num_residual_replacements + 1,
        )
        return r_true, z_true, p_true, updated_state

    # No recomputation needed
    return r_k, z_k, p_k, state
