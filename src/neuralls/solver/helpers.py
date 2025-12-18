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


def _stable_dot_product(
    a: NDArray, b: NDArray, dtype: type[np.floating] = np.float64  # type: ignore[type-arg]
) -> float:
    """Compute dot product with overflow prevention via balanced scaling.

    Mathematically equivalent to np.dot(a, b) but scales both vectors
    by the same factor to prevent overflow, then adjusts the result.

    Mathematical Theory:
        For dot(a, b), if we compute dot(a/s, b/s), we get:
            sum((a_i/s) * (b_i/s)) = sum(a_i * b_i) / s^2 = dot(a,b) / s^2
        So: dot(a, b) = s^2 * dot(a/s, b/s)

        This is mathematically exact, just reordered for numerical stability.

    Numerical Stability Behaviors:
        **Overflow Prevention:**
        - Threshold: Uses sqrt(0.1 * finfo.max) ≈ 1.3e154 for float64
        - Why: If max(|a_i|, |b_i|) > 1.3e154, individual products could overflow
        - Action: Scale both vectors to safe range, compute, then rescale result
        - Trade-off: Slight rounding error vs catastrophic overflow

        **Underflow Handling:**
        - Small vectors (e.g., 1e-300): May underflow to exact zero
        - Why acceptable: Zero is a valid dot product (e.g., orthogonal vectors)
        - Detection: Caller should check result with np.isfinite() if needed

        **Precision Preservation:**
        - Normal range (~1e-100 to 1e100): No scaling, exact np.dot() result
        - Moderate range (~1e100 to 1e154): No scaling, full precision maintained
        - Only extreme values trigger scaling, minimizing precision loss

    Threshold Justification:
        Uses 0.1 * finfo.max (≈1.8e307 for float64) as overflow threshold:
        - Leaves 10x safety margin for intermediate calculations
        - Follows SciPy convention of fractional max for numerical headroom
        - Prevents false triggers on legitimate large-scale problems

    Args:
        a: First vector for dot product, shape (n,).
        b: Second vector for dot product, shape (n,).
        dtype: Data type for overflow threshold calculation (default: np.float64).

    Returns:
        float: Dot product a^T b, computed stably to avoid overflow.

    Examples:
        >>> # Normal case - no scaling
        >>> _stable_dot_product(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        11.0

        >>> # Extreme case - automatic scaling prevents overflow
        >>> a = np.array([1e150, 1e150, 1e150])
        >>> b = np.array([1e150, 1e150, 1e150])
        >>> result = _stable_dot_product(a, b)  # Would overflow with np.dot
        >>> np.isfinite(result)
        True

    References:
        SciPy conventions for numerical stability in iterative solvers.
    """
    max_a = np.max(np.abs(a))
    max_b = np.max(np.abs(b))

    # Get dtype-specific overflow threshold (no magic values)
    from ..constants import get_overflow_threshold

    overflow_threshold = get_overflow_threshold(dtype)

    # Check for overflow risk using logarithms to avoid intermediate overflow
    # If log(max_a) + log(max_b) > log(threshold), then max_a * max_b > threshold
    # Use this only if both values are large enough that multiplication might overflow
    sqrt_threshold = np.sqrt(overflow_threshold)  # ~1.3e154 for float64

    if max_a > sqrt_threshold or max_b > sqrt_threshold:
        # At least one value is large - need to scale
        # Scale both vectors by the same factor to keep product safe
        # Want: max(max_a, max_b) * scale ≈ sqrt(sqrt_threshold)
        max_val = max(max_a, max_b)
        target = np.sqrt(sqrt_threshold)  # ~1.1e77 for float64
        scale = target / max_val
        # Mathematically: dot(a,b) = (1/scale^2) * dot(a*scale, b*scale)
        return np.dot(a * scale, b * scale) / (scale * scale)

    return np.dot(a, b)


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

        Breakdown Detection (Proactive + Reactive):
        **Proactive - Absolute Threshold (NEW):**
        - Threshold: eps^2 ≈ 4.93e-32 for float64 (SciPy convention)
        - Why eps^2: Detects when curvature lost all numerical significance
        - When triggered: |d_k| < 4.93e-32
        - Action: Restart with p = z (steepest descent)
        - Justification: Below machine precision squared, curvature value is
          indistinguishable from zero in floating point arithmetic

        **Reactive - Relative Threshold (Existing):**
        - Threshold: eps_curv * ||p_k||_2^2 (default eps_curv = 1e-14)
        - Why relative: Accounts for problem scaling and direction magnitude
        - When triggered: d_k < 1e-14 * ||p_k||_2^2 or d_k <= 0
        - Action: Restart with p = z
        - Justification: Relative check catches non-SPD behavior in scaled problems

        **Restart vs Breakdown:**
        - Absolute check (eps^2) → restart (recoverable via steepest descent)
        - Relative check (eps_curv) → restart (non-SPD preconditioner behavior)
        - Both use restart, not breakdown, because steepest descent can recover

    Numerical Stability Behaviors:
        **Overflow Prevention:**
        - Uses _stable_dot_product for p_k^T q_k
        - Prevents overflow when preconditioner produces large directions
        - Critical for neural network preconditioners with unbounded outputs

        **Two-Level Protection:**
        - Level 1 (absolute): Catches numerical precision loss
        - Level 2 (relative): Catches mathematical non-SPD behavior
        - Both trigger restart, allowing algorithm to recover

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
    # Compute curvature using stable dot product to prevent overflow
    d_raw = _stable_dot_product(p_k, q_k)

    # Check for breakdown using SciPy convention
    from ..constants import get_breakdown_tol

    breakdown_tol = get_breakdown_tol(np.float64)

    if np.abs(d_raw) < breakdown_tol:
        logger.warning("FCG Breakdown: Curvature d_raw={:.2e} below tolerance", d_raw)
        new_state = replace(state, restart=True, num_restarts=state.num_restarts + 1)
        return d_raw, new_state

    # Check for non-SPD behavior using relative threshold
    # Use norm() instead of dot(p_k, p_k) for better numerical stability
    p_k_norm_sq = norm(p_k) ** 2
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

    Breakdown Detection (Proactive + Reactive):
        **Proactive - Denominator Check (NEW):**
        - Threshold: eps^2 ≈ 4.93e-32 for float64 (SciPy BICG/BICGSTAB convention)
        - Why eps^2: Square of machine epsilon catches loss of precision in squared norms
        - When triggered: |d_k| < 4.93e-32
        - Action: Return alpha=0, set breakdown=True, terminate solver
        - Justification: Below eps^2, division would amplify rounding errors beyond
          machine precision, making results meaningless

        **Reactive - NaN/Inf Detection:**
        - Checks numerator (p_k^T r_k) and final alpha for NaN/Inf
        - Catches overflow that slipped through stable dot product
        - Indicates preconditioner produced unbounded outputs

        **Why This Order:**
        1. Check d_k first - prevents division by near-zero
        2. Compute numerator with stable dot product - prevents overflow
        3. Check numerator validity - catches preconditioner failures
        4. Compute division - now safe
        5. Final NaN/Inf check - defensive safety net

    Numerical Stability Behaviors:
        **Underflow in Numerator:**
        - If p_k ⊥ r_k (perpendicular): numerator = 0.0 exactly
        - Result: alpha = 0.0 (valid, no breakdown)
        - Why valid: Zero step means current direction doesn't help

        **Overflow Prevention:**
        - Uses _stable_dot_product for p_k^T r_k
        - Prevents overflow from large preconditioner outputs
        - Maintains mathematical correctness via balanced scaling

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
    # Check denominator for vanishing/breakdown following SciPy BICG convention
    # SciPy uses: if np.abs(rho_cur) < rhotol where rhotol = eps**2
    from ..constants import get_breakdown_tol

    breakdown_tol = get_breakdown_tol(np.float64)

    if np.abs(d_k) < breakdown_tol:
        logger.warning(
            "FCG Breakdown: Curvature d_k={:.2e} below breakdown tolerance {:.2e}",
            d_k,
            breakdown_tol,
        )
        return 0.0, replace(state, breakdown=True)

    # Compute numerator using stable dot product to prevent overflow
    numerator = _stable_dot_product(p_k, r_k)

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

    Breakdown Detection (Proactive):
        **Denominator Underflow Check (NEW):**
        - Threshold: eps^2 ≈ 4.93e-32 for float64
        - Why eps^2: ||r_old||^2 is already squared, so compare to eps^2
        - When triggered: ||r_old||^2 < 4.93e-32
        - Action: Return beta=0, set restart=True, increment restart counter
        - Justification: When squared norm underflows below machine precision
          squared, we've lost all information about the residual magnitude

        **Physical Interpretation:**
        - For 3-element vector: each element must be > 1.28e-16 to avoid trigger
        - Values below this threshold: Either truly zero or lost to rounding
        - Why restart: Residual norm below eps^2 means either:
          a) Converged (but should have been caught by convergence check)
          b) Numerical precision exhausted (need fresh start)

    Restart Criterion (Reactive):
        **Beta Out of Bounds:**
        - Restart if β_k < 0 or β_k > beta_max (typically 1e10)
        - Negative beta: Impossible in exact arithmetic, indicates rounding error
        - Large beta: Loss of conjugacy, residual increasing despite progress
        - Action: Return beta=0, restart with steepest descent

    Numerical Stability Behaviors:
        **Underflow Protection:**
        - Extreme underflow (e.g., 1e-300): ||r||^2 → 0.0, triggers breakdown
        - Small but finite (e.g., 1e-10): ||r||^2 ≈ 3e-20 > eps^2, computes normally
        - Threshold prevents division: 0.0 / 0.0 → NaN

        **Overflow Prevention:**
        - Uses norm() instead of dot(r, r) for squared norms
        - norm() is more numerically stable (LAPACK-based)
        - Avoids intermediate overflow in dot product computation

        **Why norm() > dot(r, r):**
        - norm() uses optimized BLAS/LAPACK routines with overflow guards
        - dot(r, r) is direct summation, vulnerable to overflow
        - Both mathematically equivalent, but norm() has better numerical properties

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
    # Use scipy.linalg.norm() which is more numerically stable than dot(r,r)
    # for computing squared norms
    r_old_norm_sq = norm(r_old) ** 2
    r_new_norm_sq = norm(r_new) ** 2

    # Avoid division by zero using SciPy's breakdown tolerance
    from ..constants import get_breakdown_tol

    breakdown_tol = get_breakdown_tol()

    if r_old_norm_sq < breakdown_tol:
        logger.warning("FCG Breakdown: ||r_old||^2={:.2e} below tolerance", r_old_norm_sq)
        return 0.0, replace(state, restart=True, num_restarts=state.num_restarts + 1)

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
    """Residual management disabled - returns inputs unchanged.

    This function previously performed:
    - Periodic true residual recomputation (every m_replacement iterations)
    - Divergence detection and recovery

    Now disabled to allow pure CG algorithm without artificial interventions.
    Only the restart mechanism (for non-SPD preconditioners) remains active.

    Args:
        All arguments maintained for API compatibility.

    Returns:
        Tuple of (r_k, z_k, p_k, state) unchanged.
    """
    # Disabled - return inputs unchanged
    return r_k, z_k, p_k, state
