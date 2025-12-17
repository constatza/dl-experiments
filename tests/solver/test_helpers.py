"""Tests for flexible PCG helper functions.

This module tests all 7 helper functions that implement the individual
steps of the flexible PCG algorithm. Tests cover both good-path behavior
and essential error/edge cases.

Test Coverage:
1. initialize_state() - proper initialization
2. convergence_check() - convergence detection
3. curvature() - restart on negative/small curvature
4. step_length() - breakdown detection
5. beta_update() - restart on large/negative beta
6. direction_update() - proper direction computation
7. residual_management() - divergence detection, periodic replacement
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.linalg import norm

from neuralls.constants import (
    DEFAULT_ATOL,
    DEFAULT_BETA_MAX,
    DEFAULT_CURVATURE_EPSILON,
    DEFAULT_DIVERGENCE_FACTOR,
    DEFAULT_RESIDUAL_REPLACEMENT_FREQ,
)
from neuralls.solver.helpers import (
    beta_update,
    convergence_check,
    curvature,
    direction_update,
    initialize_state,
    residual_management,
    step_length,
)
from neuralls.solver.convergence import CombinedToleranceCriterion
from neuralls.solver.state import IterationState

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


# =============================================================================
# Test Data Fixtures for Helper Functions
# =============================================================================


@pytest.fixture
def residual_vector_small() -> NDArray:
    """Small residual vector for testing.

    Returns:
        5D residual vector with ||r||_2 = 1.0.

    Theory:
        Unit norm residual simplifies convergence checks.
    """
    r = np.array([0.6, 0.0, 0.8, 0.0, 0.0], dtype=np.float64)
    return r / norm(r)  # Normalize to unit norm


@pytest.fixture
def residual_decreasing_small() -> NDArray:
    """Smaller residual vector (decreasing from residual_vector_small).

    Returns:
        5D residual vector with ||r||_2 = 0.1.
    """
    r = np.array([0.06, 0.0, 0.08, 0.0, 0.0], dtype=np.float64)
    return r / norm(r) * 0.1


@pytest.fixture
def rhs_unit_norm() -> NDArray:
    """RHS vector with unit norm.

    Returns:
        5D RHS vector with ||b||_2 = 1.0.
    """
    b = np.array([0.6, 0.0, 0.8, 0.0, 0.0], dtype=np.float64)
    return b / norm(b)


@pytest.fixture
def search_direction_small() -> NDArray:
    """Small search direction vector.

    Returns:
        5D search direction vector.
    """
    return np.array([1.0, 0.5, 0.3, 0.2, 0.1], dtype=np.float64)


@pytest.fixture
def preconditioned_residual_small() -> NDArray:
    """Small preconditioned residual vector.

    Returns:
        5D preconditioned residual z_k.
    """
    return np.array([0.8, 0.4, 0.2, 0.1, 0.05], dtype=np.float64)


@pytest.fixture
def matrix_vector_product_positive(search_direction_small: NDArray) -> NDArray:
    """Matrix-vector product with positive curvature.

    Args:
        search_direction_small: Search direction p_k.

    Returns:
        5D vector q_k = A p_k with p_k^T q_k > 0.

    Theory:
        For SPD matrix A, p^T A p > 0 always.
        Here we construct q to ensure positive curvature.
    """
    # Construct q such that p^T q > 0
    return search_direction_small * 2.0


@pytest.fixture
def matrix_vector_product_negative(search_direction_small: NDArray) -> NDArray:
    """Matrix-vector product with negative curvature.

    Args:
        search_direction_small: Search direction p_k.

    Returns:
        5D vector q_k with p_k^T q_k < 0.

    Theory:
        Negative curvature indicates non-SPD behavior.
        Triggers restart mechanism.
    """
    # Construct q such that p^T q < 0
    return -search_direction_small * 2.0


@pytest.fixture
def base_state() -> IterationState:
    """Base state for helper function tests.

    Returns:
        IterationState with basic setup.
    """
    return IterationState(
        converged=False,
        residual_history=[1.0],
        restart=False,
        breakdown=False,
        divergence=False,
        num_restarts=0,
        num_residual_replacements=0,
        iterations=0,
    )


# =============================================================================
# Test initialize_state()
# =============================================================================


def test_initialize_state_basic(residual_vector_small: NDArray, rhs_unit_norm: NDArray) -> None:
    """Test initialize_state creates correct initial state.

    Args:
        residual_vector_small: Initial residual r_0.
        rhs_unit_norm: RHS vector for b_norm.

    Theory:
        Initial state must have:
        - Empty residual history (will be populated by convergence_check)
        - All flags False
        - Zero counters
        - Iteration 0
    """
    b_norm = norm(rhs_unit_norm)
    state = initialize_state(residual_vector_small, b_norm)

    assert not state.converged
    assert len(state.residual_history) == 1
    assert state.residual_history[0] == pytest.approx(norm(residual_vector_small))
    assert not state.restart
    assert not state.breakdown
    assert not state.divergence
    assert state.num_restarts == 0
    assert state.num_residual_replacements == 0
    assert state.iterations == 0


def test_initialize_state_zero_residual() -> None:
    """Test initialize_state with zero residual (already converged).

    Theory:
        If r_0 = 0, system is already solved (x_0 is exact solution).
        State should have residual norm = 0.
    """
    r0 = np.zeros(5, dtype=np.float64)
    b_norm = 1.0

    state = initialize_state(r0, b_norm)

    assert state.residual_history[0] == pytest.approx(0.0)
    assert state.iterations == 0


# =============================================================================
# Test convergence_check()
# =============================================================================


def test_convergence_check_not_converged(
    residual_vector_small: NDArray,
    rhs_unit_norm: NDArray,
    base_state: IterationState,
) -> None:
    """Test convergence_check with residual above tolerance.

    Args:
        residual_vector_small: Residual with ||r|| = 1.0.
        rhs_unit_norm: RHS with ||b|| = 1.0.
        base_state: Initial state.

    Theory:
        Convergence criterion: ||r|| <= max(rtol * ||b||, atol).
        With ||r|| = 1.0, ||b|| = 1.0, rtol = 1e-6, atol = 1e-14:
        threshold = max(1e-6 * 1.0, 1e-14) = 1e-6.
        Since 1.0 > 1e-6, not converged.
    """
    b_norm = norm(rhs_unit_norm)
    rtol = 1e-6
    atol = DEFAULT_ATOL

    criterion = CombinedToleranceCriterion(rtol=rtol, atol=atol)
    state = convergence_check(residual_vector_small, b_norm, criterion, base_state)

    assert not state.converged
    assert len(state.residual_history) == 2  # Base had [1.0], now has [1.0, norm(r)]


def test_convergence_check_converged_relative(
    residual_decreasing_small: NDArray,
    rhs_unit_norm: NDArray,
    base_state: IterationState,
) -> None:
    """Test convergence_check with residual below relative tolerance.

    Args:
        residual_decreasing_small: Residual with ||r|| = 0.1.
        rhs_unit_norm: RHS with ||b|| = 1.0.
        base_state: Initial state.

    Theory:
        With ||r|| = 0.1, ||b|| = 1.0, rtol = 0.2:
        threshold = max(0.2 * 1.0, 1e-14) = 0.2.
        Since 0.1 <= 0.2, converged.
    """
    b_norm = norm(rhs_unit_norm)
    rtol = 0.2  # Loose tolerance for testing
    atol = DEFAULT_ATOL

    criterion = CombinedToleranceCriterion(rtol=rtol, atol=atol)
    state = convergence_check(residual_decreasing_small, b_norm, criterion, base_state)

    assert state.converged


def test_convergence_check_converged_absolute() -> None:
    """Test convergence_check with residual below absolute tolerance.

    Theory:
        With ||r|| = 1e-15, ||b|| = 1.0, rtol = 1e-6, atol = 1e-14:
        threshold = max(1e-6 * 1.0, 1e-14) = 1e-6.
        Since 1e-15 <= 1e-6, converged via atol.
    """
    r = np.full(5, 1e-15, dtype=np.float64)
    b_norm = 1.0
    rtol = 1e-6
    atol = 1e-14
    state = IterationState(
        converged=False,
        residual_history=[],
        restart=False,
        breakdown=False,
        divergence=False,
        num_restarts=0,
        num_residual_replacements=0,
        iterations=0,
    )

    criterion = CombinedToleranceCriterion(rtol=rtol, atol=atol)
    result = convergence_check(r, b_norm, criterion, state)

    assert result.converged


def test_convergence_check_zero_rhs() -> None:
    """Test convergence_check with zero RHS (b=0, x=0 is solution).

    Theory:
        For b = 0, solution is x = 0, so r = 0.
        Convergence should be immediate via atol.
    """
    r = np.zeros(5, dtype=np.float64)
    b_norm = 0.0
    rtol = 1e-6
    atol = 1e-14
    state = IterationState(
        converged=False,
        residual_history=[],
        restart=False,
        breakdown=False,
        divergence=False,
        num_restarts=0,
        num_residual_replacements=0,
        iterations=0,
    )

    criterion = CombinedToleranceCriterion(rtol=rtol, atol=atol)
    result = convergence_check(r, b_norm, criterion, state)

    assert result.converged


# =============================================================================
# Test curvature()
# =============================================================================


def test_curvature_positive(
    search_direction_small: NDArray,
    matrix_vector_product_positive: NDArray,
    preconditioned_residual_small: NDArray,
    base_state: IterationState,
) -> None:
    """Test curvature computation with positive curvature (no restart).

    Args:
        search_direction_small: Search direction p_k.
        matrix_vector_product_positive: A p_k with p^T A p > 0.
        preconditioned_residual_small: Preconditioned residual z_k.
        base_state: Initial state.

    Theory:
        For SPD matrix, p^T A p > 0 always.
        No restart should be triggered.
    """
    eps_curv = DEFAULT_CURVATURE_EPSILON

    d_k, state = curvature(
        search_direction_small,
        matrix_vector_product_positive,
        preconditioned_residual_small,
        eps_curv,
        base_state,
    )

    # Positive curvature computed correctly
    expected_d = np.dot(search_direction_small, matrix_vector_product_positive)
    assert d_k == pytest.approx(expected_d)
    assert d_k > 0

    # No restart
    assert not state.restart
    assert state.num_restarts == 0


def test_curvature_negative_triggers_restart(
    search_direction_small: NDArray,
    matrix_vector_product_negative: NDArray,
    preconditioned_residual_small: NDArray,
    base_state: IterationState,
) -> None:
    """Test curvature with negative curvature triggers restart.

    Args:
        search_direction_small: Search direction p_k.
        matrix_vector_product_negative: A p_k with p^T A p < 0.
        preconditioned_residual_small: Preconditioned residual z_k.
        base_state: Initial state.

    Theory:
        Negative curvature indicates non-SPD behavior.
        Algorithm must restart with steepest descent direction.
    """
    eps_curv = DEFAULT_CURVATURE_EPSILON

    d_k, state = curvature(
        search_direction_small,
        matrix_vector_product_negative,
        preconditioned_residual_small,
        eps_curv,
        base_state,
    )

    # Negative curvature detected
    expected_d = np.dot(search_direction_small, matrix_vector_product_negative)
    assert expected_d < 0

    # Restart triggered
    assert state.restart
    assert state.num_restarts == 1


def test_curvature_small_triggers_restart(
    preconditioned_residual_small: NDArray,
    base_state: IterationState,
) -> None:
    """Test curvature with tiny curvature triggers restart.

    Args:
        preconditioned_residual_small: Preconditioned residual z_k.
        base_state: Initial state.

    Theory:
        Very small curvature causes division by near-zero in step length.
        Restart avoids numerical overflow.
    """
    # Construct search direction and product with very small curvature
    p_k = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q_k = np.array([1e-16, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)  # p^T q ~ 1e-16
    eps_curv = DEFAULT_CURVATURE_EPSILON

    d_k, state = curvature(p_k, q_k, preconditioned_residual_small, eps_curv, base_state)

    # Small curvature detected
    expected_d = np.dot(p_k, q_k)
    assert expected_d > 0
    assert expected_d < eps_curv * np.dot(p_k, p_k)

    # Restart triggered
    assert state.restart
    assert state.num_restarts == 1


# =============================================================================
# Test step_length()
# =============================================================================


def test_step_length_valid(
    residual_vector_small: NDArray,
    preconditioned_residual_small: NDArray,
    base_state: IterationState,
) -> None:
    """Test step_length computation with valid inputs.

    Args:
        residual_vector_small: Residual r_k.
        preconditioned_residual_small: Preconditioned residual z_k.
        base_state: Initial state.

    Theory:
        Step length: α_k = (p_k^T r_k) / d_k (FCG formula).
        For first iteration, p_k = z_k, so (p_k^T r_k) = (z_k^T r_k).
        With valid inputs, should return positive α.
    """
    d_k = 2.0  # Positive curvature
    eps_breakdown = 1e-14
    # For first iteration: p = z (steepest descent direction)
    p = preconditioned_residual_small

    alpha, state = step_length(
        p,
        residual_vector_small,
        preconditioned_residual_small,
        d_k,
        eps_breakdown,
        base_state,
    )

    # Valid alpha computed using FCG formula: (p^T r) / d_k
    expected_alpha = np.dot(p, residual_vector_small) / d_k
    assert alpha == pytest.approx(expected_alpha)
    assert alpha > 0

    # No breakdown
    assert not state.breakdown


def test_step_length_breakdown_small_denominator(
    residual_vector_small: NDArray,
    preconditioned_residual_small: NDArray,
    base_state: IterationState,
) -> None:
    """Test step_length with extremely small denominator.

    Args:
        residual_vector_small: Residual r_k.
        preconditioned_residual_small: Preconditioned residual z_k.
        base_state: Initial state.

    Theory:
        Small d_k validation is done by curvature(), not step_length().
        step_length() only checks if the RESULT is finite.
        With d_k = 1e-15, the result will be huge but still finite.
    """
    d_k = 1e-15  # Very small but curvature() would have caught this
    eps_breakdown = 1e-14
    # For first iteration: p = z (steepest descent direction)
    p = preconditioned_residual_small

    alpha, state = step_length(
        p,
        residual_vector_small,
        preconditioned_residual_small,
        d_k,
        eps_breakdown,
        base_state,
    )

    # Result is finite (even if huge), so no breakdown
    # Note: In practice, curvature() would restart before this
    assert np.isfinite(alpha)
    assert not state.breakdown


def test_step_length_breakdown_nan_numerator(base_state: IterationState) -> None:
    """Test step_length breakdown with NaN in numerator.

    Args:
        base_state: Initial state.

    Theory:
        NaN in p^T r indicates preconditioner or search direction failure.
        Breakdown flag prevents propagation.
    """
    r = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    z = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    p = np.array([np.nan, 1.0, 1.0], dtype=np.float64)  # NaN in search direction
    d_k = 2.0
    eps_breakdown = 1e-14

    alpha, state = step_length(p, r, z, d_k, eps_breakdown, base_state)

    # Breakdown detected
    assert alpha == 0.0
    assert state.breakdown


def test_step_length_breakdown_inf_result(base_state: IterationState) -> None:
    """Test step_length breakdown with inf in result.

    Args:
        base_state: Initial state.

    Theory:
        Very large numerator / very small denominator = Inf.
        Breakdown detection prevents overflow in x update.
    """
    r = np.array([1e200, 1e200, 1e200], dtype=np.float64)
    z = np.array([1e200, 1e200, 1e200], dtype=np.float64)
    p = z.copy()  # First iteration: p = z
    d_k = 1e-200  # Causes actual overflow to inf
    eps_breakdown = 1e-14

    alpha, state = step_length(p, r, z, d_k, eps_breakdown, base_state)

    # Breakdown detected due to infinite result
    assert alpha == 0.0
    assert state.breakdown


# =============================================================================
# Test beta_update()
# =============================================================================


def test_beta_update_valid_decreasing_residual(
    residual_vector_small: NDArray,
    residual_decreasing_small: NDArray,
    base_state: IterationState,
) -> None:
    """Test beta_update with decreasing residual (normal case).

    Args:
        residual_vector_small: Previous residual r_k with ||r|| = 1.0.
        residual_decreasing_small: New residual r_{k+1} with ||r|| = 0.1.
        base_state: Initial state.

    Theory:
        Beta = ||r_{k+1}||^2 / ||r_k||^2.
        With decreasing residuals: beta < 1 (good).
        No restart should be triggered.
    """
    beta_max = DEFAULT_BETA_MAX

    beta, state = beta_update(
        residual_decreasing_small,
        residual_vector_small,
        beta_max,
        base_state,
    )

    # Beta in valid range
    r_old_norm_sq = np.dot(residual_vector_small, residual_vector_small)
    r_new_norm_sq = np.dot(residual_decreasing_small, residual_decreasing_small)
    expected_beta = r_new_norm_sq / r_old_norm_sq
    assert beta == pytest.approx(expected_beta)
    assert 0 <= beta < 1

    # No restart
    assert not state.restart
    assert state.num_restarts == 0


def test_beta_update_restart_large_beta(
    residual_decreasing_small: NDArray,
    base_state: IterationState,
) -> None:
    """Test beta_update restart on large beta (> beta_max).

    Args:
        residual_decreasing_small: New residual (larger in this test).
        base_state: Initial state.

    Theory:
        If beta > beta_max, loss of conjugacy detected.
        Algorithm restarts with steepest descent.
    """
    # Construct r_old very small, r_new large -> beta > beta_max
    r_old = np.array([1e-6, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    r_new = residual_decreasing_small  # ||r_new|| = 0.1
    beta_max = 1e3  # Small beta_max for testing

    beta, state = beta_update(r_new, r_old, beta_max, base_state)

    # Large beta would exceed beta_max
    r_old_norm_sq = np.dot(r_old, r_old)
    r_new_norm_sq = np.dot(r_new, r_new)
    expected_beta_raw = r_new_norm_sq / r_old_norm_sq
    assert expected_beta_raw > beta_max

    # Restart triggered, beta set to 0
    assert beta == 0.0
    assert state.restart
    assert state.num_restarts == 1


def test_beta_update_zero_old_residual(
    residual_decreasing_small: NDArray,
    base_state: IterationState,
) -> None:
    """Test beta_update with zero old residual (already converged).

    Args:
        residual_decreasing_small: New residual.
        base_state: Initial state.

    Theory:
        If ||r_old|| ~ 0, system already converged.
        Avoid division by zero, return beta = 0.
    """
    r_old = np.zeros(5, dtype=np.float64)
    r_new = residual_decreasing_small
    beta_max = DEFAULT_BETA_MAX

    beta, state = beta_update(r_new, r_old, beta_max, base_state)

    # Beta = 0 (avoid division by zero)
    assert beta == 0.0
    assert not state.breakdown  # Not a breakdown, just converged


# =============================================================================
# Test direction_update()
# =============================================================================


def test_direction_update_normal_case(
    preconditioned_residual_small: NDArray,
    search_direction_small: NDArray,
    base_state: IterationState,
) -> None:
    """Test direction_update with normal conjugate direction.

    Args:
        preconditioned_residual_small: New preconditioned residual z_{k+1}.
        search_direction_small: Old search direction p_k.
        base_state: Initial state.

    Theory:
        Normal case: p_{k+1} = z_{k+1} + β p_k.
        Conjugate direction formula with momentum.
    """
    beta = 0.5

    p_new, state = direction_update(
        preconditioned_residual_small,
        search_direction_small,
        beta,
        base_state,
    )

    # Conjugate direction computed correctly
    expected_p = preconditioned_residual_small + beta * search_direction_small
    np.testing.assert_allclose(p_new, expected_p)

    # State unchanged
    assert not state.restart


def test_direction_update_restart_uses_steepest_descent(
    preconditioned_residual_small: NDArray,
    search_direction_small: NDArray,
) -> None:
    """Test direction_update with restart uses z_{k+1} only.

    Args:
        preconditioned_residual_small: New preconditioned residual z_{k+1}.
        search_direction_small: Old search direction p_k (ignored on restart).

    Theory:
        On restart: p_{k+1} = z_{k+1} (steepest descent).
        Momentum term β p_k ignored.
    """
    state_restart = IterationState(
        converged=False,
        residual_history=[1.0],
        restart=True,  # Restart flag set
        breakdown=False,
        divergence=False,
        num_restarts=1,
        num_residual_replacements=0,
        iterations=1,
    )
    beta = 0.5  # Ignored on restart

    p_new, state = direction_update(
        preconditioned_residual_small,
        search_direction_small,
        beta,
        state_restart,
    )

    # Steepest descent direction (no momentum)
    np.testing.assert_allclose(p_new, preconditioned_residual_small)


def test_direction_update_zero_beta(
    preconditioned_residual_small: NDArray,
    search_direction_small: NDArray,
    base_state: IterationState,
) -> None:
    """Test direction_update with beta = 0 (steepest descent).

    Args:
        preconditioned_residual_small: New preconditioned residual z_{k+1}.
        search_direction_small: Old search direction p_k.
        base_state: Initial state.

    Theory:
        With beta = 0: p_{k+1} = z_{k+1} (no momentum).
        Equivalent to restart but without restart flag.
    """
    beta = 0.0

    p_new, state = direction_update(
        preconditioned_residual_small,
        search_direction_small,
        beta,
        base_state,
    )

    # No momentum term
    expected_p = preconditioned_residual_small + 0.0 * search_direction_small
    np.testing.assert_allclose(p_new, expected_p)


# =============================================================================
# Test residual_management()
# =============================================================================


@pytest.fixture
def simple_spd_matrix() -> NDArray:
    """Simple 5x5 SPD matrix for residual management tests.

    Returns:
        5x5 diagonal SPD matrix.
    """
    return np.diag([2.0, 3.0, 4.0, 5.0, 6.0])


@pytest.fixture
def simple_preconditioner() -> Callable[[NDArray, object], NDArray]:
    """Simple identity preconditioner.

    Returns:
        Preconditioner function.
    """

    def precond(r: NDArray, context: object) -> NDArray:
        return r.copy()

    return precond


@pytest.fixture
def simple_context_builder() -> Callable[[int], None]:
    """Simple context builder (returns None).

    Returns:
        Context builder function.
    """

    def builder(k: int) -> None:
        return None

    return builder


def test_residual_management_no_action(
    simple_spd_matrix: NDArray,
    simple_preconditioner: Callable[[NDArray, object], NDArray],
    rhs_unit_norm: NDArray,
    residual_vector_small: NDArray,
    preconditioned_residual_small: NDArray,
    search_direction_small: NDArray,
    simple_context_builder: Callable[[int], None],
    base_state: IterationState,
) -> None:
    """Test residual_management with no recomputation needed.

    Args:
        simple_spd_matrix: System matrix.
        simple_preconditioner: Identity preconditioner.
        rhs_unit_norm: RHS vector.
        residual_vector_small: Current residual.
        preconditioned_residual_small: Current preconditioned residual.
        search_direction_small: Current search direction.
        simple_context_builder: Context builder.
        base_state: Initial state.

    Theory:
        If k != 0 mod m_replacement and no divergence:
        - No recomputation
        - Vectors unchanged
        - State unchanged
    """
    x_k = np.zeros(5, dtype=np.float64)
    k = 5  # Not divisible by m_replacement=50
    m_replacement = DEFAULT_RESIDUAL_REPLACEMENT_FREQ
    gamma_div = DEFAULT_DIVERGENCE_FACTOR

    r_new, z_new, p_new, state = residual_management(
        simple_spd_matrix,
        simple_preconditioner,
        rhs_unit_norm,
        x_k,
        residual_vector_small,
        preconditioned_residual_small,
        search_direction_small,
        base_state,
        k,
        m_replacement,
        gamma_div,
        simple_context_builder,
    )

    # No recomputation
    np.testing.assert_allclose(r_new, residual_vector_small)
    np.testing.assert_allclose(z_new, preconditioned_residual_small)
    np.testing.assert_allclose(p_new, search_direction_small)

    # State unchanged
    assert not state.divergence
    assert state.num_residual_replacements == 0


def test_residual_management_periodic_recomputation(
    simple_spd_matrix: NDArray,
    simple_preconditioner: Callable[[NDArray, object], NDArray],
    rhs_unit_norm: NDArray,
    residual_vector_small: NDArray,
    preconditioned_residual_small: NDArray,
    search_direction_small: NDArray,
    simple_context_builder: Callable[[int], None],
    base_state: IterationState,
) -> None:
    """Test residual_management periodic recomputation.

    Args:
        simple_spd_matrix: System matrix.
        simple_preconditioner: Identity preconditioner.
        rhs_unit_norm: RHS vector.
        residual_vector_small: Current residual.
        preconditioned_residual_small: Current preconditioned residual.
        search_direction_small: Current search direction.
        simple_context_builder: Context builder.
        base_state: Initial state.

    Theory:
        Every m_replacement iterations, recompute true residual:
        - r_true = b - A x_k
        - z_true = M^{-1} r_true
        - p_true = z_true
    """
    x_k = np.zeros(5, dtype=np.float64)
    k = 50  # Divisible by m_replacement=50
    m_replacement = 50
    gamma_div = DEFAULT_DIVERGENCE_FACTOR

    r_new, z_new, p_new, state = residual_management(
        simple_spd_matrix,
        simple_preconditioner,
        rhs_unit_norm,
        x_k,
        residual_vector_small,
        preconditioned_residual_small,
        search_direction_small,
        base_state,
        k,
        m_replacement,
        gamma_div,
        simple_context_builder,
    )

    # True residual recomputed
    expected_r_true = rhs_unit_norm - simple_spd_matrix @ x_k
    np.testing.assert_allclose(r_new, expected_r_true)

    # Replacement counter incremented
    assert state.num_residual_replacements == 1
    assert not state.divergence


def test_residual_management_divergence_detection(
    simple_spd_matrix: NDArray,
    simple_preconditioner: Callable[[NDArray, object], NDArray],
    rhs_unit_norm: NDArray,
    preconditioned_residual_small: NDArray,
    search_direction_small: NDArray,
    simple_context_builder: Callable[[int], None],
    base_state: IterationState,
) -> None:
    """Test residual_management divergence detection and recovery.

    Args:
        simple_spd_matrix: System matrix.
        simple_preconditioner: Identity preconditioner.
        rhs_unit_norm: RHS vector with ||b|| = 1.0.
        preconditioned_residual_small: Current preconditioned residual.
        search_direction_small: Current search direction.
        simple_context_builder: Context builder.
        base_state: Initial state.

    Theory:
        If ||r_k|| > gamma_div * ||b||, divergence detected:
        - Recompute true residual immediately
        - Set divergence flag
        - Increment replacement counter
    """
    # Create large residual to trigger divergence
    b_norm = norm(rhs_unit_norm)
    gamma_div = DEFAULT_DIVERGENCE_FACTOR
    large_residual = np.ones(5, dtype=np.float64) * gamma_div * b_norm * 2.0

    x_k = np.zeros(5, dtype=np.float64)
    k = 5

    r_new, z_new, p_new, state = residual_management(
        simple_spd_matrix,
        simple_preconditioner,
        rhs_unit_norm,
        x_k,
        large_residual,  # Divergent residual
        preconditioned_residual_small,
        search_direction_small,
        base_state,
        k,
        DEFAULT_RESIDUAL_REPLACEMENT_FREQ,
        gamma_div,
        simple_context_builder,
    )

    # Divergence detected
    assert state.divergence
    assert state.num_residual_replacements == 1

    # True residual recomputed
    expected_r_true = rhs_unit_norm - simple_spd_matrix @ x_k
    np.testing.assert_allclose(r_new, expected_r_true)


def test_residual_management_disabled_recomputation(
    simple_spd_matrix: NDArray,
    simple_preconditioner: Callable[[NDArray, object], NDArray],
    rhs_unit_norm: NDArray,
    residual_vector_small: NDArray,
    preconditioned_residual_small: NDArray,
    search_direction_small: NDArray,
    simple_context_builder: Callable[[int], None],
    base_state: IterationState,
) -> None:
    """Test residual_management with recomputation disabled (m_replacement=0).

    Args:
        simple_spd_matrix: System matrix.
        simple_preconditioner: Identity preconditioner.
        rhs_unit_norm: RHS vector.
        residual_vector_small: Current residual.
        preconditioned_residual_small: Current preconditioned residual.
        search_direction_small: Current search direction.
        simple_context_builder: Context builder.
        base_state: Initial state.

    Theory:
        If m_replacement = 0, periodic recomputation disabled.
        Only divergence detection active.
    """
    x_k = np.zeros(5, dtype=np.float64)
    k = 50
    m_replacement = 0  # Disabled
    gamma_div = DEFAULT_DIVERGENCE_FACTOR

    r_new, z_new, p_new, state = residual_management(
        simple_spd_matrix,
        simple_preconditioner,
        rhs_unit_norm,
        x_k,
        residual_vector_small,
        preconditioned_residual_small,
        search_direction_small,
        base_state,
        k,
        m_replacement,
        gamma_div,
        simple_context_builder,
    )

    # No recomputation even at k=50
    np.testing.assert_allclose(r_new, residual_vector_small)
    assert state.num_residual_replacements == 0
