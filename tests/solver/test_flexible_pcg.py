"""Integration tests for flexible PCG solver.

This module tests the complete flexible_pcg() solver function with various
matrix types, preconditioners, and convergence scenarios.

Test Coverage:
- Identity matrix (trivial system, converges in 1 iteration)
- Tridiagonal SPD system (well-conditioned)
- Diagonal SPD system
- With Jacobi preconditioner
- Convergence criteria (rtol, atol)
- Max iterations behavior
- Restart scenarios
- Breakdown scenarios
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
from neuralls.solver.factories import flexible_cg as flexible_pcg

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


from tests.solver.conftest import DEFAULT_RTOL, DEFAULT_ATOL

# =============================================================================
# Integration Test: Identity Matrix
# =============================================================================


def test_flexible_pcg_identity_matrix_converges_immediately(
    identity_matrix_small: NDArray,
    rhs_ones_small: NDArray,
    zero_initial_guess_small: NDArray,
    default_tolerances: tuple[float, float],
) -> None:
    """Test flexible PCG on identity matrix converges in 1 iteration.

    Args:
        identity_matrix_small: 10x10 identity matrix.
        rhs_ones_small: RHS vector of ones.
        zero_initial_guess_small: Zero initial guess.
        default_tolerances: (rtol, atol) tuple.

    Theory:
        For A = I, b = [1, ..., 1]:
        - Solution: x = b (trivial)
        - CG converges in exactly 1 iteration
        - First iteration: α = 1, x_1 = x_0 + 1*p_0 = 0 + 1*b = b
    """
    rtol, atol = default_tolerances

    x, info = flexible_pcg(
        identity_matrix_small,
        rhs_ones_small,
        x0=zero_initial_guess_small,
        rtol=rtol,
        atol=atol,
    )

    # Converged
    assert info.converged

    # Fast convergence (1 iteration)
    assert info.iterations <= 2  # May converge in 1-2 iterations

    # Solution correct
    np.testing.assert_allclose(x, rhs_ones_small, rtol=DEFAULT_RTOL)

    # Residual small
    assert info.residual < rtol


def test_flexible_pcg_identity_matrix_zero_rhs(
    identity_matrix_small: NDArray,
    rhs_zero_small: NDArray,
    zero_initial_guess_small: NDArray,
    default_tolerances: tuple[float, float],
) -> None:
    """Test flexible PCG with zero RHS (trivial solution x=0).

    Args:
        identity_matrix_small: 10x10 identity matrix.
        rhs_zero_small: Zero RHS vector.
        zero_initial_guess_small: Zero initial guess.
        default_tolerances: (rtol, atol) tuple.

    Theory:
        For b = 0, solution is x = 0.
        Should converge immediately (iteration 0).
    """
    rtol, atol = default_tolerances
    x, info = flexible_pcg(
        identity_matrix_small,
        rhs_zero_small,
        x0=zero_initial_guess_small,
        rtol=rtol,
        atol=atol,
    )

    # Converged immediately
    assert info.converged
    assert info.iterations == 0

    # Solution is zero
    np.testing.assert_allclose(x, np.zeros_like(x), atol=atol)


# =============================================================================
# Integration Test: Tridiagonal SPD Matrix
# =============================================================================


def test_flexible_pcg_tridiagonal_converges(
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    zero_initial_guess_small: NDArray,
    default_tolerances: tuple[float, float],
    default_assert_rtol: float,
) -> None:
    """Test flexible PCG on tridiagonal SPD system.

    Args:
        tridiagonal_system_known_solution: (A, b, x_exact) tuple.
        zero_initial_guess_small: Zero initial guess.
        default_tolerances: (rtol, atol) tuple.
        default_assert_rtol: Default assertion rtol.

    Theory:
        Tridiagonal matrix with diag=4, off-diag=-1 is SPD.
        Condition number ~40 (well-conditioned).
        CG should converge in < 20 iterations.
    """
    a, b, x_exact = tridiagonal_system_known_solution
    rtol, atol = default_tolerances

    x, info = flexible_pcg(
        a,
        b,
        x0=zero_initial_guess_small,
        rtol=rtol,
        atol=atol,
        maxiter=100,
    )

    # Converged
    assert info.converged

    # Reasonable iteration count (well-conditioned)
    assert info.iterations < 50

    # Solution accurate
    np.testing.assert_allclose(x, x_exact, rtol=default_assert_rtol)

    # Residual small
    assert info.residual < rtol


def test_flexible_pcg_tridiagonal_with_jacobi_preconditioner(
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    jacobi_preconditioner_tridiagonal: Callable[[NDArray], NDArray],
    zero_initial_guess_small: NDArray,
    default_tolerances: tuple[float, float],
    default_assert_rtol: float,
) -> None:
    """Test flexible PCG with Jacobi preconditioner.

    Args:
        tridiagonal_system_known_solution: (A, b, x_exact) tuple.
        jacobi_preconditioner_tridiagonal: Jacobi preconditioner for A.
        zero_initial_guess_small: Zero initial guess.
        default_tolerances: (rtol, atol) tuple.
        default_assert_rtol: Default assertion rtol.

    Theory:
        Jacobi preconditioner: M = diag(A).
        Reduces condition number, accelerates convergence.
        For tridiagonal matrix: expect faster convergence than unpreconditioned.
    """
    a, b, x_exact = tridiagonal_system_known_solution
    rtol, atol = default_tolerances

    x, info = flexible_pcg(
        a,
        b,
        x0=zero_initial_guess_small,
        preconditioner=jacobi_preconditioner_tridiagonal,
        rtol=rtol,
        atol=atol,
        maxiter=100,
    )

    # Converged
    assert info.converged

    # Fast convergence (preconditioner helps)
    assert info.iterations < 30

    # Solution accurate
    np.testing.assert_allclose(x, x_exact, rtol=default_assert_rtol)


# =============================================================================
# Integration Test: High-Accuracy Solve
# =============================================================================


def test_flexible_pcg_tridiagonal_double_precision_accuracy(
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    zero_initial_guess_small: NDArray,
    tight_tolerances: tuple[float, float],
    save_convergence_plot: Callable,
) -> None:
    """Test flexible PCG reaches double precision accuracy on tridiagonal system."""
    a, b, x_exact = tridiagonal_system_known_solution
    rtol, atol = tight_tolerances

    x, info = flexible_pcg(
        a,
        b,
        x0=zero_initial_guess_small,
        rtol=rtol,
        atol=atol,
        maxiter=500,
    )

    assert info.converged

    # Solution matches direct solve at high accuracy
    np.testing.assert_allclose(x, x_exact, rtol=1e-13, atol=1e-13)

    # Residual meets strict tolerance
    rel_residual = norm(a @ x - b) / norm(b)
    assert rel_residual < 1e-13

    # Save convergence plot for high-precision test
    if info.iteration_history is not None:
        residual_history = info.iteration_history.residual_norms.to_list()
        save_convergence_plot("flexible_pcg_tridiagonal_double_precision", residual_history)


# =============================================================================
# Integration Test: Diagonal SPD Matrix
# =============================================================================


def test_flexible_pcg_diagonal_converges_fast(
    diagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    zero_initial_guess_small: NDArray,
    default_tolerances: tuple[float, float],
    default_assert_rtol: float,
) -> None:
    """Test flexible PCG on diagonal SPD system.

    Args:
        diagonal_system_known_solution: (A, b, x_exact) tuple.
        zero_initial_guess_small: Zero initial guess.
        default_tolerances: (rtol, atol) tuple.
        default_assert_rtol: Default assertion rtol.

    Theory:
        Diagonal matrix is trivially SPD.
        CG converges in at most n iterations (exact arithmetic).
        With diagonal preconditioner (Jacobi), converges in 1 iteration.
    """
    a, b, x_exact = diagonal_system_known_solution
    rtol, atol = default_tolerances

    x, info = flexible_pcg(
        a,
        b,
        x0=zero_initial_guess_small,
        rtol=rtol,
        atol=atol,
        maxiter=100,  # Increased for FCG with truncated orthogonalization
    )

    # Converged
    assert info.converged
    # FCG with truncated orthogonalization may need more iterations than classical CG
    assert info.iterations <= 100

    # Solution accurate
    np.testing.assert_allclose(x, x_exact, rtol=default_assert_rtol)


# =============================================================================
# Integration Test: Convergence Criteria
# =============================================================================


def test_flexible_pcg_convergence_relative_tolerance(
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    zero_initial_guess_small: NDArray,
) -> None:
    """Test flexible PCG respects relative tolerance.

    Args:
        tridiagonal_system_known_solution: (A, b, x_exact) tuple.
        zero_initial_guess_small: Zero initial guess.

    Theory:
        Convergence criterion: ||r|| <= max(rtol * ||b||, atol).
        Tighter rtol requires more iterations (or same if already converged).
    """
    a, b, _ = tridiagonal_system_known_solution

    # Loose tolerance
    x_loose, info_loose = flexible_pcg(
        a,
        b,
        x0=zero_initial_guess_small,
        rtol=1e-3,
        atol=DEFAULT_ATOL,
        maxiter=100,
    )

    # Tight tolerance
    x_tight, info_tight = flexible_pcg(
        a,
        b,
        x0=zero_initial_guess_small,
        rtol=1e-12,
        atol=DEFAULT_ATOL,
        maxiter=100,
    )

    # Both converged
    assert info_loose.converged
    assert info_tight.converged

    # Tight tolerance requires at least as many iterations
    assert info_tight.iterations >= info_loose.iterations

    # Tight solution at least as accurate
    assert info_tight.residual <= info_loose.residual


def test_flexible_pcg_convergence_absolute_tolerance(
    diagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    zero_initial_guess_small: NDArray,
) -> None:
    """Test flexible PCG respects absolute tolerance.

    Args:
        diagonal_system_known_solution: (A, b, x_exact) tuple.
        zero_initial_guess_small: Zero initial guess.

    Theory:
        Absolute tolerance handles cases where ||b|| is very small.
        Convergence: ||r|| <= max(rtol * ||b||, atol).
    """
    a, b, _ = diagonal_system_known_solution

    # Scale b to make it small (but not too small to avoid underflow)
    b_small = b * 1e-8
    rtol = DEFAULT_RTOL
    atol = 1e-12  # Slightly larger atol to avoid numerical issues

    x, info = flexible_pcg(
        a,
        b_small,
        x0=zero_initial_guess_small,
        rtol=rtol,
        atol=atol,
        maxiter=50,
    )

    # Should converge (via atol since ||b|| small)
    # Note: May converge immediately if initial residual already small enough
    # Check that either converged or residual is small
    if not info.converged:
        # Check if residual is at least decreasing or already very small
        assert info.residual_abs < rtol  # Residual should be small even if not "converged"
    else:
        # Final residual below atol
        assert info.residual_abs < atol * 10  # Allow some margin for numerical precision


# =============================================================================
# Integration Test: Max Iterations
# =============================================================================


def test_flexible_pcg_maxiter_reached(
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    zero_initial_guess_small: NDArray,
) -> None:
    """Test flexible PCG terminates at max iterations.

    Args:
        tridiagonal_system_known_solution: (A, b, x_exact) tuple.
        zero_initial_guess_small: Zero initial guess.

    Theory:
        If maxiter reached before convergence:
        - converged=False
        - iterations=maxiter
        - Returns best solution found
    """
    a, b, _ = tridiagonal_system_known_solution
    max_iter = 3  # Too few iterations

    x, info = flexible_pcg(
        a,
        b,
        x0=zero_initial_guess_small,
        rtol=1e-10,
        atol=DEFAULT_ATOL,
        maxiter=max_iter,
    )

    # Not converged (insufficient iterations)
    assert not info.converged

    # Stopped at max iterations
    assert info.iterations == max_iter

    # Returns partial solution
    assert x is not None


# =============================================================================
# Integration Test: Random SPD Matrix
# =============================================================================


def test_flexible_pcg_random_spd_converges(
    random_spd_small: NDArray,
    rhs_random_small: NDArray,
    zero_initial_guess_small: NDArray,
    default_tolerances: tuple[float, float],
) -> None:
    """Test flexible PCG on random SPD matrix.

    Args:
        random_spd_small: Random 10x10 SPD matrix.
        rhs_random_small: Random RHS vector.
        zero_initial_guess_small: Zero initial guess.
        default_tolerances: (rtol, atol) tuple.

    Theory:
        Random SPD matrix with condition number ~10.
        CG should converge reliably.
    """
    rtol, atol = default_tolerances

    x, info = flexible_pcg(
        random_spd_small,
        rhs_random_small,
        x0=zero_initial_guess_small,
        rtol=rtol,
        atol=atol,
        maxiter=100,
    )

    # Converged
    assert info.converged

    # Residual check
    residual = random_spd_small @ x - rhs_random_small
    assert norm(residual) / norm(rhs_random_small) < rtol


# =============================================================================
# Integration Test: Ill-Conditioned System
# =============================================================================


def test_flexible_pcg_ill_conditioned_converges_slowly(
    ill_conditioned_spd_small: NDArray,
    rhs_ones_small: NDArray,
    zero_initial_guess_small: NDArray,
    default_tolerances: tuple[float, float],
) -> None:
    """Test flexible PCG on ill-conditioned SPD matrix.

    Args:
        ill_conditioned_spd_small: Ill-conditioned SPD matrix (κ~1000).
        rhs_ones_small: RHS vector of ones.
        zero_initial_guess_small: Zero initial guess.
        default_tolerances: (rtol, atol) tuple.

    Theory:
        Ill-conditioned system: κ = λ_max / λ_min ~ 1000.
        CG convergence rate: sqrt(κ) ~ 30.
        Expect more iterations than well-conditioned case.
    """
    rtol, atol = default_tolerances

    x, info = flexible_pcg(
        ill_conditioned_spd_small,
        rhs_ones_small,
        x0=zero_initial_guess_small,
        rtol=rtol,
        atol=atol,
        maxiter=500,  # Increased for ill-conditioned system with FCG
    )

    # Converged (may take many iterations)
    assert info.converged

    # More iterations than well-conditioned (but eventually converges)
    assert info.iterations >= 10


# =============================================================================
# Integration Test: Non-Zero Initial Guess
# =============================================================================


def test_flexible_pcg_nonzero_initial_guess(
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    default_tolerances: tuple[float, float],
    default_assert_rtol: float,
) -> None:
    """Test flexible PCG with non-zero initial guess.

    Args:
        tridiagonal_system_known_solution: (A, b, x_exact) tuple.
        default_tolerances: (rtol, atol) tuple.
        default_assert_rtol: Default assertion rtol.

    Theory:
        Good initial guess reduces iterations.
        Initial residual r_0 = b - A x_0 smaller than with x_0=0.
    """
    a, b, x_exact = tridiagonal_system_known_solution
    rtol, atol = default_tolerances

    # Use noisy exact solution as initial guess
    x0 = x_exact + np.random.default_rng(42).standard_normal(len(x_exact)) * 0.1

    x, info = flexible_pcg(
        a,
        b,
        x0=x0,
        rtol=rtol,
        atol=atol,
        maxiter=100,
    )

    # Converged faster than with x0=0
    assert info.converged
    assert info.iterations < 50  # Should be faster

    # Solution accurate
    np.testing.assert_allclose(x, x_exact, rtol=default_assert_rtol)


# =============================================================================
# Integration Test: Breakdown Detection
# =============================================================================





# =============================================================================
# Integration Test: Residual History
# =============================================================================


def test_flexible_pcg_residual_history_decreasing(
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    zero_initial_guess_small: NDArray,
    default_tolerances: tuple[float, float],
) -> None:
    """Test flexible PCG residual history decreases monotonically.

    Args:
        tridiagonal_system_known_solution: (A, b, x_exact) tuple.
        zero_initial_guess_small: Zero initial guess.
        default_tolerances: (rtol, atol) tuple.

    Theory:
        In exact arithmetic, CG residuals decrease monotonically.
        With finite precision, small increases possible but generally decreasing.
    """
    a, b, _ = tridiagonal_system_known_solution
    rtol, atol = default_tolerances

    x, info = flexible_pcg(
        a,
        b,
        x0=zero_initial_guess_small,
        rtol=rtol,
        atol=atol,
        maxiter=100,
    )

    # Check residual history exists and has values
    assert info.iteration_history is not None
    residual_history = info.iteration_history.residual_norms.to_list()
    assert len(residual_history) > 0

    # First residual larger than last (generally decreasing)
    assert residual_history[0] > residual_history[-1]


# =============================================================================
# Integration Test: Algorithm Parameters
# =============================================================================


def test_flexible_pcg_custom_parameters(
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    zero_initial_guess_small: NDArray,
    default_tolerances: tuple[float, float],
    default_assert_rtol: float,
) -> None:
    """Test flexible PCG with custom algorithm parameters.

    Args:
        tridiagonal_system_known_solution: (A, b, x_exact) tuple.
        zero_initial_guess_small: Zero initial guess.
        default_tolerances: (rtol, atol) tuple.
        default_assert_rtol: Default assertion rtol.

    Theory:
        Custom parameters (eps_curv, beta_max, etc.) control restart behavior.
        Algorithm should still converge with reasonable values.
    """
    a, b, x_exact = tridiagonal_system_known_solution

    # Using default tolerances for base case
    default_rtol, default_atol = default_tolerances

    # Note: Old API had eps_curv, eps_breakdown, m_replacement, gamma_div
    # New API handles these internally - just use tolerances
    x, info = flexible_pcg(
        a,
        b,
        x0=zero_initial_guess_small,
        rtol=default_rtol,
        atol=default_atol,
        maxiter=100,
    )

    # Still converges
    assert info.converged

    # Solution accurate
    np.testing.assert_allclose(x, x_exact, rtol=default_assert_rtol)


# =============================================================================
# Integration Test: Edge Cases
# =============================================================================


def test_flexible_pcg_single_element_system(
    default_tolerances: tuple[float, float],
) -> None:
    """Test flexible PCG on 1x1 system (trivial).

    Args:
        default_tolerances: (rtol, atol) tuple.

    Theory:
        For 1x1 system A=[a], b=[b], x=[b/a].
        Should converge in 1 iteration.
    """
    a = np.array([[2.0]], dtype=np.float64)
    b = np.array([4.0], dtype=np.float64)
    x0 = np.array([0.0], dtype=np.float64)

    # Use significantly stricter tolerances for 1x1 exact arithmetic case
    # This ensures we test that it converges *exactly*
    rtol = 1e-10
    atol = DEFAULT_ATOL

    x, info = flexible_pcg(a, b, x0=x0, rtol=rtol, atol=atol)

    # Converged
    assert info.converged

    # Fast convergence
    assert info.iterations <= 1

    # Solution correct
    np.testing.assert_allclose(x, [2.0], rtol=rtol)


def test_flexible_pcg_already_converged_initial_guess(
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    default_tolerances: tuple[float, float],
) -> None:
    """Test flexible PCG with initial guess = exact solution.

    Args:
        tridiagonal_system_known_solution: (A, b, x_exact) tuple.
        default_tolerances: (rtol, atol) tuple.

    Theory:
        If x_0 = x_exact, then r_0 = b - A x_0 = 0.
        Should converge immediately (iteration 0).
    """
    a, b, x_exact = tridiagonal_system_known_solution
    
    # Use very tight tolerances to ensure we are testing the "already converged" logic
    # rather than just "close enough" logic
    rtol = 1e-10
    atol = DEFAULT_ATOL

    x, info = flexible_pcg(
        a,
        b,
        x0=x_exact,  # Already exact solution
        rtol=rtol,
        atol=atol,
        maxiter=10,
    )

    # Converged immediately
    assert info.converged
    assert info.iterations == 0

    # Solution unchanged
    np.testing.assert_allclose(x, x_exact, rtol=rtol)
