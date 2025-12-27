"""Integration tests for symmetric solver testing.

This module ensures all three solvers (FCG, PCG, SciPy CG) receive equal test
coverage on various matrix types and preconditioners. All tests use integration
tolerances (rtol=1e-6, atol=1e-14) to verify functionality without requiring
extreme precision.

Test Coverage:
- 3 solvers × 5 matrix types = 15 tests (parametrized)
- 3 solvers × 3 preconditioners × 3 matrices = 27 tests (parametrized)
- Edge cases: non-zero x0, zero RHS, max iterations
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


# =============================================================================
# Test: All Solvers × All Matrices (No Preconditioner)
# =============================================================================


@pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
@pytest.mark.parametrize(
    "matrix_fixture",
    [
        "identity_matrix_small",
        "tridiagonal_spd_small",
        "diagonal_spd_small",
        "random_spd_small",
        "ill_conditioned_spd_small",
    ],
)
def test_solver_converges_on_spd_matrices(
    solver_name: str,
    matrix_fixture: str,
    solver_factories: dict[str, Callable],
    rhs_ones_small: NDArray,
    zero_initial_guess_small: NDArray,
    integration_tolerances: tuple[float, float],
    request: pytest.FixtureRequest,
) -> None:
    """Test all solvers converge on various SPD matrices without preconditioning.

    Theory:
        All CG variants should converge on SPD matrices, though iteration
        counts may vary. Without preconditioning, convergence depends on
        the condition number of the matrix.

    Coverage:
        3 solvers × 5 matrix types = 15 test runs
    """
    solver = solver_factories[solver_name]
    matrix = request.getfixturevalue(matrix_fixture)
    rtol, atol = integration_tolerances

    x, result = solver(
        matrix,
        rhs_ones_small,
        x0=zero_initial_guess_small,
        rtol=rtol,
        atol=atol,
        max_iterations=200,
    )

    # Verify convergence
    assert result.converged, f"{solver_name} failed to converge on {matrix_fixture}"

    # Verify residual tolerance
    assert result.residual_abs < max(rtol * result.rhs_norm, atol)

    # Verify solution is reasonable (not NaN/Inf)
    assert np.all(np.isfinite(x))


# =============================================================================
# Test: All Solvers × All Preconditioners × Selected Matrices
# =============================================================================


@pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
@pytest.mark.parametrize(
    "precond_fixture",
    [
        "identity_preconditioner",
        "jacobi_preconditioner_tridiagonal",
        "ilu_preconditioner_tridiagonal",
    ],
)
@pytest.mark.parametrize(
    "system_fixture",
    [
        "identity_system_known_solution",
        "tridiagonal_system_known_solution",
        "diagonal_system_known_solution",
    ],
)
def test_solver_with_preconditioners(
    solver_name: str,
    precond_fixture: str,
    system_fixture: str,
    solver_factories: dict[str, Callable],
    integration_tolerances: tuple[float, float],
    request: pytest.FixtureRequest,
) -> None:
    """Test all solvers with all preconditioners on standard systems.

    Theory:
        Preconditioners should improve convergence (reduce iterations) while
        maintaining solution accuracy. All solver/preconditioner combinations
        should converge successfully.

    Coverage:
        3 solvers × 3 preconditioners × 3 systems = 27 test runs

    Note:
        identity_preconditioner needs special handling for call signature.
    """
    solver = solver_factories[solver_name]
    A, b, x_exact = request.getfixturevalue(system_fixture)
    rtol, atol = integration_tolerances

    # Get preconditioner (handle identity's different signature)
    precond = request.getfixturevalue(precond_fixture)
    if precond_fixture == "identity_preconditioner":
        # Identity preconditioner takes (r, context) → wrap to take just r
        precond_wrapped = lambda r: precond(r, None)
    else:
        precond_wrapped = precond

    x, result = solver(
        A,
        b,
        preconditioner=precond_wrapped,
        rtol=rtol,
        atol=atol,
        max_iterations=200,
    )

    # Verify convergence
    assert result.converged, (
        f"{solver_name} with {precond_fixture} failed on {system_fixture}"
    )

    # Verify residual tolerance
    assert result.residual_abs < max(rtol * result.rhs_norm, atol)

    # Verify solution accuracy (integration tolerance, not convergence)
    np.testing.assert_allclose(x, x_exact, rtol=1e-5, atol=1e-12)


# =============================================================================
# Test: Non-Zero Initial Guess
# =============================================================================


@pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
def test_solver_with_nonzero_initial_guess(
    solver_name: str,
    solver_factories: dict[str, Callable],
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    integration_tolerances: tuple[float, float],
    test_seed: int,
) -> None:
    """Test all solvers handle non-zero initial guess correctly.

    Theory:
        CG should work with any initial guess x0. Non-zero x0 affects
        the initial residual r0 = b - A @ x0, but convergence is still
        guaranteed for SPD systems.

    Coverage:
        3 solvers × 1 test each = 3 test runs
    """
    solver = solver_factories[solver_name]
    A, b, x_exact = tridiagonal_system_known_solution
    rtol, atol = integration_tolerances

    # Create random initial guess (different from zero)
    rng = np.random.default_rng(test_seed)
    x0 = rng.standard_normal(len(b))

    x, result = solver(A, b, x0=x0, rtol=rtol, atol=atol, max_iterations=200)

    # Verify convergence
    assert result.converged

    # Verify solution accuracy
    np.testing.assert_allclose(x, x_exact, rtol=1e-5, atol=1e-12)


# =============================================================================
# Test: Zero RHS
# =============================================================================


@pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
def test_solver_with_zero_rhs(
    solver_name: str,
    solver_factories: dict[str, Callable],
    tridiagonal_spd_small: NDArray,
    rhs_zero_small: NDArray,
    zero_initial_guess_small: NDArray,
    integration_tolerances: tuple[float, float],
) -> None:
    """Test all solvers handle zero RHS correctly.

    Theory:
        For b = 0, the solution is x = 0 regardless of matrix A.
        CG should converge immediately (iteration 0 or 1).

    Coverage:
        3 solvers × 1 test each = 3 test runs
    """
    solver = solver_factories[solver_name]
    rtol, atol = integration_tolerances

    x, result = solver(
        tridiagonal_spd_small,
        rhs_zero_small,
        x0=zero_initial_guess_small,
        rtol=rtol,
        atol=atol,
        max_iterations=10,
    )

    # Should converge immediately
    assert result.converged
    assert result.iterations <= 1

    # Solution should be zero
    np.testing.assert_allclose(x, np.zeros_like(x), atol=1e-14)


# =============================================================================
# Test: Max Iterations Behavior
# =============================================================================


@pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
def test_solver_respects_max_iterations(
    solver_name: str,
    solver_factories: dict[str, Callable],
    ill_conditioned_spd_small: NDArray,
    rhs_ones_small: NDArray,
    zero_initial_guess_small: NDArray,
) -> None:
    """Test all solvers respect max_iterations limit.

    Theory:
        For ill-conditioned systems with tight tolerances, CG may not
        converge within a small iteration limit. The solver should
        stop at max_iterations and report non-convergence.

    Coverage:
        3 solvers × 1 test each = 3 test runs
    """
    solver = solver_factories[solver_name]

    # Use very tight tolerance with small iteration limit
    x, result = solver(
        ill_conditioned_spd_small,
        rhs_ones_small,
        x0=zero_initial_guess_small,
        rtol=1e-14,  # Very tight tolerance
        atol=1e-14,
        max_iterations=5,  # Too few iterations to converge
    )

    # Should hit max iterations
    assert result.iterations == 5

    # May or may not have converged (depends on luck), but should
    # respect the iteration limit
    assert result.iterations <= 5


# =============================================================================
# Test: Convergence Criteria Validation
# =============================================================================


@pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
def test_solver_satisfies_rtol_criterion(
    solver_name: str,
    solver_factories: dict[str, Callable],
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    integration_tolerances: tuple[float, float],
) -> None:
    """Test all solvers satisfy relative tolerance criterion.

    Theory:
        CG should terminate when ||r_k|| / ||r_0|| < rtol.
        This is the standard relative residual convergence criterion.

    Coverage:
        3 solvers × 1 test each = 3 test runs
    """
    solver = solver_factories[solver_name]
    A, b, _ = tridiagonal_system_known_solution
    rtol, atol = integration_tolerances

    x, result = solver(A, b, rtol=rtol, atol=atol, max_iterations=200)

    assert result.converged

    # Verify relative residual criterion
    # ||r_k|| / ||b|| ≤ rtol (using ||b|| as proxy for ||r_0|| since x0=0)
    relative_residual = result.residual_abs / result.rhs_norm
    assert relative_residual <= rtol * 1.01  # Small margin for numerical errors


@pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
def test_solver_satisfies_atol_criterion_for_small_rhs(
    solver_name: str,
    solver_factories: dict[str, Callable],
    tridiagonal_spd_small: NDArray,
    integration_tolerances: tuple[float, float],
) -> None:
    """Test all solvers satisfy absolute tolerance criterion for small RHS.

    Theory:
        For very small ||b||, the relative criterion rtol may be too strict.
        CG should also terminate when ||r_k|| < atol (absolute criterion).

    Coverage:
        3 solvers × 1 test each = 3 test runs
    """
    solver = solver_factories[solver_name]
    rtol, atol = integration_tolerances

    # Create very small RHS
    b_small = np.ones(10) * 1e-10
    x0 = np.zeros(10)

    x, result = solver(
        tridiagonal_spd_small, b_small, x0=x0, rtol=rtol, atol=atol, max_iterations=200
    )

    assert result.converged

    # Verify absolute residual criterion
    assert result.residual_abs <= atol * 1.01


# =============================================================================
# Test: Solution Accuracy vs Direct Solver (Integration Tolerances)
# =============================================================================


@pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
@pytest.mark.parametrize(
    "system_fixture",
    [
        "tridiagonal_system_known_solution",
        "diagonal_system_known_solution",
    ],
)
def test_solver_matches_direct_solve(
    solver_name: str,
    system_fixture: str,
    solver_factories: dict[str, Callable],
    integration_tolerances: tuple[float, float],
    request: pytest.FixtureRequest,
) -> None:
    """Test all solvers match direct solver within integration tolerances.

    Theory:
        CG computes the same solution as np.linalg.solve (up to tolerance).
        This verifies solver correctness by comparing against trusted reference.

    Coverage:
        3 solvers × 2 systems = 6 test runs
    """
    solver = solver_factories[solver_name]
    A, b, x_exact = request.getfixturevalue(system_fixture)
    rtol, atol = integration_tolerances

    x, result = solver(A, b, rtol=rtol, atol=atol, max_iterations=200)

    assert result.converged

    # Compare to direct solver (integration tolerance expectations)
    np.testing.assert_allclose(x, x_exact, rtol=1e-5, atol=1e-12)


# =============================================================================
# Test: Ill-Conditioned Systems with Extended Iterations
# =============================================================================


@pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
@pytest.mark.parametrize(
    "precond_fixture",
    ["jacobi_preconditioner_factory", "ilu_preconditioner_factory"],
)
def test_solver_handles_ill_conditioned_with_preconditioner(
    solver_name: str,
    precond_fixture: str,
    solver_factories: dict[str, Callable],
    ill_conditioned_spd_small: NDArray,
    rhs_ones_small: NDArray,
    integration_tolerances: tuple[float, float],
    request: pytest.FixtureRequest,
) -> None:
    """Test all solvers can handle ill-conditioned systems with preconditioning.

    Theory:
        Ill-conditioned systems (κ ≈ 1000) require more iterations and
        benefit significantly from preconditioning. All solvers should
        eventually converge with sufficient iterations.

    Coverage:
        3 solvers × 2 preconditioners = 6 test runs
    """
    solver = solver_factories[solver_name]
    rtol, atol = integration_tolerances

    # Create preconditioner for ill-conditioned matrix
    precond_factory = request.getfixturevalue(precond_fixture)
    precond = precond_factory(ill_conditioned_spd_small)

    x, result = solver(
        ill_conditioned_spd_small,
        rhs_ones_small,
        preconditioner=precond,
        rtol=rtol,
        atol=atol,
        max_iterations=500,  # Extended iterations for ill-conditioned
    )

    # Should converge with enough iterations + preconditioner
    assert result.converged, f"{solver_name} failed on ill-conditioned system"

    # Verify solution is reasonable
    assert np.all(np.isfinite(x))
    assert result.residual_abs < max(rtol * result.rhs_norm, atol)
