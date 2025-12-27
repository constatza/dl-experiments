"""High-precision convergence tests for solver validation.

This module tests all three solvers (FCG, PCG, SciPy CG) with strict convergence
tolerances (rtol=1e-12, atol=1e-14) to verify double-precision accuracy.

Uses small 10×10 matrices for convergence tests to achieve true rtol=1e-12 accuracy.
All tests use convergence_tolerances fixture - no custom tolerance hacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


# =============================================================================
# Test: Double-Precision Accuracy
# =============================================================================


@pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
@pytest.mark.parametrize(
    "precond_fixture",
    ["jacobi_preconditioner_tridiagonal", "ilu_preconditioner_tridiagonal"],
)
@pytest.mark.parametrize(
    "system_fixture",
    [
        "tridiagonal_system_known_solution",
        "diagonal_system_known_solution",
    ],
)
def test_solver_achieves_double_precision(
    solver_name: str,
    precond_fixture: str,
    system_fixture: str,
    solver_factories: dict[str, Callable],
    convergence_tolerances: tuple[float, float],
    request: pytest.FixtureRequest,
) -> None:
    """Verify solvers achieve double-precision accuracy with strict tolerances.

    Theory:
        With rtol=1e-12, CG should achieve ~12 digits of relative accuracy.
        Using 10×10 matrices where this is achievable.

    Coverage:
        3 solvers × 2 preconditioners × 2 systems = 12 test runs
    """
    solver = solver_factories[solver_name]
    A, b, x_exact = request.getfixturevalue(system_fixture)
    precond = request.getfixturevalue(precond_fixture)
    rtol, atol = convergence_tolerances

    x, result = solver(
        A,
        b,
        preconditioner=precond,
        rtol=rtol,
        atol=atol,
        max_iterations=200,
    )

    assert result.converged, (
        f"{solver_name} with {precond_fixture} failed on {system_fixture}"
    )

    # Use same convergence tolerances for solution accuracy
    np.testing.assert_allclose(x, x_exact, rtol=rtol, atol=atol)


# =============================================================================
# Test: Residual Equation Verification
# =============================================================================


@pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
@pytest.mark.parametrize(
    "system_fixture",
    [
        "tridiagonal_system_known_solution",
        "diagonal_system_known_solution",
    ],
)
def test_solver_residual_equation(
    solver_name: str,
    system_fixture: str,
    solver_factories: dict[str, Callable],
    convergence_tolerances: tuple[float, float],
    request: pytest.FixtureRequest,
) -> None:
    """Verify residual equation r = b - A @ x is satisfied accurately.

    Theory:
        The residual equation r_k = b - A @ x_k should hold exactly
        in exact arithmetic. Using 10×10 matrices.

    Coverage:
        3 solvers × 2 systems = 6 test runs
    """
    solver = solver_factories[solver_name]
    A, b, _ = request.getfixturevalue(system_fixture)
    rtol, atol = convergence_tolerances

    x, result = solver(A, b, rtol=rtol, atol=atol, max_iterations=200)

    assert result.converged

    # Compute residual explicitly
    r_computed = b - A @ x

    # Verify residual norm matches reported value
    r_norm_computed = float(np.linalg.norm(r_computed))
    np.testing.assert_allclose(r_norm_computed, result.residual_abs, rtol=rtol, atol=atol)

    # Verify residual is small
    np.testing.assert_allclose(r_computed, np.zeros_like(r_computed), atol=atol)


# =============================================================================
# Test: Solution Accuracy vs Direct Solver
# =============================================================================


@pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
@pytest.mark.parametrize(
    "system_fixture",
    ["tridiagonal_system_known_solution", "diagonal_system_known_solution"],
)
def test_solver_matches_direct_solver_high_precision(
    solver_name: str,
    system_fixture: str,
    solver_factories: dict[str, Callable],
    convergence_tolerances: tuple[float, float],
    request: pytest.FixtureRequest,
) -> None:
    """Verify iterative solver matches direct solver at high precision.

    Theory:
        For well-conditioned SPD systems, CG should produce solutions
        that match np.linalg.solve. Using 10×10 matrices.

    Coverage:
        3 solvers × 2 systems = 6 test runs
    """
    solver = solver_factories[solver_name]
    A, b, x_exact = request.getfixturevalue(system_fixture)
    rtol, atol = convergence_tolerances

    x, result = solver(A, b, rtol=rtol, atol=atol, max_iterations=200)

    assert result.converged

    # Use convergence tolerances for solution accuracy
    np.testing.assert_allclose(x, x_exact, rtol=rtol, atol=atol)




# =============================================================================
# Test: Iteration Count Bounds
# =============================================================================


@pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
def test_solver_iteration_count_reasonable(
    solver_name: str,
    solver_factories: dict[str, Callable],
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    jacobi_preconditioner_tridiagonal: Callable[[NDArray], NDArray],
    convergence_tolerances: tuple[float, float],
) -> None:
    """Verify iteration count is reasonable for well-conditioned systems.

    Theory:
        For n×n well-conditioned SPD matrix with good preconditioner,
        CG should converge in O(sqrt(κ)) iterations. Using 10×10 matrix.

    Coverage:
        3 solvers × 1 test each = 3 test runs
    """
    solver = solver_factories[solver_name]
    A, b, _ = tridiagonal_system_known_solution
    rtol, atol = convergence_tolerances

    x, result = solver(
        A,
        b,
        preconditioner=jacobi_preconditioner_tridiagonal,
        rtol=rtol,
        atol=atol,
        max_iterations=200,
    )

    assert result.converged

    # For 10×10 tridiagonal with Jacobi, expect reasonable iteration count
    MAX_EXPECTED_ITERATIONS = 50
    assert result.iterations <= MAX_EXPECTED_ITERATIONS, (
        f"{solver_name} took {result.iterations} iterations, "
        f"expected ≤ {MAX_EXPECTED_ITERATIONS}"
    )


