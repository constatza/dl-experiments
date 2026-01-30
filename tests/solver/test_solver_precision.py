"""Symmetric precision tests for all iterative solvers.

This module tests all solver implementations (FCG, PCG, SciPy CG) with identical
test problems and tolerance requirements. All solvers should pass the same precision
benchmarks.

Test Matrix:
- Solvers: FlexibleCG, PreconditionedCG, SciPyCG
- Problems: Medium-sized (50x50) systems from existing fixtures
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from neuralls.solver import flexible_cg, preconditioned_cg, scipy_cg

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@pytest.mark.parametrize(
    "solver_func,solver_name",
    [
        (flexible_cg, "FCG"),
        (preconditioned_cg, "PCG"),
        (scipy_cg, "SciPy"),
    ],
)
def test_solver_convergence_precision_medium_tridiagonal(
    tridiagonal_system_medium: tuple[NDArray, NDArray, NDArray],
    solver_func: Callable,
    solver_name: str,
    convergence_tolerances: tuple[float, float],
) -> None:
    """Test all solvers converge to high precision on medium tridiagonal system.

    All solvers should:
    1. Converge to the specified tolerance
    2. Produce accurate solution (||x - x_exact|| / ||x_exact|| < 1e-10)
    3. Have accurate final residual
    """
    a, b, x_exact = tridiagonal_system_medium
    rtol, atol = convergence_tolerances

    # Solve
    x, result = solver_func(
        a,
        b,
        rtol=rtol,
        atol=atol,
        maxiter=500,
    )

    # 1. Should converge
    assert result.converged, f"{solver_name} failed to converge"

    # 2. Solution accuracy
    solution_error = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
    assert solution_error < 1e-10, (
        f"{solver_name} solution error {solution_error:.2e} exceeds 1e-10"
    )

    # 3. Residual accuracy
    actual_residual = b - a @ x
    residual_norm = np.linalg.norm(actual_residual)
    rhs_norm = np.linalg.norm(b)
    relative_residual = residual_norm / rhs_norm

    assert relative_residual < rtol, (
        f"{solver_name} relative residual {relative_residual:.2e} exceeds {rtol:.2e}"
    )


@pytest.mark.parametrize(
    "solver_func,solver_name",
    [
        (flexible_cg, "FCG"),
        (preconditioned_cg, "PCG"),
        (scipy_cg, "SciPy"),
    ],
)
def test_solver_convergence_precision_medium_diagonal(
    diagonal_system_medium: tuple[NDArray, NDArray, NDArray],
    solver_func: Callable,
    solver_name: str,
    convergence_tolerances: tuple[float, float],
) -> None:
    """Test all solvers converge to high precision on medium diagonal system."""
    a, b, x_exact = diagonal_system_medium
    rtol, atol = convergence_tolerances

    x, result = solver_func(
        a,
        b,
        rtol=rtol,
        atol=atol,
        maxiter=500,
    )

    # Should converge
    assert result.converged, f"{solver_name} failed to converge"

    # Solution accuracy
    solution_error = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
    assert solution_error < 1e-10, (
        f"{solver_name} solution error {solution_error:.2e} exceeds 1e-10"
    )


@pytest.mark.parametrize(
    "solver_func,solver_name",
    [
        (flexible_cg, "FCG"),
        (preconditioned_cg, "PCG"),
        (scipy_cg, "SciPy"),
    ],
)
def test_solver_iteration_consistency_medium(
    tridiagonal_system_medium: tuple[NDArray, NDArray, NDArray],
    solver_func: Callable,
    solver_name: str,
    convergence_tolerances: tuple[float, float],
) -> None:
    """Test that solvers produce reasonable iteration counts."""
    a, b, x_exact = tridiagonal_system_medium
    rtol, atol = convergence_tolerances

    x, result = solver_func(
        a,
        b,
        rtol=rtol,
        atol=atol,
        maxiter=500,
    )

    # Should converge
    assert result.converged

    # Iteration count should be reasonable (< 200 for well-conditioned 50x50)
    assert result.iterations < 200, (
        f"{solver_name} took {result.iterations} iterations, "
        f"expected < 200 for well-conditioned system"
    )


@pytest.mark.parametrize(
    "solver_func,solver_name",
    [
        (flexible_cg, "FCG"),
        (preconditioned_cg, "PCG"),
        (scipy_cg, "SciPy"),
    ],
)
def test_solver_handles_zero_rhs(
    tridiagonal_spd_medium: NDArray,
    solver_func: Callable,
    solver_name: str,
) -> None:
    """Test all solvers handle zero RHS correctly (x=0 solution)."""
    a = tridiagonal_spd_medium
    b = np.zeros(a.shape[0], dtype=np.float64)
    x_exact = np.zeros(a.shape[0], dtype=np.float64)

    x, result = solver_func(
        a,
        b,
        rtol=1e-12,
        atol=1e-14,
        maxiter=100,
    )

    # Should converge immediately (residual is already zero)
    assert result.converged
    assert result.iterations <= 1  # Should converge in 0 or 1 iteration

    # Solution should be zero
    assert np.linalg.norm(x - x_exact) < 1e-14
