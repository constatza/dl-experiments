"""Tests for Jacobi preconditioner.

Tests diagonal scaling preconditioner to ensure:
- Correctly computes z_i = r_i / A_ii
- Preserves signs correctly
- Improves convergence vs identity
- Handles near-zero diagonal elements
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neuralls.domain.solver.preconditioners import JacobiPreconditioner

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def test_jacobi_preserves_signs(
    jacobi_preconditioner_factory: Callable[[NDArray], Callable[[NDArray], NDArray]],
) -> None:
    """Verify Jacobi preconditioner preserves sign of residual components.

    Theory:
        Jacobi: z_i = r_i / A_ii
        For positive diagonal A_ii > 0, sign(z_i) = sign(r_i).
        For negative diagonal A_ii < 0, sign(z_i) = sign(r_i) (inverted twice).

    This test ensures Jacobi handles negative diagonal elements correctly.
    """
    # Create diagonal matrix with mixed signs
    diag = np.array([2.0, -3.0, 4.0, -5.0, 1.0], dtype=np.float64)
    A = np.diag(diag)
    precond = jacobi_preconditioner_factory(A)

    # Test residual with mixed signs
    r = np.array([1.0, -1.0, 2.0, -2.0, 0.5], dtype=np.float64)
    z = precond(r)

    # Expected: z_i = r_i / A_ii
    expected = r / diag
    np.testing.assert_allclose(z, expected, rtol=1e-14)

    # Verify signs are preserved correctly
    assert np.all(np.sign(z) == np.sign(expected))


def test_jacobi_improves_convergence(
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    identity_preconditioner: Callable[[NDArray, object], NDArray],
    jacobi_preconditioner_tridiagonal: Callable[[NDArray], NDArray],
    integration_tolerances: tuple[float, float],
) -> None:
    """Verify Jacobi preconditioner reduces iterations vs identity.

    Theory:
        For diagonally dominant matrices, Jacobi preconditioning improves
        the condition number, leading to faster convergence.

        Expected: iterations_jacobi ≤ iterations_identity
    """
    from neuralls.domain.solver import flexible_cg

    A, b, _ = tridiagonal_system_known_solution
    rtol, atol = integration_tolerances

    # Solve with identity (no preconditioning)
    _, result_identity = flexible_cg(
        A,
        b,
        preconditioner=lambda r: identity_preconditioner(r, None),
        rtol=rtol,
        atol=atol,
        maxiter=200,
    )

    # Solve with Jacobi
    _, result_jacobi = flexible_cg(
        A,
        b,
        preconditioner=jacobi_preconditioner_tridiagonal,
        rtol=rtol,
        atol=atol,
        maxiter=200,
    )

    # Both should converge
    assert result_identity.converged
    assert result_jacobi.converged

    # Jacobi should require ≤ iterations
    assert result_jacobi.iterations <= result_identity.iterations


def test_jacobi_correctness_on_well_conditioned_diagonal(
    jacobi_preconditioner_factory: Callable[[NDArray], Callable[[NDArray], NDArray]],
) -> None:
    """Verify Jacobi preconditioner computes z_i = r_i / A_ii correctly.

    Theory:
        Jacobi preconditioner: z = diag(A)^{-1} @ r
        For diagonal matrix, this simplifies to element-wise division.

    Note:
        This test uses well-conditioned diagonal elements to avoid
        numerical issues.
    """
    # Create well-conditioned diagonal matrix
    A = np.diag([2.0, 3.0, 4.0, 5.0, 1.0])
    precond = jacobi_preconditioner_factory(A)

    # Apply to test residual
    r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = precond(r)

    # Expected: z_i = r_i / A_ii
    expected = r / np.diag(A)
    np.testing.assert_allclose(z, expected, rtol=1e-14, atol=1e-14)


def test_jacobi_preconditioner_from_matrix() -> None:
    """Verify Jacobi preconditioner computes diagonal inverse from matrix."""
    # Simple diagonal matrix
    A = np.diag([2.0, 4.0, 1.0])

    # Pass matrix - preconditioner computes diagonal inverse internally
    precond = JacobiPreconditioner(A)

    r = np.array([2.0, 4.0, 1.0])
    z = precond.apply(r)

    # Expected: z_i = r_i / A_ii = [2/2, 4/4, 1/1] = [1.0, 1.0, 1.0]
    expected = np.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(z, expected)


def test_jacobi_preconditioner_with_tridiagonal(tridiagonal_spd_small: NDArray) -> None:
    """Verify Jacobi preconditioner from actual SPD matrix."""
    A = tridiagonal_spd_small

    # Pass matrix directly
    precond = JacobiPreconditioner(A)

    r = np.ones(A.shape[0])
    z = precond.apply(r)

    # z_i = r_i / A_ii for diagonal scaling
    expected = r / np.diag(A)
    np.testing.assert_allclose(z, expected)


def test_jacobi_handles_near_zero_diagonal() -> None:
    """Verify Jacobi preconditioner protects against near-zero diagonals."""
    # Matrix with near-zero diagonal element
    A = np.diag([2.0, 1e-15, 1.0])

    precond = JacobiPreconditioner(A)
    r = np.array([2.0, 4.0, 1.0])
    z = precond.apply(r)

    # Near-zero diagonal should be treated as 1.0
    # Expected: [2/2, 4/1, 1/1] = [1.0, 4.0, 1.0]
    expected = np.array([1.0, 4.0, 1.0])
    np.testing.assert_allclose(z, expected)


def test_jacobi_preserves_dtype(well_conditioned_matrix: NDArray) -> None:
    """Test that JacobiPreconditioner preserves float64 dtype."""
    precond = JacobiPreconditioner(well_conditioned_matrix)
    residual = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    result = precond.apply(residual)

    assert result.dtype == np.float64
