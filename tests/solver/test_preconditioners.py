"""Tests for preconditioner correctness and performance.

This module tests standard preconditioners (Identity, Jacobi, ILU) to ensure:
1. Mathematical correctness (factorization accuracy, sign preservation)
2. Convergence improvement (vs no preconditioning)
3. Performance ordering (ILU ≥ Jacobi ≥ Identity by iteration count)

All tests use integration tolerances (rtol=1e-6, atol=1e-14) since we're
testing functionality, not requiring extreme precision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def test_identity_preconditioner_is_noop(
    identity_preconditioner: Callable[[NDArray, object], NDArray],
    rhs_ones_small: NDArray,
) -> None:
    """Verify identity preconditioner returns copy of input.

    Theory:
        Identity preconditioner: M = I, so z = M^{-1}r = r.
        Should return exact copy without modification.
    """
    r = rhs_ones_small.copy()
    z = identity_preconditioner(r, None)

    # Verify z == r
    np.testing.assert_array_equal(z, r)

    # Verify it's a copy (not same object)
    assert z is not r


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
    from neuralls.solver import flexible_cg

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


def test_ilu_factorization_correctness(
    tridiagonal_spd_small: NDArray,
    ilu_preconditioner_factory: Callable[[NDArray], Callable[[NDArray], NDArray]],
) -> None:
    """Verify ILU factorization approximately satisfies A ≈ LU.

    Theory:
        ILU produces incomplete factorization: A ≈ L @ U
        For well-conditioned SPD matrices, approximation should be good.

    Note:
        We test by verifying M^{-1} @ A ≈ I, where M = LU.
        For perfect factorization, this would be exact.
    """
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import spilu

    A = tridiagonal_spd_small
    n = A.shape[0]

    # Create ILU factorization
    A_csc = csc_matrix(A)
    ilu = spilu(A_csc)

    # Compute M^{-1} @ A by applying ILU to each column of A
    M_inv_A = np.zeros_like(A)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        A_col_i = A @ e_i
        M_inv_A[:, i] = ilu.solve(A_col_i)

    # For good factorization, M^{-1} @ A should be close to identity
    I = np.eye(n)

    # Use Frobenius norm and numpy's comparison
    # Threshold chosen based on expected ILU approximation quality
    ILU_APPROXIMATION_THRESHOLD = 0.5
    np.testing.assert_array_less(
        np.linalg.norm(M_inv_A - I, ord="fro"),
        ILU_APPROXIMATION_THRESHOLD
    )


def test_ilu_improves_over_jacobi(
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    jacobi_preconditioner_tridiagonal: Callable[[NDArray], NDArray],
    ilu_preconditioner_tridiagonal: Callable[[NDArray], NDArray],
    integration_tolerances: tuple[float, float],
) -> None:
    """Verify ILU requires ≤ iterations than Jacobi.

    Theory:
        ILU approximates A^{-1} better than Jacobi for general sparse matrices.
        Should result in better conditioning and fewer iterations.

        Expected: iterations_ilu ≤ iterations_jacobi
    """
    from neuralls.solver import preconditioned_cg

    A, b, _ = tridiagonal_system_known_solution
    rtol, atol = integration_tolerances

    # Solve with Jacobi
    _, result_jacobi = preconditioned_cg(
        A,
        b,
        preconditioner=jacobi_preconditioner_tridiagonal,
        rtol=rtol,
        atol=atol,
        maxiter=100,
    )

    # Solve with ILU
    _, result_ilu = preconditioned_cg(
        A,
        b,
        preconditioner=ilu_preconditioner_tridiagonal,
        rtol=rtol,
        atol=atol,
        maxiter=100,
    )

    # Both should converge
    assert result_jacobi.converged
    assert result_ilu.converged

    # ILU should require ≤ iterations than Jacobi
    assert result_ilu.iterations <= result_jacobi.iterations


def test_preconditioner_ordering(
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    identity_preconditioner: Callable[[NDArray, object], NDArray],
    jacobi_preconditioner_tridiagonal: Callable[[NDArray], NDArray],
    ilu_preconditioner_tridiagonal: Callable[[NDArray], NDArray],
    integration_tolerances: tuple[float, float],
) -> None:
    """Verify preconditioner ordering: ILU ≤ Jacobi ≤ Identity by iteration count.

    Theory:
        Better approximations to A^{-1} lead to better conditioning and
        fewer iterations:
            ILU approximates A^{-1} better than Jacobi
            Jacobi approximates A^{-1} better than Identity

        Expected: iterations_ilu ≤ iterations_jacobi ≤ iterations_identity
    """
    from neuralls.solver import flexible_cg

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

    # Solve with ILU
    _, result_ilu = flexible_cg(
        A,
        b,
        preconditioner=ilu_preconditioner_tridiagonal,
        rtol=rtol,
        atol=atol,
        maxiter=200,
    )

    # All should converge
    assert result_identity.converged
    assert result_jacobi.converged
    assert result_ilu.converged

    # Verify ordering: ILU ≤ Jacobi ≤ Identity
    assert result_ilu.iterations <= result_jacobi.iterations
    assert result_jacobi.iterations <= result_identity.iterations


def test_ilu_on_diagonal_matrix_matches_jacobi(
    diagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    jacobi_preconditioner_factory: Callable[[NDArray], Callable[[NDArray], NDArray]],
    ilu_preconditioner_diagonal: Callable[[NDArray], NDArray],
    integration_tolerances: tuple[float, float],
) -> None:
    """Verify ILU on diagonal matrix gives same results as Jacobi.

    Theory:
        For diagonal matrices, ILU factorization is exact: L = I, U = D.
        Therefore, ILU and Jacobi should give identical results.

        Expected: iterations_ilu == iterations_jacobi
    """
    from neuralls.solver import preconditioned_cg

    A, b, _ = diagonal_system_known_solution
    rtol, atol = integration_tolerances

    # Create Jacobi preconditioner for diagonal matrix
    jacobi_precond = jacobi_preconditioner_factory(A)

    # Solve with Jacobi
    _, result_jacobi = preconditioned_cg(
        A,
        b,
        preconditioner=jacobi_precond,
        rtol=rtol,
        atol=atol,
        maxiter=100,
    )

    # Solve with ILU
    _, result_ilu = preconditioned_cg(
        A,
        b,
        preconditioner=ilu_preconditioner_diagonal,
        rtol=rtol,
        atol=atol,
        maxiter=100,
    )

    # Both should converge
    assert result_jacobi.converged
    assert result_ilu.converged

    # For diagonal matrix, ILU == Jacobi, so same iterations
    assert result_ilu.iterations == result_jacobi.iterations


def test_jacobi_correctness_on_well_conditioned_diagonal(
    jacobi_preconditioner_factory: Callable[[NDArray], Callable[[NDArray], NDArray]],
) -> None:
    """Verify Jacobi preconditioner computes z_i = r_i / A_ii correctly.

    Theory:
        Jacobi preconditioner: z = diag(A)^{-1} @ r
        For diagonal matrix, this simplifies to element-wise division.

    Note:
        This test uses well-conditioned diagonal elements to avoid
        numerical issues. For safety with near-zero diagonals, use
        JacobiBuilder from neuralls.preconditioner.builders.
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
