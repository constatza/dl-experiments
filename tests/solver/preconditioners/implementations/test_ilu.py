"""Tests for ILU preconditioner.

Tests Incomplete LU factorization preconditioner to ensure:
- Correctly factorizes sparse matrices
- Improves convergence vs Jacobi
- Accepts both sparse and dense matrices
- Handles edge cases
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neuralls.domain.solver.preconditioners import ILUPreconditioner

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


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
    identity = np.eye(n)

    # Use Frobenius norm and numpy's comparison
    # Threshold chosen based on expected ILU approximation quality
    ILU_APPROXIMATION_THRESHOLD = 0.5
    np.testing.assert_array_less(
        np.linalg.norm(M_inv_A - identity, ord="fro"), ILU_APPROXIMATION_THRESHOLD
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
    from neuralls.domain.solver import pcg

    A, b, _ = tridiagonal_system_known_solution
    rtol, atol = integration_tolerances

    # Solve with Jacobi
    _, result_jacobi = pcg(
        A,
        b,
        preconditioner=jacobi_preconditioner_tridiagonal,
        rtol=rtol,
        atol=atol,
        maxiter=100,
    )

    # Solve with ILU
    _, result_ilu = pcg(
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
    from neuralls.domain.solver import pcg

    A, b, _ = diagonal_system_known_solution
    rtol, atol = integration_tolerances

    # Create Jacobi preconditioner for diagonal matrix
    jacobi_precond = jacobi_preconditioner_factory(A)

    # Solve with Jacobi
    _, result_jacobi = pcg(
        A,
        b,
        preconditioner=jacobi_precond,
        rtol=rtol,
        atol=atol,
        maxiter=100,
    )

    # Solve with ILU
    _, result_ilu = pcg(
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


def test_ilu_preconditioner_from_matrix() -> None:
    """Verify ILU preconditioner computes factorization from matrix."""
    # Simple SPD tridiagonal matrix
    A = np.array([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 4.0]])

    # Pass matrix directly - preconditioner computes ILU internally
    precond = ILUPreconditioner(A)

    r = np.array([1.0, 2.0, 3.0])
    z = precond.apply(r)

    # Verify output is reasonable (not NaN/Inf)
    assert np.all(np.isfinite(z))
    assert z.shape == r.shape

    # ILU should approximate A^{-1}, so A @ z ≈ r
    result = A @ z
    np.testing.assert_allclose(result, r, rtol=1e-2)  # Approximate


def test_ilu_preconditioner_with_larger_matrix() -> None:
    """Verify ILU with larger tridiagonal matrix."""
    # 5x5 tridiagonal matrix
    n = 5
    A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)

    precond = ILUPreconditioner(A)
    r = np.ones(n)
    z = precond.apply(r)

    # Verify output is reasonable
    assert np.all(np.isfinite(z))
    assert z.shape == (n,)


def test_ilu_preconditioner_accepts_dense_matrix(
    dense_spd_matrix: NDArray,
) -> None:
    """Test ILUPreconditioner with dense matrix input.

    Preconditioner should convert to CSC format internally.
    """
    # Should not raise error
    precond = ILUPreconditioner(dense_spd_matrix)

    # Should apply without error
    residual_5d = np.ones(5)
    result = precond.apply(residual_5d)

    # Result should have correct shape
    assert result.shape == (5,)
    assert result.dtype == np.float64


def test_ilu_preserves_dtype(dense_spd_matrix: NDArray) -> None:
    """Test that ILUPreconditioner preserves float64 dtype."""
    precond = ILUPreconditioner(dense_spd_matrix)
    residual = np.ones(5, dtype=np.float64)
    result = precond.apply(residual)

    assert result.dtype == np.float64
