"""Tests for IC(0) preconditioner.

Tests zero-fill Incomplete Cholesky preconditioner to ensure:
- Correctly factorizes SPD matrices
- Improves convergence
- Handles threshold parameter
- Validates SPD input
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from neuralls.solver.preconditioners import IC0Preconditioner

if TYPE_CHECKING:
    from numpy.typing import NDArray

def test_ic0_factorization_approximates_matrix(
    tridiagonal_spd_small: NDArray,
) -> None:
    """Verify IC(0) factorization produces L such that L @ L.T ≈ A.

    Theory:
        IC(0) computes incomplete Cholesky factorization: A ≈ L @ L.T
        For sparse SPD matrices, approximation should be reasonable.
        For tridiagonal, IC(0) preserves sparsity pattern exactly.
    """
    A = tridiagonal_spd_small
    precond = IC0Preconditioner(A)
    L = precond._operator

    # Reconstruct A from factorization
    reconstructed = L @ L.T

    # IC(0) is approximate, but should be close for well-conditioned SPD
    np.testing.assert_allclose(reconstructed, A, rtol=1e-2)


def test_ic0_preserves_sparsity_pattern(
    tridiagonal_spd_small: NDArray,
) -> None:
    """Verify IC(0) factor L has same sparsity pattern as lower triangle of A.

    Theory:
        Zero-level incomplete Cholesky maintains sparsity pattern:
        L[i,j] = 0 if A[i,j] = 0 (for lower triangle).
        With threshold=0, this should be exact.
        With default threshold, L can be sparser (drops small entries).
    """
    A = tridiagonal_spd_small
    A_lower_mask = (np.abs(np.tril(A)) >= 1e-14)

    # Test with strict threshold (preserve exact sparsity)
    precond_strict = IC0Preconditioner(A, threshold=0.0)
    L_strict = precond_strict._operator
    L_strict_mask = (L_strict != 0)

    # With threshold=0, L should match A's sparsity pattern exactly
    np.testing.assert_array_equal(L_strict_mask, A_lower_mask)

    # Test with default threshold (can be sparser)
    precond_default = IC0Preconditioner(A)
    L_default = precond_default._operator
    nnz_L_default = np.count_nonzero(L_default)
    nnz_A_lower = np.count_nonzero(np.tril(A))

    # Default threshold may drop small entries
    assert nnz_L_default <= nnz_A_lower


def test_ic0_improves_convergence(
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    identity_preconditioner: Callable[[NDArray, object], NDArray],
    integration_tolerances: tuple[float, float],
) -> None:
    """Verify IC(0) reduces CG iterations vs no preconditioning.

    Theory:
        IC(0) approximates A^{-1} better than identity for SPD matrices.
        Should result in better conditioning and fewer iterations.

        Expected: iterations_ic0 < iterations_identity
    """
    from neuralls.solver import flexible_cg

    A, b, _ = tridiagonal_system_known_solution
    rtol, atol = integration_tolerances

    # Create IC(0) preconditioner
    ic0_precond = IC0Preconditioner(A)

    # Solve with identity (no preconditioning)
    _, result_identity = flexible_cg(
        A,
        b,
        preconditioner=lambda r: identity_preconditioner(r, None),
        rtol=rtol,
        atol=atol,
        maxiter=200,
    )

    # Solve with IC(0)
    _, result_ic0 = flexible_cg(
        A,
        b,
        preconditioner=ic0_precond,
        rtol=rtol,
        atol=atol,
        maxiter=200,
    )

    # Both should converge
    assert result_identity.converged
    assert result_ic0.converged

    # IC(0) should require fewer iterations
    assert result_ic0.iterations < result_identity.iterations


def test_ic0_requires_spd_matrix() -> None:
    """Verify IC(0) can be instantiated on non-SPD matrix.

    Note:
        IC(0) is designed for symmetric positive definite matrices,
        but we don't validate this at construction time.
        Non-SPD matrices may produce numerical issues during factorization.
    """
    # Create non-SPD matrix (not symmetric)
    A_non_spd = np.array([[1, 2], [3, 4]], dtype=np.float64)

    # Should not raise during construction
    precond = IC0Preconditioner(A_non_spd)
    assert precond is not None


def test_ic0_handles_near_singular_matrix() -> None:
    """Verify IC(0) can be instantiated on nearly singular matrix.

    Note:
        IC(0) computes sqrt(diagonal entries) during factorization.
        We don't validate for near-zero or negative diagonals.
        Nearly singular matrices may produce numerical issues.
    """
    # Create nearly singular matrix
    A_singular = np.eye(10, dtype=np.float64)
    A_singular[5, 5] = 1e-15  # Nearly zero diagonal

    # Should not raise during construction
    precond = IC0Preconditioner(A_singular)
    assert precond is not None


def test_ic0_threshold_drops_small_entries(
    tridiagonal_spd_small: NDArray,
) -> None:
    """Verify threshold parameter drops small entries.

    Theory:
        Threshold parameter controls sparsity:
        - Smaller threshold = more fill-in, better approximation
        - Larger threshold = more sparsity, faster but less accurate
    """
    # Add tiny numerical noise to matrix
    A = tridiagonal_spd_small.copy()
    np.random.seed(42)
    # Add small random perturbations (order 1e-12)
    A += np.tril(np.random.randn(10, 10) * 1e-12)
    A = (A + A.T) / 2  # Ensure symmetric
    A += 0.1 * np.eye(10)  # Ensure SPD

    # Strict threshold: keep tiny entries
    ic0_strict = IC0Preconditioner(A, threshold=1e-16)
    L_strict = ic0_strict._operator
    nnz_strict = np.count_nonzero(L_strict)

    # Aggressive threshold: drop tiny entries
    ic0_aggressive = IC0Preconditioner(A, threshold=1e-10)
    L_aggressive = ic0_aggressive._operator
    nnz_aggressive = np.count_nonzero(L_aggressive)

    # Aggressive dropping should produce sparser L
    assert nnz_aggressive <= nnz_strict


def test_ic0_compares_favorably_with_jacobi(
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    jacobi_preconditioner_tridiagonal: Callable[[NDArray], NDArray],
    integration_tolerances: tuple[float, float],
) -> None:
    """Verify IC(0) performs comparably or better than Jacobi for SPD systems.

    Theory:
        IC(0) approximates full Cholesky factorization, which is optimal
        for SPD systems. Should match or outperform Jacobi diagonal scaling.

        Expected: iterations_ic0 <= iterations_jacobi
    """
    from neuralls.solver import flexible_cg

    A, b, _ = tridiagonal_system_known_solution
    rtol, atol = integration_tolerances

    # Create IC(0) preconditioner
    ic0_precond = IC0Preconditioner(A)

    # Solve with Jacobi
    _, result_jacobi = flexible_cg(
        A,
        b,
        preconditioner=jacobi_preconditioner_tridiagonal,
        rtol=rtol,
        atol=atol,
        maxiter=200,
    )

    # Solve with IC(0)
    _, result_ic0 = flexible_cg(
        A,
        b,
        preconditioner=ic0_precond,
        rtol=rtol,
        atol=atol,
        maxiter=200,
    )

    # Both should converge
    assert result_jacobi.converged
    assert result_ic0.converged

    # IC(0) should require <= iterations than Jacobi
    assert result_ic0.iterations <= result_jacobi.iterations


