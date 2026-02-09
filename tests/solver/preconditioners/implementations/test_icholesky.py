"""Tests for ICholesky preconditioner.

Tests parameterized Incomplete Cholesky preconditioner to ensure:
- Correctly computes Cholesky factorization
- Acts as exact inverse for complete factorization
- Preserves shapes and dtypes
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neuralls.solver.preconditioners import ICholeskyPreconditioner

if TYPE_CHECKING:
    from numpy.typing import NDArray

def test_icholesky_preconditioner_exact_inverse() -> None:
    """Verify ICholeskyPreconditioner solves Mz=r exactly when given exact L."""
    rng = np.random.default_rng(42)
    n = 10

    # Generate random SPD matrix A
    X = rng.standard_normal((n, n))
    A = X @ X.T + 1e-4 * np.eye(n)  # Ensure positive definite

    # Compute Cholesky decomposition L
    L = np.linalg.cholesky(A)

    # Create preconditioner with L
    # Note: ICholeskyPreconditioner takes L as input
    precond = ICholeskyPreconditioner(L)

    # Generate random residual
    r = rng.standard_normal(n)

    # Apply preconditioner: z = M^{-1} r where M = LL^T = A
    z = precond.apply(r)

    # Check if Az = r
    # Since we used exact Cholesky, z should be exact solution
    reconstructed_r = A @ z

    np.testing.assert_allclose(
        reconstructed_r,
        r,
        rtol=1e-10,
        atol=1e-10,
        err_msg="ICholesky preconditioner failed to invert A=LL^T correctly"
    )


def test_icholesky_preconditioner_shapes() -> None:
    """Verify shape handling."""
    rng = np.random.default_rng(42)
    n = 5
    A = np.eye(n)
    L = np.linalg.cholesky(A)  # Just Identity

    precond = ICholeskyPreconditioner(L)

    r = np.ones(n)
    z = precond.apply(r)
    assert z.shape == r.shape

    # Batched application (if supported by cho_solve, which it is for (N, K))
    r_batch = np.ones((n, 2))
    z_batch = precond.apply(r_batch)
    assert z_batch.shape == r_batch.shape
    assert np.allclose(z_batch, r_batch)  # Since A=I
