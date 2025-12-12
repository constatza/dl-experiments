from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from src.solver import flexible_cg, preconditioned_cg

# Define very strict tolerances for double precision accuracy goals
DOUBLE_PRECISION_ATOL = 1e-14
DOUBLE_PRECISION_RTOL = 1e-15 # Closer to machine epsilon for float64


def _tol_bound(b: NDArray, atol: float, rtol: float) -> float:
    return max(atol, rtol * float(np.linalg.norm(b)))


@pytest.fixture(scope="module")
def spd_system() -> tuple[NDArray, NDArray, NDArray]:
    """Deterministic 20x20 SPD system with known solution."""
    n = 20
    diag = np.full(n, 4.0, dtype=np.float64)
    off1 = np.full(n - 1, -0.5, dtype=np.float64)
    off2 = np.full(n - 2, -0.1, dtype=np.float64)
    A = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(A, diag)
    np.fill_diagonal(A[1:], off1)
    np.fill_diagonal(A[:, 1:], off1)
    np.fill_diagonal(A[2:], off2)
    np.fill_diagonal(A[:, 2:], off2)
    solution = np.linspace(1.0, 2.0, n, dtype=np.float64)
    b = A @ solution
    return A, b, solution


def jacobi_preconditioner_callable(A_matrix: NDArray) -> LinearOperator:
    """Creates a Jacobi preconditioner as a SciPy LinearOperator."""
    diag_inv = 1.0 / np.diag(A_matrix)

    def matvec(vec: np.ndarray) -> np.ndarray:
        return diag_inv * vec

    return LinearOperator(matvec=matvec, shape=A_matrix.shape, dtype=np.float64)


def test_pcg_jacobi_double_precision_accuracy(spd_system: tuple[NDArray, NDArray, NDArray]) -> None:
    """
    Tests that preconditioned_cg (SciPy wrapper) with Jacobi preconditioner
    can reach double precision accuracy for the spd_system.
    """
    A, b, x_true = spd_system
    x0 = np.zeros_like(b)

    jacobi_M = jacobi_preconditioner_callable(A)

    x_sol, info = preconditioned_cg(
        A,
        b,
        x0,
        preconditioner=jacobi_M,
        atol=DOUBLE_PRECISION_ATOL,
        rtol=DOUBLE_PRECISION_RTOL,
        max_iter=200, # Sufficiently high max_iter
    )

    assert info.converged
    assert info.residual_abs <= _tol_bound(b, DOUBLE_PRECISION_ATOL, DOUBLE_PRECISION_RTOL)
    assert np.allclose(x_sol, x_true, atol=DOUBLE_PRECISION_ATOL, rtol=DOUBLE_PRECISION_RTOL)


def test_fcg_jacobi_double_precision_accuracy(spd_system: tuple[NDArray, NDArray, NDArray]) -> None:
    """
    Tests that flexible_cg with Jacobi preconditioner can reach double precision accuracy
    for the spd_system. This is a challenging goal for FCG.
    """
    A, b, x_true = spd_system
    x0 = np.zeros_like(b)

    jacobi_M = jacobi_preconditioner_callable(A)

    x_sol, info = flexible_cg(
        A,
        b,
        x0,
        preconditioner=jacobi_M,
        atol=DOUBLE_PRECISION_ATOL,
        rtol=DOUBLE_PRECISION_RTOL,
        max_iter=200, # Sufficiently high max_iter
        m_max=20, # Use a larger m_max for better orthogonalization
        # Using default eps_curv and eps_breakdown
    )

    assert info.converged
    assert info.residual_abs <= _tol_bound(b, DOUBLE_PRECISION_ATOL, DOUBLE_PRECISION_RTOL)
    assert np.allclose(x_sol, x_true, atol=DOUBLE_PRECISION_ATOL, rtol=DOUBLE_PRECISION_RTOL)
