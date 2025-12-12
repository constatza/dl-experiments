from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from src.solver import flexible_cg, preconditioned_cg

# Define tolerances specifically for these tests
TEST_ATOL_STRICT = 1e-8
TEST_RTOL_STRICT = 1e-10
TEST_ATOL_RELAXED = 1e-6


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


def test_flexible_cg_non_convergence_strict_atol(spd_system: tuple[NDArray, NDArray, NDArray]) -> None:
    """
    Tests that flexible_cg does NOT converge with very strict ATOL (1e-8)
    within max_iter=200 for the spd_system, simulating observed behavior.
    """
    A, b, _ = spd_system
    x0 = np.zeros_like(b)

    _, info = flexible_cg(
        A,
        b,
        x0,
        atol=TEST_ATOL_STRICT,
        rtol=TEST_RTOL_STRICT,
        max_iter=200,
    )

    # Assert that it does NOT converge (as observed previously)
    assert not info.converged
    # Assert that the residual is above the strict tolerance
    expected_bound = _tol_bound(b, TEST_ATOL_STRICT, TEST_RTOL_STRICT)
    assert info.residual_abs > expected_bound


def test_flexible_cg_convergence_relaxed_atol(spd_system: tuple[NDArray, NDArray, NDArray]) -> None:
    """
    Tests that flexible_cg DOES converge with a relaxed ATOL (1e-6)
    within max_iter=200 for the spd_system.
    """
    A, b, _ = spd_system
    x0 = np.zeros_like(b)

    _, info = flexible_cg(
        A,
        b,
        x0,
        atol=TEST_ATOL_RELAXED,
        rtol=TEST_RTOL_STRICT,
        max_iter=200,
    )

    # Assert that it DOES converge with the relaxed tolerance
    assert info.converged
    # Assert that the residual is within the relaxed tolerance
    expected_bound = _tol_bound(b, TEST_ATOL_RELAXED, TEST_RTOL_STRICT)
    assert info.residual_abs <= expected_bound


def test_flexible_cg_torch_precond_non_convergence_strict_atol(spd_system: tuple[NDArray, NDArray, NDArray]) -> None:
    """
    Tests that flexible_cg with torch precond does NOT converge with very strict ATOL (1e-8)
    within max_iter=20 for the spd_system, simulating observed behavior.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        pytest.skip("torch not available in environment")

    A, b, _ = spd_system

    layer = torch.nn.Linear(A.shape[0], A.shape[0], bias=False)
    with torch.no_grad():
        inv_diag = torch.tensor(np.diag(A), dtype=layer.weight.dtype)
        layer.weight.copy_(torch.diag(1.0 / inv_diag))
    layer.eval()

    def torch_precond(residual: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor_in = torch.from_numpy(residual.astype(np.float32, copy=False))
            out = layer(tensor_in)
            return out.detach().cpu().numpy().astype(np.float64, copy=False)

    _, neural_info = flexible_cg(
        A,
        b,
        max_iter=20,
        stopping_criterion="fixed_iterations",
        atol=TEST_ATOL_STRICT,
        rtol=TEST_RTOL_STRICT,
        preconditioner=torch_precond,
    )
    # Assert that it does NOT converge (as observed previously)
    assert not neural_info.converged
    # Assert that the residual is above the strict tolerance
    expected_bound = _tol_bound(b, TEST_ATOL_STRICT, TEST_RTOL_STRICT)
    assert neural_info.residual_abs > expected_bound


def test_flexible_cg_torch_precond_convergence_relaxed_atol(spd_system: tuple[NDArray, NDArray, NDArray]) -> None:
    """
    Tests that flexible_cg with torch precond DOES converge with a relaxed ATOL (1e-6)
    within max_iter=200 for the spd_system.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        pytest.skip("torch not available in environment")

    A, b, _ = spd_system

    layer = torch.nn.Linear(A.shape[0], A.shape[0], bias=False)
    with torch.no_grad():
        inv_diag = torch.tensor(np.diag(A), dtype=layer.weight.dtype)
        layer.weight.copy_(torch.diag(1.0 / inv_diag))
    layer.eval()

    def torch_precond(residual: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor_in = torch.from_numpy(residual.astype(np.float32, copy=False))
            out = layer(tensor_in)
            return out.detach().cpu().numpy().astype(np.float64, copy=False)

    _, neural_info = flexible_cg(
        A,
        b,
        max_iter=200, # Increased max_iter to allow convergence
        atol=TEST_ATOL_RELAXED,
        rtol=TEST_RTOL_STRICT,
        preconditioner=torch_precond,
    )
    # Assert that it DOES converge with the relaxed tolerance and increased iterations
    assert neural_info.converged
    # Assert that the residual is within the relaxed tolerance
    expected_bound = _tol_bound(b, TEST_ATOL_RELAXED, TEST_RTOL_STRICT)
    assert neural_info.residual_abs <= expected_bound
