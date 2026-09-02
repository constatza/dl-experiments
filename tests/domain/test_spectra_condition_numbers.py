"""Tests for spectra.py's matrix-free (power-iteration) condition number estimate.

Two things are checked against independently-computed oracles (never via
spectra.py's own removed helpers):

1. Correctness — the estimate matches the *exact eigenvalue ratio*
   (``np.linalg.eigvals``) of the preconditioned operator, which is what
   power iteration is designed to estimate. This is NOT always the same as
   the SVD-based 2-norm condition number (``np.linalg.cond``) the old code
   computed: they coincide only when the preconditioned operator is
   symmetric/normal (e.g. any diagonal case below), and legitimately diverge
   for a general non-symmetric ``M^-1 A`` — that's a deliberate, disclosed
   change of definition, not a bug.
2. Speed — the entire point of replacing the dense-matrix-plus-SVD approach
   is that it's faster for real (large) matrices; that's verified directly
   by timing both on a matrix large enough for the difference to show.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch
from torchalg.preconditioners.base import Preconditioner
from torchalg.preconditioners.implementations import Identity, JacobiPreconditioner

from neuralls.domain.analysis.spectra import compute_condition_numbers


def _exact_eigenvalue_ratio(matrix: torch.Tensor, preconditioner: Preconditioner) -> float:
    """Ground-truth eigenvalue-ratio condition number: exact dense build + eigvals.

    This is the same quantity ``compute_condition_numbers`` estimates via
    power iteration — the correct oracle for correctness checks.
    """
    columns = [preconditioner(matrix[:, idx]) for idx in range(matrix.shape[1])]
    precond_matrix = torch.stack(columns, dim=1).numpy()
    eigenvalues = np.linalg.eigvals(precond_matrix).real
    return float(eigenvalues.max() / eigenvalues.min())


def _exact_svd_condition_number(matrix: torch.Tensor, preconditioner: Preconditioner) -> float:
    """The old production method (dense build + ``np.linalg.cond``) being replaced.

    Used only for the speed comparison below — not a correctness oracle,
    since it measures a different quantity (SVD 2-norm condition number) for
    non-symmetric preconditioned operators.
    """
    columns = [preconditioner(matrix[:, idx]) for idx in range(matrix.shape[1])]
    precond_matrix = torch.stack(columns, dim=1).numpy()
    return float(np.linalg.cond(precond_matrix))


@pytest.fixture
def diagonal_matrix() -> torch.Tensor:
    """A diagonal SPD matrix with an analytically known 2-norm condition number of 100."""
    return torch.diag(torch.tensor([1.0, 10.0, 100.0], dtype=torch.float64))


@pytest.fixture
def tridiagonal_spd_matrix() -> torch.Tensor:
    """A small, well-conditioned symmetric tridiagonal SPD matrix."""
    return torch.tensor([[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]], dtype=torch.float64)


@pytest.fixture
def dense_spd_matrix() -> torch.Tensor:
    """A small, wider-spectrum dense SPD matrix (deterministic, not random)."""
    return torch.tensor([[10.0, 2.0, 1.0], [2.0, 5.0, 0.5], [1.0, 0.5, 500.0]], dtype=torch.float64)


def test_diagonal_matrix_identity_matches_analytical_condition_number(
    diagonal_matrix: torch.Tensor,
) -> None:
    """For the identity preconditioner, the estimate is exact for a diagonal matrix."""
    cond_numbers = compute_condition_numbers(diagonal_matrix.numpy(), {"none": Identity()})
    assert cond_numbers["none"] == pytest.approx(100.0, rel=1e-2)


def test_diagonal_matrix_jacobi_matches_analytical_condition_number(
    diagonal_matrix: torch.Tensor,
) -> None:
    """Jacobi-preconditioning a diagonal matrix yields the identity operator (condition 1)."""
    cond_numbers = compute_condition_numbers(
        diagonal_matrix.numpy(), {"jacobi": JacobiPreconditioner(diagonal_matrix)}
    )
    assert cond_numbers["jacobi"] == pytest.approx(1.0, rel=1e-2)


@pytest.mark.parametrize("matrix_name", ["tridiagonal_spd_matrix", "dense_spd_matrix"])
@pytest.mark.parametrize("preconditioner_name", ["identity", "jacobi"])
def test_power_iteration_matches_exact_eigenvalue_ratio(
    request: pytest.FixtureRequest, matrix_name: str, preconditioner_name: str
) -> None:
    """The fast estimate agrees with the exact eigenvalue-ratio condition number."""
    matrix: torch.Tensor = request.getfixturevalue(matrix_name)
    preconditioner: Preconditioner = (
        Identity() if preconditioner_name == "identity" else JacobiPreconditioner(matrix)
    )

    expected = _exact_eigenvalue_ratio(matrix, preconditioner)
    cond_numbers = compute_condition_numbers(matrix.numpy(), {preconditioner_name: preconditioner})

    assert cond_numbers[preconditioner_name] == pytest.approx(expected, rel=1e-2)


def test_power_iteration_is_faster_than_exact_dense_svd() -> None:
    """The whole point of the matrix-free estimate is to beat the O(n^3) exact method.

    A wide-spectrum, diagonally-dominant SPD matrix at a size (n=600) large
    enough that the old dense-build-plus-SVD approach's cost is measurable,
    but small enough to keep the test itself fast.
    """
    n = 600
    diag_values = torch.linspace(1.0, 1.0e4, n, dtype=torch.float64)
    off_diag = 0.1 * torch.ones(n - 1, dtype=torch.float64)
    matrix = torch.diag(diag_values) + torch.diag(off_diag, 1) + torch.diag(off_diag, -1)
    preconditioner = JacobiPreconditioner(matrix)

    start = time.perf_counter()
    compute_condition_numbers(matrix.numpy(), {"jacobi": preconditioner})
    fast_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    _exact_svd_condition_number(matrix, preconditioner)
    exact_elapsed = time.perf_counter() - start

    assert fast_elapsed < exact_elapsed * 0.5, (
        f"power-iteration estimate ({fast_elapsed:.3f}s) should be markedly faster than "
        f"the exact dense-SVD method it replaces ({exact_elapsed:.3f}s)"
    )
