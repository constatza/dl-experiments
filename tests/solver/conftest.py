"""Shared fixtures for solver tests.

This module provides composable, modular fixtures for testing the flexible PCG
solver. All test data is created via fixtures following project guidelines:
- NEVER create data inside test functions
- Use modular, composable fixtures
- Separate data creation from test logic
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.sparse import diags

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


# =============================================================================
# Constants for Test Configuration
# =============================================================================
SMALL_MATRIX_SIZE = 10
MEDIUM_MATRIX_SIZE = 50
TRIDIAGONAL_DIAG_VALUE = 4.0
TRIDIAGONAL_OFFDIAG_VALUE = -1.0
TEST_SEED = 42
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-14
DEFAULT_ASSERT_RTOL = 1e-5
DEFAULT_MAX_ITER = 1000


# =============================================================================
# SPD Matrix Fixtures
# =============================================================================


@pytest.fixture
def identity_matrix_small() -> NDArray:
    """Small identity matrix (10x10) - trivial SPD system.

    Returns:
        10x10 identity matrix as dense array.

    Theory:
        Identity matrix has eigenvalues all equal to 1, condition number 1.
        CG converges in exactly 1 iteration for any preconditioner.
    """
    return np.eye(SMALL_MATRIX_SIZE, dtype=np.float64)


@pytest.fixture
def tridiagonal_spd_small() -> NDArray:
    """Small tridiagonal SPD matrix (10x10) - well-conditioned.

    Returns:
        10x10 tridiagonal matrix: diag=4, off-diag=-1.

    Theory:
        Tridiagonal matrix with diag=4, off-diag=-1 is SPD.
        Eigenvalues: λ_k = 4 - 2*cos(k*π/(n+1)) for k=1..n.
        Condition number κ ≈ 4n²/π² for large n.
        For n=10: κ ≈ 40, well-conditioned.
    """
    n = SMALL_MATRIX_SIZE
    sparse_mat = diags(
        [TRIDIAGONAL_OFFDIAG_VALUE, TRIDIAGONAL_DIAG_VALUE, TRIDIAGONAL_OFFDIAG_VALUE],
        [-1, 0, 1],
        shape=(n, n),
        format="csr",
        dtype=np.float64,
    )
    return sparse_mat.toarray()  # Convert to dense for solver compatibility


@pytest.fixture
def diagonal_spd_small() -> NDArray:
    """Small diagonal SPD matrix (10x10) with varying diagonal entries.

    Returns:
        10x10 diagonal matrix with entries [1, 2, ..., 10].

    Theory:
        Diagonal matrix is trivially SPD if all diagonal entries > 0.
        Eigenvalues are the diagonal entries.
        Condition number κ = 10/1 = 10 (well-conditioned).
        CG converges in at most n steps, but often faster.
    """
    n = SMALL_MATRIX_SIZE
    return np.diag(np.arange(1, n + 1, dtype=np.float64))


@pytest.fixture
def random_spd_small(test_seed: int) -> NDArray:
    """Small random SPD matrix (10x10) with fixed seed.

    Args:
        test_seed: Random seed for reproducibility.

    Returns:
        10x10 random SPD matrix.

    Theory:
        Construct SPD matrix as A = Q @ D @ Q.T where:
        - Q is random orthogonal matrix (from QR decomposition)
        - D is diagonal with positive eigenvalues
        This guarantees SPD with controlled condition number.
    """
    n = SMALL_MATRIX_SIZE
    rng = np.random.default_rng(test_seed)

    # Generate random matrix and extract orthogonal Q
    random_mat = rng.standard_normal((n, n))
    q, _ = np.linalg.qr(random_mat)

    # Create diagonal with eigenvalues in range [1, 10]
    eigenvalues = rng.uniform(1.0, 10.0, n)
    d = np.diag(eigenvalues)

    # Construct SPD matrix
    return q @ d @ q.T


@pytest.fixture
def ill_conditioned_spd_small(test_seed: int) -> NDArray:
    """Small ill-conditioned SPD matrix (10x10) - challenging system.

    Args:
        test_seed: Random seed for reproducibility.

    Returns:
        10x10 SPD matrix with condition number ~1000.

    Theory:
        Ill-conditioned systems have large condition number κ = λ_max / λ_min.
        CG convergence rate depends on sqrt(κ).
        For κ = 1000: expect ~20 iterations to reach rtol=1e-6.
    """
    n = SMALL_MATRIX_SIZE
    rng = np.random.default_rng(test_seed)

    # Generate orthogonal matrix
    random_mat = rng.standard_normal((n, n))
    q, _ = np.linalg.qr(random_mat)

    # Create eigenvalues with large spread: [0.01, 0.02, ..., 0.09, 10]
    eigenvalues = np.concatenate([np.linspace(0.01, 0.09, n - 1), [10.0]])
    d = np.diag(eigenvalues)

    return q @ d @ q.T


# =============================================================================
# RHS Vector Fixtures
# =============================================================================


@pytest.fixture
def rhs_ones_small() -> NDArray:
    """RHS vector of all ones (size 10).

    Returns:
        10D vector of ones.

    Theory:
        b = [1, 1, ..., 1] is a common test case.
        For identity matrix: solution x = b (trivial).
        For diagonal D: solution x_i = 1/D_ii.
    """
    return np.ones(SMALL_MATRIX_SIZE, dtype=np.float64)


@pytest.fixture
def rhs_random_small(test_seed: int) -> NDArray:
    """Random RHS vector (size 10) with fixed seed.

    Args:
        test_seed: Random seed for reproducibility.

    Returns:
        10D random vector from standard normal distribution.
    """
    rng = np.random.default_rng(test_seed)
    return rng.standard_normal(SMALL_MATRIX_SIZE)


@pytest.fixture
def rhs_zero_small() -> NDArray:
    """Zero RHS vector (size 10) - trivial solution x=0.

    Returns:
        10D zero vector.

    Theory:
        For b=0, solution is x=0 regardless of matrix A.
        CG should converge immediately (iteration 0).
    """
    return np.zeros(SMALL_MATRIX_SIZE, dtype=np.float64)


# =============================================================================
# Test System Fixtures (Matrix + RHS with Known Solutions)
# =============================================================================


@pytest.fixture
def identity_system_known_solution(
    identity_matrix_small: NDArray,
    rhs_ones_small: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Identity matrix system with known solution x = b.

    Args:
        identity_matrix_small: Identity matrix.
        rhs_ones_small: RHS vector.

    Returns:
        Tuple of (A, b, x_exact) where A @ x_exact = b.

    Theory:
        For A = I, solution is x = b (trivial).
        CG converges in exactly 1 iteration.
    """
    a = identity_matrix_small
    b = rhs_ones_small
    x_exact = b.copy()
    return a, b, x_exact


@pytest.fixture
def tridiagonal_system_known_solution(
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Tridiagonal SPD system with known solution.

    Args:
        tridiagonal_spd_small: Tridiagonal matrix.
        rhs_ones_small: RHS vector.

    Returns:
        Tuple of (A, b, x_exact) where A @ x_exact = b.

    Theory:
        Solve A x = b exactly using direct solver.
        Provides ground truth for convergence testing.
    """
    a = tridiagonal_spd_small
    b = rhs_ones_small
    x_exact = np.linalg.solve(a, b)
    return a, b, x_exact


@pytest.fixture
def diagonal_system_known_solution(
    diagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Diagonal SPD system with known solution.

    Args:
        diagonal_spd_small: Diagonal matrix.
        rhs_ones_small: RHS vector.

    Returns:
        Tuple of (A, b, x_exact) where A @ x_exact = b.

    Theory:
        For diagonal D, solution is x_i = b_i / D_ii.
        CG converges in at most n iterations (often 1 with preconditioner).
    """
    a = diagonal_spd_small
    b = rhs_ones_small
    x_exact = b / np.diag(a)
    return a, b, x_exact


# =============================================================================
# Preconditioner Fixtures
# =============================================================================


@pytest.fixture
def identity_preconditioner() -> Callable[[NDArray, object], NDArray]:
    """Identity preconditioner M = I.

    Returns:
        Preconditioner function precond(r, ctx) -> z = r.

    Theory:
        Identity preconditioner: z = M^{-1} r = r.
        Equivalent to unpreconditioned CG.
    """

    def precond(r: NDArray, context: object) -> NDArray:
        """Apply identity preconditioner.

        Args:
            r: Residual vector.
            context: Iteration context (unused).

        Returns:
            z = r (no preconditioning).
        """
        return r.copy()

    return precond


@pytest.fixture
def jacobi_preconditioner_factory() -> Callable[
    [NDArray], Callable[[NDArray, object], NDArray]
]:
    """Factory for Jacobi preconditioner M = diag(A).

    Returns:
        Factory function that creates preconditioner from matrix A.

    Theory:
        Jacobi preconditioner: M = diag(A), so z_i = r_i / A_ii.
        Effective for diagonally dominant matrices.
        Cheap to compute and apply.
    """

    def factory(a: NDArray) -> Callable[[NDArray, object], NDArray]:
        """Create Jacobi preconditioner for matrix A.

        Args:
            a: System matrix.

        Returns:
            Preconditioner function.
        """
        diag_inv = 1.0 / np.diag(a)

        def precond(r: NDArray, context: object) -> NDArray:
            """Apply Jacobi preconditioner.

            Args:
                r: Residual vector.
                context: Iteration context (unused).

            Returns:
                z = diag(A)^{-1} r.
            """
            return diag_inv * r

        return precond

    return factory


@pytest.fixture
def jacobi_preconditioner_tridiagonal(
    tridiagonal_spd_small: NDArray,
    jacobi_preconditioner_factory: Callable[
        [NDArray], Callable[[NDArray, object], NDArray]
    ],
) -> Callable[[NDArray, object], NDArray]:
    """Jacobi preconditioner for tridiagonal test matrix.

    Args:
        tridiagonal_spd_small: Tridiagonal matrix.
        jacobi_preconditioner_factory: Factory for creating Jacobi preconditioner.

    Returns:
        Preconditioner function for tridiagonal matrix.
    """
    return jacobi_preconditioner_factory(tridiagonal_spd_small)


@pytest.fixture
def nan_preconditioner() -> Callable[[NDArray, object], NDArray]:
    """Broken preconditioner that returns NaN - for breakdown testing.

    Returns:
        Preconditioner function that returns NaN vector.

    Theory:
        Used to test breakdown detection in flexible PCG.
        Should trigger breakdown flag immediately.
    """

    def precond(r: NDArray, context: object) -> NDArray:
        """Return NaN vector to trigger breakdown.

        Args:
            r: Residual vector.
            context: Iteration context (unused).

        Returns:
            Vector of NaNs.
        """
        return np.full_like(r, np.nan)

    return precond


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def test_seed() -> int:
    """Random seed for reproducible tests.

    Returns:
        Fixed random seed value.
    """
    return TEST_SEED


@pytest.fixture
def default_tolerances() -> tuple[float, float]:
    """Default convergence tolerances (rtol, atol).

    Returns:
        Tuple of (rtol, atol) for convergence testing.
    """
    return DEFAULT_RTOL, DEFAULT_ATOL


@pytest.fixture
def default_assert_rtol() -> float:
    """Default relative tolerance for assertion checks.

    Returns:
        Default assertion rtol (typically 1e-5).
    """
    return DEFAULT_ASSERT_RTOL


@pytest.fixture
def tight_tolerances() -> tuple[float, float]:
    """Tight convergence tolerances for high-accuracy checks."""
    return 1e-14, 1e-14


@pytest.fixture
def default_max_iterations() -> int:
    """Default maximum iterations for solver.

    Returns:
        Maximum iteration count.
    """
    return DEFAULT_MAX_ITER


@pytest.fixture
def zero_initial_guess_small() -> NDArray:
    """Zero initial guess (size 10).

    Returns:
        10D zero vector for x0.

    Theory:
        x0 = 0 is the most common initial guess.
        Initial residual r0 = b - A*0 = b.
    """
    return np.zeros(SMALL_MATRIX_SIZE, dtype=np.float64)


# =============================================================================
# Plot Saving Fixtures
# =============================================================================


@pytest.fixture
def solver_plots_dir() -> Path:
    """Directory for solver convergence plots.

    Returns:
        Path to tests/artifacts/solver_plots/ directory.
    """
    plots_dir = Path(__file__).parent.parent / "artifacts" / "solver_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


@pytest.fixture
def save_convergence_plot(solver_plots_dir: Path) -> Callable:
    """Create function to save convergence plots.

    Args:
        solver_plots_dir: Directory for saving plots.

    Returns:
        Function that saves convergence plot: save_plot(name, history, out_dir=None)
    """
    def _save_plot(
        name: str,
        history: list[float] | np.ndarray,
        out_dir: Path | None = None,
    ) -> None:
        """Save convergence history plot.

        Args:
            name: Test name for plot title and filename.
            history: Residual history values.
            out_dir: Optional output directory (defaults to solver_plots_dir).
        """
        if not history or len(history) == 0:
            return

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.semilogy(history, marker="o")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual norm")
        ax.set_title(name)
        fig.tight_layout()

        output_dir = out_dir if out_dir is not None else solver_plots_dir
        output_path = output_dir / f"{name}.png"
        fig.savefig(output_path)
        plt.close(fig)

    return _save_plot
