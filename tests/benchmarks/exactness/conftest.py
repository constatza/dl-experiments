"""Fixtures and utilities for CG iteration count exactness benchmarks."""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from neuralls.solver.strategies.convergence import CombinedToleranceCriterion


# System sizes for benchmarks
BENCHMARK_SIZES = [100, 200, 500, 1000]

# Matrix types
MATRIX_TYPES = ["tridiagonal", "diagonal"]

# Single source of truth for all tolerances
EXACTNESS_RTOL = 1e-12
EXACTNESS_ATOL = 1e-14

# Solution comparison tolerances
# Worst case: if both solvers converge within rtol of exact solution,
# they can differ by up to 2*rtol from each other
SOLUTION_COMPARISON_RTOL = 2 * EXACTNESS_RTOL
SOLUTION_COMPARISON_ATOL = 2 * EXACTNESS_ATOL

# Iteration count difference thresholds
ITER_DIFF_THRESHOLD_FCG_PCG = 2  # Allow up to 2 iterations difference for FCG vs PCG

# Random seeds
SEED_MATRIX = 42
SEED_RHS = 123


def _create_tridiagonal_spd(n: int) -> NDArray:
    """Create tridiagonal SPD matrix.

    Args:
        n: Matrix size

    Returns:
        Tridiagonal SPD matrix with diag=4, off-diag=-1
    """
    from scipy.sparse import diags

    sparse_mat = diags(
        [-1.0, 4.0, -1.0],
        [-1, 0, 1],
        shape=(n, n),
        dtype=np.float64,
    )
    return sparse_mat.toarray()


def _estimate_tridiagonal_kappa(n: int) -> float:
    """Estimate condition number for tridiagonal SPD matrix.

    Args:
        n: Matrix size

    Returns:
        Estimated condition number (κ ≈ 4n²/π²)
    """
    return 4 * n**2 / (np.pi**2)


def _create_random_spd(
    n: int,
    kappa: float,
    rng: np.random.Generator,
) -> tuple[NDArray, float]:
    """Create random SPD matrix with target condition number.

    Args:
        n: Matrix size
        kappa: Target condition number
        rng: Random number generator

    Returns:
        Tuple of (A, actual_kappa) where A is SPD matrix
    """
    # Generate orthogonal matrix via QR
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))

    # Generate eigenvalues with desired condition number
    lambda_min = 1.0
    lambda_max = kappa * lambda_min
    eigenvalues = np.linspace(lambda_min, lambda_max, n)

    # Construct A = Q @ diag(λ) @ Q.T
    A = Q @ np.diag(eigenvalues) @ Q.T

    # Compute actual condition number
    actual_kappa = np.linalg.cond(A)

    return A, actual_kappa


def generate_spd_system(
    size: int,
    matrix_type: str,
    seed_matrix: int = SEED_MATRIX,
    seed_rhs: int = SEED_RHS,
) -> tuple[NDArray, NDArray, NDArray, float]:
    """Generate SPD system with known solution.

    Args:
        size: Matrix size (n × n)
        matrix_type: One of "tridiagonal", "diagonal", "random"
        seed_matrix: Random seed for matrix generation
        seed_rhs: Random seed for RHS generation

    Returns:
        Tuple of (A, b, x_exact, kappa) where:
            - A: SPD matrix (size × size)
            - b: RHS vector
            - x_exact: Exact solution
            - kappa: Condition number
    """
    rng_mat = np.random.default_rng(seed_matrix)
    rng_rhs = np.random.default_rng(seed_rhs)

    if matrix_type == "tridiagonal":
        # Tridiagonal: diag=4, off-diag=-1
        # κ ≈ 4n²/π² for large n
        A = _create_tridiagonal_spd(size)
        kappa = _estimate_tridiagonal_kappa(size)

    elif matrix_type == "diagonal":
        # Diagonal with entries [1, 2, ..., size]
        # κ = size / 1 = size
        A = np.diag(np.arange(1, size + 1, dtype=np.float64))
        kappa = float(size)

    elif matrix_type == "random":
        # Random SPD with controlled condition number
        # Use QR to get orthogonal Q, then Q @ D @ Q.T
        kappa_target = 100.0  # Well-conditioned
        A, kappa = _create_random_spd(size, kappa_target, rng_mat)

    else:
        raise ValueError(f"Unknown matrix_type: {matrix_type}")

    # Generate exact solution and RHS
    x_exact = rng_rhs.standard_normal(size)
    b = A @ x_exact

    return A, b, x_exact, kappa


@pytest.fixture(scope="session", autouse=True)
def clear_results_before_tests() -> None:
    """Clear all result markdown files before test session starts."""
    results_dir = Path(__file__).parent / "results"
    if results_dir.exists():
        for result_file in results_dir.glob("*.md"):
            result_file.unlink()
    yield


@pytest.fixture
def convergence_tolerances() -> tuple[float, float]:
    """High-precision convergence tolerances for exactness benchmarks.

    Returns:
        Tuple of (rtol, atol) for solver convergence
    """
    return EXACTNESS_RTOL, EXACTNESS_ATOL


def check_residual_norm(
    A: NDArray,
    b: NDArray,
    x: NDArray,
    rtol: float,
    atol: float,
    label: str = "solution",
) -> None:
    """Check that solution satisfies convergence criterion.

    Args:
        A: System matrix
        b: RHS vector
        x: Solution vector
        rtol: Relative tolerance
        atol: Absolute tolerance
        label: Label for error message

    Raises:
        AssertionError: If solution doesn't satisfy convergence criterion
    """
    residual = b - A @ x
    residual_norm = np.linalg.norm(residual)
    b_norm = np.linalg.norm(b)
    threshold = max(rtol * b_norm, atol)

    if residual_norm > threshold:
        raise AssertionError(
            f"{label} doesn't satisfy convergence criterion: "
            f"||r||={residual_norm:.2e} > threshold={threshold:.2e} "
            f"(max({rtol:.1e} * ||b||, {atol:.1e}))"
        )


def check_residuals_match(
    A: NDArray,
    b: NDArray,
    x1: NDArray,
    x2: NDArray,
    rtol: float,
    atol: float,
) -> tuple[bool, dict[str, float]]:
    """Check if residuals of two solutions match using convergence criterion.

    Compares residual norms using the combined tolerance criterion:
    ||r1 - r2|| <= max(rtol * ||b||, atol)

    This reuses CombinedToleranceCriterion to avoid duplicate logic.

    Args:
        A: System matrix
        b: RHS vector
        x1: First solution vector
        x2: Second solution vector
        rtol: Relative tolerance (already doubled at call site)
        atol: Absolute tolerance (already doubled at call site)

    Returns:
        Tuple of (matches, diagnostics) where:
            - matches: True if residuals match within tolerance
            - diagnostics: Dict with residual norms and differences
    """
    # Create criterion with provided tolerances (already doubled at call site)
    criterion = CombinedToleranceCriterion(rtol=rtol, atol=atol)

    # Compute residual difference efficiently: r1 - r2 = A(x1 - x2)
    r_diff = A @ (x1 - x2)

    # Check convergence using existing criterion
    b_norm = np.linalg.norm(b)
    matches = criterion.has_converged(r_diff, b_norm)

    # Diagnostics (compute r1, r2 only for reporting)
    r1 = b - A @ x1
    r2 = b - A @ x2
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    diff_norm = np.linalg.norm(r_diff)

    diagnostics = {
        "r1_norm": r1_norm,
        "r2_norm": r2_norm,
        "r1_rel": r1_norm / b_norm,
        "r2_rel": r2_norm / b_norm,
        "diff_norm": diff_norm,
        "b_norm": b_norm,
        "threshold": criterion.threshold(b_norm),
    }

    return matches, diagnostics


def assert_solutions_match(
    A: NDArray,
    b: NDArray,
    x1: NDArray,
    x2: NDArray,
    rtol: float,
    atol: float,
    label1: str = "x1",
    label2: str = "x2",
) -> None:
    """Assert residuals of two solutions match using convergence criterion.

    Compares residual norms using the combined tolerance criterion:
    ||r1 - r2|| <= max(rtol * ||b||, atol)

    Args:
        A: System matrix
        b: RHS vector
        x1: First solution vector
        x2: Second solution vector
        rtol: Relative tolerance (already doubled at call site)
        atol: Absolute tolerance (already doubled at call site)
        label1: Name/description of first solution (for error message)
        label2: Name/description of second solution (for error message)

    Raises:
        AssertionError: If residuals don't match within tolerance
    """
    matches, diag = check_residuals_match(A, b, x1, x2, rtol, atol)

    if not matches:
        raise AssertionError(
            f"Residuals for {label1} and {label2} don't match within tolerance.\n"
            f"  ||r_{label1}|| = {diag['r1_norm']:.3e} "
            f"(relative: {diag['r1_rel']:.3e})\n"
            f"  ||r_{label2}|| = {diag['r2_norm']:.3e} "
            f"(relative: {diag['r2_rel']:.3e})\n"
            f"  ||r_{label1} - r_{label2}|| = {diag['diff_norm']:.3e}\n"
            f"  ||b|| = {diag['b_norm']:.3e}\n"
            f"  Threshold = max({rtol:.1e} * ||b||, {atol:.1e}) = "
            f"{diag['threshold']:.3e}\n"
            f"  Tolerances: rtol={rtol:.1e}, atol={atol:.1e}"
        )


def append_comparison_result(
    filename: str,
    title_prefix: str,
    headers: tuple[str, str],
    size: int,
    kappa: float,
    baseline_iters: int,
    test_iters: int,
    diff: int,
    exact: bool,
) -> None:
    """Append comparison result to markdown file.

    Args:
        filename: Output markdown filename
        title_prefix: Prefix for title (e.g., "FCG(-1) vs PCG")
        headers: Tuple of (baseline_header, test_header) for table
        size: System size
        kappa: Condition number
        baseline_iters: Baseline solver iteration count
        test_iters: Test solver iteration count
        diff: Iteration count difference
        exact: Whether counts match exactly
    """
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    filepath = results_dir / filename

    # Status can mean either "exact" or "within threshold" depending on test
    status = "✓ EXACT" if exact else "✗ MISMATCH"
    baseline_header, test_header = headers
    row = (
        f"| {size} | {kappa:.0f} | {baseline_iters} | {test_iters} | "
        f"{diff:+d} | {status} |\n"
    )

    # First write creates file with header
    if not filepath.exists():
        matrix_name = (
            Path(filename)
            .stem.replace("fcg_vs_pcg_fixed_", "")
            .replace("fcg_pcg_", "")
            .replace("pcg_scipy_", "")
            .replace("pcg_ours_ortho_scipy_", "")
            .replace("_", " ")
            .title()
        )
        table_header = (
            f"| Size | κ | {baseline_header} | {test_header} | Diff | Status |\n"
            f"|------|---|{'-' * len(baseline_header)}|{'-' * len(test_header)}|------|--------|\n"
        )
        with filepath.open("w") as f:
            f.write(f"# {title_prefix}: {matrix_name}\n\n")
            f.write(table_header)
            f.write(row)
    else:
        with filepath.open("a") as f:
            f.write(row)
