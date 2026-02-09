"""Fixtures for from-file benchmarks."""

import os

import numpy as np
import pytest
from neuralls.io.base import load_matrix
from scipy import linalg


# List of files to benchmark.
# Default is empty, user must supply valid paths either here or via env var.
MATRIX_PATHS = (r"/data/projects/graph-cg/data/raw/SpectralData/benchmarks/Ktilde.txt",)
RHS_PATHS = (r"/data/projects/graph-cg/data/raw/SpectralData/benchmarks/ftilde.txt",)
L_PATH = r"/data/projects/graph-cg/data/raw/SpectralData/benchmarks/L.txt"

# Allow overriding/extending via env var BENCHMARK_MATRIX_PATHS
if "BENCHMARK_MATRIX_PATHS" in os.environ:
    paths = os.environ["BENCHMARK_MATRIX_PATHS"].split(",")


# Use [None] if empty to ensure at least one test is collected and then skipped with a message
@pytest.fixture(params=MATRIX_PATHS if MATRIX_PATHS else [None])
def matrix_path(request):
    """Fixture providing matrix file path.

    Skips test if no path provided or if file does not exist.
    """
    path = request.param
    check_existence(path)
    return path


@pytest.fixture(params=RHS_PATHS if RHS_PATHS else [None])
def rhs_path(request):
    """Fixture providing right-hand side file path.

    Skips test if no path provided or if file does not exist.
    """
    path = request.param
    check_existence(path)
    return path


@pytest.fixture
def system(matrix_path, rhs_path):
    """Load matrix and construct exact solution.

    Returns:
        tuple: (A, b, x_exact, L) where x_exact is computed solution.
    """
    A = load_matrix(matrix_path, delimiter=",")
    L = load_matrix(L_PATH, delimiter=",")
    # Zero out near-zero entries using centralized threshold
    A[np.absolute(A) <= SPARSITY_THRESHOLD] = 0
    n = A.shape[0]
    b = load_matrix(rhs_path, delimiter=",")
    x_exact = linalg.solve(A, b)
    return A, b, x_exact, L


@pytest.fixture
def convergence_tolerances():
    """Tolerances for solver convergence check."""
    return CONVERGENCE_RTOL, CONVERGENCE_ATOL


# Sparsity threshold for determining non-zero entries
SPARSITY_THRESHOLD = 1e-19  # Entries with |value| <= this are considered zero

# Convergence tolerances for CG solvers
CONVERGENCE_RTOL = 1e-12  # Relative tolerance for convergence
CONVERGENCE_ATOL = 1e-14  # Absolute tolerance for convergence

# IC0 comparison tolerances
IC0_MATRIX_RTOL = 1e-12  # Relative tolerance for L matrix comparison
IC0_MATRIX_ATOL = 1e-14  # Absolute tolerance for L matrix comparison
IC0_SPARSITY_COUNT_TOLERANCE = (
    5  # Allow up to 5 non-zero count difference due to numerical precision
)
IC0_SOLUTION_RTOL = 1e-10  # Relative tolerance for solution comparison
IC0_SOLUTION_ATOL = 1e-12  # Absolute tolerance for solution comparison
IC0_THRESHOLD = 0.0  # Use no thresholding to match reference
# Note: Integer metrics (sparsity counts, iteration counts) use exact equality (zero tolerance)


def check_existence(path):
    if not os.path.exists(path) or path is None:
        pytest.skip(f"Matrix file not found: {path}")
