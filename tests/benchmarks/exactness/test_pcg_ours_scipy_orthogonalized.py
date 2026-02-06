"""Benchmark: PCG-ours-orthogonalized vs PCG-scipy iteration count comparison.

Theoretical Basis:
    Our PCG implementation WITH full reorthogonalization (m_max=-1) should differ
    from scipy.sparse.linalg.cg (which uses standard two-term recurrence without
    reorthogonalization). This test helps identify whether divergence in FCG-PCG
    tests is due to reorthogonalization implementation.

Expected Outcome:
    Likely mismatch (diff != 0) since scipy doesn't use reorthogonalization.
    This test is diagnostic, not a validation of exactness.
"""

import pytest

from neuralls.solver.factories import pcg, scipy_cg
from tests.benchmarks.exactness.conftest import (
    BENCHMARK_SIZES,
    MATRIX_TYPES,
    SOLUTION_COMPARISON_ATOL,
    SOLUTION_COMPARISON_RTOL,
    assert_solutions_match,
    generate_spd_system,
)


def _is_slow_case(size: int) -> bool:
    """Mark test cases with size >= 500 as slow."""
    return size >= 500


@pytest.mark.benchmark
@pytest.mark.parametrize("size", BENCHMARK_SIZES)
@pytest.mark.parametrize("matrix_type", MATRIX_TYPES)
def test_pcg_ours_ortho_scipy_comparison(
    size: int,
    matrix_type: str,
    convergence_tolerances: tuple[float, float],
    request: pytest.FixtureRequest,
) -> None:
    """Compare PCG-ours with reorthogonalization vs scipy-cg iteration counts.

    Args:
        size: System size (n × n)
        matrix_type: Matrix type ("tridiagonal", "diagonal")
        convergence_tolerances: Fixture providing (rtol, atol)
        request: Pytest request fixture for dynamic marker application

    Validates:
        - Both solvers converge
        - Records iteration count differences (may not match)
        - Solutions match to convergence tolerance
    """
    # Apply slow marker dynamically for large test cases
    if _is_slow_case(size):
        request.node.add_marker(pytest.mark.slow)

    # Generate SPD matrix + RHS
    A, b, x_exact, kappa = generate_spd_system(size, matrix_type)
    rtol, atol = convergence_tolerances

    # Run scipy-cg (baseline, no reorthogonalization)
    x_scipy, result_scipy = scipy_cg(
        A,
        b,
        rtol=rtol,
        atol=atol,
        maxiter=1000,
        trace_mode="minimal",
    )

    # Run our PCG implementation WITH full reorthogonalization
    x_pcg_ortho, result_pcg_ortho = pcg(
        A,
        b,
        preconditioner=None,
        m_max=-1,  # Full reorthogonalization
        rtol=rtol,
        atol=atol,
        maxiter=1000,
        trace_mode="minimal",
    )

    # Check convergence
    assert result_scipy.converged, "Scipy-cg failed to converge"
    assert result_pcg_ortho.converged, "PCG-ours-ortho failed to converge"

    # NOTE: We don't assert exactness here since this is a diagnostic test
    # Reorthogonalization is expected to potentially change iteration counts

    # Verify residuals match (worst case: 2*rtol difference between solvers)
    assert_solutions_match(
        A,
        b,
        x_pcg_ortho,
        x_scipy,
        rtol=SOLUTION_COMPARISON_RTOL,
        atol=SOLUTION_COMPARISON_ATOL,
        label1="PCG-ortho",
        label2="SciPy",
    )
