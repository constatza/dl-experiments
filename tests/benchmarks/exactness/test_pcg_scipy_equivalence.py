"""Benchmark: PCG-ours vs PCG-scipy iteration count equivalence.

Theoretical Basis:
    Our PCG implementation should match scipy.sparse.linalg.cg exactly.
    Both use the standard two-term recurrence PCG algorithm.

Expected Outcome:
    Exact match (diff = 0) for all test cases.
"""

import pytest

from neuralls.solver.factories import pcg, scipy_cg
from neuralls.solver.monitoring.trace_mode import TraceMode
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
def test_pcg_scipy_exact_match(
    size: int,
    matrix_type: str,
    convergence_tolerances: tuple[float, float],
    request: pytest.FixtureRequest,
) -> None:
    """Test that PCG-ours and scipy-cg produce identical iteration counts.

    Args:
        size: System size (n × n)
        matrix_type: Matrix type ("tridiagonal", "diagonal")
        convergence_tolerances: Fixture providing (rtol, atol)
        request: Pytest request fixture for dynamic marker application

    Validates:
        - Both solvers converge
        - Iteration counts match exactly (diff = 0)
        - Solutions match to convergence tolerance
    """
    # Apply slow marker dynamically for large test cases
    if _is_slow_case(size):
        request.node.add_marker(pytest.mark.slow)

    # Generate SPD matrix + RHS
    A, b, x_exact, kappa = generate_spd_system(size, matrix_type)
    rtol, atol = convergence_tolerances

    # Run scipy-cg (baseline)
    x_scipy, result_scipy = scipy_cg(
        A,
        b,
        rtol=rtol,
        atol=atol,
        maxiter=1000,
        trace_mode=TraceMode.MINIMAL,
    )

    # Run our PCG implementation
    x_pcg, result_pcg = pcg(
        A,
        b,
        preconditioner=None,
        rtol=rtol,
        atol=atol,
        maxiter=1000,
        trace_mode=TraceMode.MINIMAL,
    )

    # Check convergence
    assert result_scipy.converged, "Scipy-cg failed to converge"
    assert result_pcg.converged, "PCG-ours failed to converge"

    # Assert iteration count exactness (0 tolerance)
    diff = result_pcg.iterations - result_scipy.iterations
    assert diff == 0, (
        f"PCG-ours and scipy-cg iteration counts must match exactly. "
        f"Scipy={result_scipy.iterations}, Ours={result_pcg.iterations}, diff={diff}"
    )

    # Verify residuals match (worst case: 2*rtol difference between solvers)
    assert_solutions_match(
        A,
        b,
        x_pcg,
        x_scipy,
        rtol=SOLUTION_COMPARISON_RTOL,
        atol=SOLUTION_COMPARISON_ATOL,
        label1="PCG",
        label2="SciPy",
    )
