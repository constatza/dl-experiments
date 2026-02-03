"""Benchmark: PCG-ours vs PCG-scipy iteration count equivalence.

Theoretical Basis:
    Our PCG implementation should match scipy.sparse.linalg.cg exactly.
    Both use the standard two-term recurrence PCG algorithm.

Expected Outcome:
    Exact match (diff = 0) for all test cases.
"""

import numpy as np
import pytest

from neuralls.solver.factories import preconditioned_cg, scipy_cg
from tests.benchmarks.exactness.conftest import (
    BENCHMARK_SIZES,
    MATRIX_TYPES,
    SOLUTION_COMPARISON_ATOL,
    SOLUTION_COMPARISON_RTOL,
    append_comparison_result,
    assert_solutions_match,
    generate_spd_system,
)


@pytest.mark.parametrize("size", BENCHMARK_SIZES)
@pytest.mark.parametrize("matrix_type", MATRIX_TYPES)
def test_pcg_scipy_exact_match(
    size: int,
    matrix_type: str,
    convergence_tolerances: tuple[float, float],
) -> None:
    """Test that PCG-ours and scipy-cg produce identical iteration counts.

    Args:
        size: System size (n × n)
        matrix_type: Matrix type ("tridiagonal", "diagonal")
        convergence_tolerances: Fixture providing (rtol, atol)

    Validates:
        - Both solvers converge
        - Iteration counts match exactly (diff = 0)
        - Solutions match to convergence tolerance
    """
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
        trace_mode="minimal",
    )

    # Run our PCG implementation
    x_pcg, result_pcg = preconditioned_cg(
        A,
        b,
        preconditioner=None,
        rtol=rtol,
        atol=atol,
        maxiter=1000,
        trace_mode="minimal",
    )

    # Check convergence
    assert result_scipy.converged, "Scipy-cg failed to converge"
    assert result_pcg.converged, "PCG-ours failed to converge"

    # Check iteration count exactness
    diff = result_pcg.iterations - result_scipy.iterations
    exact_match = diff == 0

    # Record result
    append_comparison_result(
        filename=f"pcg_scipy_{matrix_type}.md",
        title_prefix="PCG-ours vs PCG-scipy",
        headers=("Scipy Iters", "Ours Iters"),
        size=size,
        kappa=kappa,
        baseline_iters=result_scipy.iterations,
        test_iters=result_pcg.iterations,
        diff=diff,
        exact=exact_match,
    )

    # Assert exactness (0 tolerance)
    assert exact_match, (
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
