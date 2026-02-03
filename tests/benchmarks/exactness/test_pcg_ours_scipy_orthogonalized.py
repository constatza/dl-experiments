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
def test_pcg_ours_ortho_scipy_comparison(
    size: int,
    matrix_type: str,
    convergence_tolerances: tuple[float, float],
) -> None:
    """Compare PCG-ours with reorthogonalization vs scipy-cg iteration counts.

    Args:
        size: System size (n × n)
        matrix_type: Matrix type ("tridiagonal", "diagonal")
        convergence_tolerances: Fixture providing (rtol, atol)

    Validates:
        - Both solvers converge
        - Records iteration count differences (may not match)
        - Solutions match to convergence tolerance
    """
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
    x_pcg_ortho, result_pcg_ortho = preconditioned_cg(
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

    # Check iteration count difference
    diff = result_pcg_ortho.iterations - result_scipy.iterations
    exact_match = diff == 0

    # Record result
    append_comparison_result(
        filename=f"pcg_ours_ortho_scipy_{matrix_type}.md",
        title_prefix="PCG-ours-ortho vs PCG-scipy",
        headers=("Scipy Iters", "Ours-Ortho Iters"),
        size=size,
        kappa=kappa,
        baseline_iters=result_scipy.iterations,
        test_iters=result_pcg_ortho.iterations,
        diff=diff,
        exact=exact_match,
    )

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
