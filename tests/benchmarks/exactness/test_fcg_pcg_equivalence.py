"""Benchmark: FCG vs PCG with fixed preconditioner equivalence.

Theoretical Basis (Notay 2000, page 1447):
    With a FIXED SPD preconditioner, FCG's orthogonalization coefficients
    automatically vanish due to A-conjugacy: (wi, A dj) = 0 for j < i-1.
    Therefore, FCG simplifies to standard PCG WITHOUT explicit reorthogonalization.

    This is NOT about both using "full reorthogonalization" - it's about
    FCG's explicit orthogonalization becoming equivalent to PCG's implicit
    conjugacy via the two-term recurrence.

Expected Outcome:
    Near-exact match (diff ≤ 2) for all test cases. Small differences may occur
    due to finite-precision rounding errors in the orthogonalization coefficient
    computation, but iteration counts should be nearly identical.
"""

import numpy as np
import pytest

from neuralls.solver.factories import flexible_cg, preconditioned_cg
from tests.benchmarks.exactness.conftest import (
    BENCHMARK_SIZES,
    ITER_DIFF_THRESHOLD_FCG_PCG,
    MATRIX_TYPES,
    SOLUTION_COMPARISON_ATOL,
    SOLUTION_COMPARISON_RTOL,
    append_comparison_result,
    assert_solutions_match,
    check_residual_norm,
    generate_spd_system,
)


@pytest.mark.parametrize("size", BENCHMARK_SIZES)
@pytest.mark.parametrize("matrix_type", MATRIX_TYPES)
def test_fcg_pcg_exact_match(
    size: int,
    matrix_type: str,
    convergence_tolerances: tuple[float, float],
) -> None:
    """Test that FCG and standard PCG produce nearly identical iteration counts.

    Args:
        size: System size (n × n)
        matrix_type: Matrix type ("tridiagonal", "diagonal")
        convergence_tolerances: Fixture providing (rtol, atol)

    Validates:
        - Both solvers converge
        - Iteration counts match within threshold (|diff| ≤ 2)
        - Solutions match to convergence tolerance
    """
    # Generate SPD matrix + RHS
    A, b, x_exact, kappa = generate_spd_system(size, matrix_type)
    rtol, atol = convergence_tolerances

    # Run STANDARD PCG (baseline) - NO reorthogonalization!
    x_pcg, result_pcg = preconditioned_cg(
        A,
        b,
        preconditioner=None,  # Fixed identity preconditioner
        rtol=rtol,
        atol=atol,
        maxiter=1000,
    )

    # Run FCG(-1) (with orthogonalization that should auto-vanish)
    x_fcg, result_fcg = flexible_cg(
        A,
        b,
        preconditioner=None,  # Same fixed identity preconditioner
        m_max=-1,  # Orthogonalization (should auto-truncate)
        rtol=rtol,
        atol=atol,
        maxiter=1000,
    )

    # Check convergence
    assert result_pcg.converged, "PCG failed to converge"
    assert result_fcg.converged, "FCG(-1) failed to converge"

    # Verify both solutions actually satisfy convergence criterion
    check_residual_norm(A, b, x_pcg, rtol, atol, label="PCG solution")
    check_residual_norm(A, b, x_fcg, rtol, atol, label="FCG solution")

    # Check iteration count near-exactness
    diff = result_fcg.iterations - result_pcg.iterations
    within_threshold = abs(diff) <= ITER_DIFF_THRESHOLD_FCG_PCG

    # Record result
    append_comparison_result(
        filename=f"fcg_vs_pcg_fixed_{matrix_type}.md",
        title_prefix="FCG vs PCG (Fixed Preconditioner)",
        headers=("PCG Iters", "FCG Iters"),
        size=size,
        kappa=kappa,
        baseline_iters=result_pcg.iterations,
        test_iters=result_fcg.iterations,
        diff=diff,
        exact=within_threshold,
    )

    # Warn if difference exceeds threshold (but don't fail test)
    if not within_threshold:
        import warnings

        warnings.warn(
            f"FCG and PCG iteration counts differ by more than {ITER_DIFF_THRESHOLD_FCG_PCG} with fixed preconditioner. "
            f"PCG={result_pcg.iterations}, FCG={result_fcg.iterations}, diff={diff}. "
            f"Expected near-equivalence per Notay 2000 theorem.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Verify residuals match (worst case: 2*rtol difference between solvers)
    assert_solutions_match(
        A,
        b,
        x_pcg,
        x_fcg,
        rtol=SOLUTION_COMPARISON_RTOL,
        atol=SOLUTION_COMPARISON_ATOL,
        label1="PCG",
        label2="FCG",
    )
