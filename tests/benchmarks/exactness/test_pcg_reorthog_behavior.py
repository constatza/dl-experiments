"""Benchmark: PCG with/without explicit reorthogonalization.

This test documents the behavior of PCG when m_max=-1 is used.
With explicit reorthogonalization, PCG applies additional orthogonalization
on top of the two-term recurrence, which may or may not improve convergence
depending on the problem conditioning.

This is a diagnostic test - it documents behavior without asserting specific
iteration count relationships.
"""

import pytest

from neuralls.solver.factories import preconditioned_cg
from tests.benchmarks.exactness.conftest import (
    BENCHMARK_SIZES,
    MATRIX_TYPES,
    SOLUTION_COMPARISON_ATOL,
    SOLUTION_COMPARISON_RTOL,
    assert_solutions_match,
    generate_spd_system,
)


@pytest.mark.parametrize("size", BENCHMARK_SIZES)
@pytest.mark.parametrize("matrix_type", MATRIX_TYPES)
def test_pcg_reorthog_effect(
    size: int,
    matrix_type: str,
    convergence_tolerances: tuple[float, float],
) -> None:
    """Compare PCG with and without explicit reorthogonalization.

    Args:
        size: System size (n × n)
        matrix_type: Matrix type ("tridiagonal", "diagonal")
        convergence_tolerances: Fixture providing (rtol, atol)

    Validates:
        - Both variants converge
        - Solutions match to convergence tolerance

    Notes:
        This is a diagnostic test. For well-conditioned problems,
        reorthogonalization has minimal effect. For ill-conditioned
        problems, it may help or hurt depending on rounding errors.
    """
    # Generate SPD matrix + RHS
    A, b, x_exact, kappa = generate_spd_system(size, matrix_type)
    rtol, atol = convergence_tolerances

    # Run standard PCG (no reorthogonalization)
    x_pcg_std, result_pcg_std = preconditioned_cg(
        A,
        b,
        preconditioner=None,
        rtol=rtol,
        atol=atol,
        maxiter=1000,
    )

    # Run PCG with full reorthogonalization
    x_pcg_ortho, result_pcg_ortho = preconditioned_cg(
        A,
        b,
        preconditioner=None,
        m_max=-1,  # Full reorthogonalization
        rtol=rtol,
        atol=atol,
        maxiter=1000,
    )

    # Both should converge
    assert result_pcg_std.converged, "Standard PCG failed to converge"
    assert result_pcg_ortho.converged, "PCG with reorthogonalization failed to converge"

    # Residuals should match (worst case: 2*rtol difference between solvers)
    assert_solutions_match(
        A,
        b,
        x_pcg_std,
        x_pcg_ortho,
        rtol=SOLUTION_COMPARISON_RTOL,
        atol=SOLUTION_COMPARISON_ATOL,
        label1="PCG-standard",
        label2="PCG-ortho",
    )

    # Print diagnostic info (no assertions on iteration counts)
    diff = result_pcg_ortho.iterations - result_pcg_std.iterations
    print(
        f"\n{matrix_type} (n={size}, κ={kappa:.0f}): "
        f"PCG={result_pcg_std.iterations}, "
        f"PCG+ortho={result_pcg_ortho.iterations}, "
        f"diff={diff:+d}"
    )
