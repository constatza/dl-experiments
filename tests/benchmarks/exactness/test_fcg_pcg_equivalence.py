"""Benchmark: FCG vs PCG with fixed preconditioner equivalence.

Theoretical Basis (Notay 2000, page 1447):
    With a FIXED SPD preconditioner, FCG's orthogonalization coefficients
    automatically vanish due to A-conjugacy: (wi, A dj) = 0 for j < i-1.
    Therefore, FCG simplifies to standard PCG WITHOUT explicit reorthogonalization.

Expected Outcome:
    Near-exact match (diff ≤ 2) for all test cases.
"""

import numpy as np
import pytest

from neuralls.domain.solver.factories import flexible_cg, pcg
from neuralls.domain.solver.monitoring.trace_mode import TraceMode
from neuralls.domain.solver.preconditioners import ILUPreconditioner
from tests.benchmarks.exactness.conftest import (
    ITER_DIFF_THRESHOLD_FCG_PCG,
    SOLUTION_COMPARISON_ATOL,
    SOLUTION_COMPARISON_RTOL,
    assert_solutions_match,
    check_residual_norm,
    generate_spd_system,
)


# Test parameters - 8 cases total
# 2 cases with trace export: size 100, tridiagonal × precond [None, ilu]
# 6 cases without trace export: sizes [200,500,1000] × matrix_types [tridiagonal,diagonal]
TEST_CASES = [
    # Export cases (size=100, tridiagonal only) - with and without ILU
    (100, "tridiagonal", None, True),  # Export FCG traces
    (100, "tridiagonal", "ilu", True),  # Export FCG traces
    # Standard cases (larger sizes, no trace export)
    (200, "tridiagonal", None, False),
    (200, "diagonal", None, False),
    (500, "tridiagonal", None, False),
    (500, "diagonal", None, False),
    (1000, "tridiagonal", None, False),
    (1000, "diagonal", None, False),
]


def _is_slow_case(size: int) -> bool:
    """Mark test cases with size >= 500 as slow."""
    return size >= 500


@pytest.mark.benchmark
@pytest.mark.parametrize("size,matrix_type,precond_type,export_traces", TEST_CASES)
def test_fcg_pcg_equivalence(
    size: int,
    matrix_type: str,
    precond_type: str | None,
    export_traces: bool,
    convergence_tolerances: tuple[float, float],
    request: pytest.FixtureRequest,
) -> None:
    """Test FCG vs PCG equivalence with fixed preconditioner.

    Tests that FCG and PCG produce identical iteration counts with fixed preconditioners
    (Notay 2000 theorem). For specific cases (size=100, tridiagonal), also exports full
    FCG traces for offline analysis.

    Args:
        size: System size (n × n)
        matrix_type: Matrix type ("tridiagonal", "diagonal")
        precond_type: Preconditioner type (None for identity, "ilu")
        export_traces: Whether to export FCG traces for this case
        convergence_tolerances: Fixture providing (rtol, atol)
        request: Pytest request fixture for dynamic marker application

    Validates:
        - Both solvers converge
        - Iteration counts match within threshold (|diff| ≤ 2)
        - Solutions match to convergence tolerance
        - FCG traces exported when export_traces=True
    """
    # Apply slow marker dynamically for large test cases
    if _is_slow_case(size):
        request.node.add_marker(pytest.mark.slow)

    # Generate SPD matrix + RHS
    A, b, x_exact, kappa = generate_spd_system(size, matrix_type)
    x0 = np.zeros_like(b)
    rtol, atol = convergence_tolerances

    # Build preconditioner if specified
    preconditioner = None
    if precond_type == "ilu":
        preconditioner = ILUPreconditioner(A)

    # Choose trace mode based on whether we're exporting
    trace_mode = TraceMode.FULL if export_traces else TraceMode.MINIMAL

    # Run STANDARD PCG
    x_pcg, result_pcg = pcg(
        A,
        b,
        x0=x0,
        preconditioner=preconditioner,
        trace_mode=TraceMode.MINIMAL,  # PCG always minimal (we don't export PCG traces)
        rtol=rtol,
        atol=atol,
        maxiter=size * 2,
    )

    # Run FCG with orthogonalization
    x_fcg, result_fcg = flexible_cg(
        A,
        b,
        x0=x0,
        preconditioner=preconditioner,
        m_max=-1,  # Full orthogonalization
        trace_mode=trace_mode,  # Full trace only when exporting
        rtol=rtol,
        atol=atol,
        maxiter=size * 2,
    )

    # Check convergence
    precond_label = precond_type or "identity"
    assert result_pcg.converged, f"PCG with {precond_label} preconditioner failed to converge"
    assert result_fcg.converged, f"FCG with {precond_label} preconditioner failed to converge"

    # Verify both solutions actually satisfy convergence criterion
    check_residual_norm(A, b, x_pcg, rtol, atol, label="PCG solution")
    check_residual_norm(A, b, x_fcg, rtol, atol, label="FCG solution")

    # Check iteration count near-exactness
    diff = result_fcg.iterations - result_pcg.iterations
    within_threshold = abs(diff) <= ITER_DIFF_THRESHOLD_FCG_PCG

    # Assert iterations match within threshold
    assert within_threshold, (
        f"FCG and PCG iteration counts differ by more than {ITER_DIFF_THRESHOLD_FCG_PCG} "
        f"with {precond_label} preconditioner. "
        f"PCG={result_pcg.iterations}, FCG={result_fcg.iterations}, diff={diff}. "
        f"Expected near-equivalence per Notay 2000 theorem."
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
