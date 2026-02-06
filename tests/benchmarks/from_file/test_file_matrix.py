"""Benchmark solver on user-provided matrix files."""

import pytest
import numpy as np
from neuralls.solver.factories import flexible_cg, scipy_cg
from neuralls.solver.preconditioners import ICholeskyPreconditioner
from neuralls.solver.monitoring.trace_mode import TraceMode

# FCG and PCG must match exactly on iteration counts
ITER_DIFF_THRESHOLD = 0


@pytest.mark.benchmark
@pytest.mark.parametrize("precond_type", [None, "ilu"])
def test_fcg_pcg_equivalence_on_file(
    system, convergence_tolerances, precond_type, matrix_path
):
    """Test FCG vs PCG equivalence on matrix loaded from file.

    Args:
        system: Tuple (A, b, x_exact) from fixture.
        convergence_tolerances: Tuple (rtol, atol).
        precond_type: Type of preconditioner to use (None or "ilu").
        matrix_path: Path to the matrix file (from fixture).
    """
    A, b, x_exact, L = system
    rtol, atol = convergence_tolerances
    n = A.shape[0]

    # Build preconditioner
    preconditioner = None
    if precond_type == "ilu":
        preconditioner = ICholeskyPreconditioner(L)

    # Run PCG with FULL trace for export
    x_pcg, result_pcg = scipy_cg(
        A,
        b,
        preconditioner=preconditioner,
        rtol=rtol,
        atol=atol,
        maxiter=10000,
        trace_mode=TraceMode.FULL,
    )

    # Run FCG with FULL trace for export
    x_fcg, result_fcg = flexible_cg(
        A,
        b,
        preconditioner=preconditioner,
        m_max=1,  # No orthogonalization (should match PCG exactly)
        rtol=rtol,
        atol=atol,
        maxiter=10000,
        trace_mode=TraceMode.FULL,
    )

    assert result_pcg.converged, (
        f"PCG failed to converge. Iterations: {result_pcg.iterations}"
    )
    assert result_fcg.converged, (
        f"FCG failed to converge. Iterations: {result_fcg.iterations}"
    )

    # Check iteration difference
    diff = abs(result_pcg.iterations - result_fcg.iterations)
    assert diff <= ITER_DIFF_THRESHOLD, (
        f"Iteration count mismatch: PCG={result_pcg.iterations}, FCG={result_fcg.iterations}, diff={diff}"
    )
