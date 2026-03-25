"""Benchmark solver on user-provided matrix files."""

import pytest
import numpy as np
from neuralls.solver.factories import flexible_cg, scipy_cg
from neuralls.solver.preconditioners import (
    ICholeskyPreconditioner,
    IC0Preconditioner,
)
from neuralls.solver.monitoring.trace_mode import TraceMode
from .conftest import (
    IC0_MATRIX_RTOL,
    IC0_MATRIX_ATOL,
    IC0_SOLUTION_RTOL,
    IC0_SOLUTION_ATOL,
    IC0_THRESHOLD,
    IC0_SPARSITY_COUNT_TOLERANCE,
    SPARSITY_THRESHOLD,
)

# FCG and PCG must match exactly on iteration counts
ITER_DIFF_THRESHOLD = 0


@pytest.mark.benchmark
@pytest.mark.parametrize("precond_type", [None, "ilu"])
def test_fcg_pcg_equivalence_on_file(system, convergence_tolerances, precond_type, matrix_path):
    """Test FCG vs PCG equivalence on matrix loaded from file.

    Args:
        system: Tuple (A, b, x_exact) from fixture.
        convergence_tolerances: Tuple (rtol, atol).
        precond_type: Type of preconditioner to use (None or "ilu").
        matrix_path: Path to the matrix file (from fixture).
    """
    A, b, x_exact, L = system
    rtol, atol = convergence_tolerances
    A.shape[0]

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

    assert result_pcg.converged, f"PCG failed to converge. Iterations: {result_pcg.iterations}"
    assert result_fcg.converged, f"FCG failed to converge. Iterations: {result_fcg.iterations}"

    # Check iteration difference
    diff = abs(result_pcg.iterations - result_fcg.iterations)
    assert diff <= ITER_DIFF_THRESHOLD, (
        f"Iteration count mismatch: PCG={result_pcg.iterations}, FCG={result_fcg.iterations}, diff={diff}"
    )


@pytest.mark.benchmark
def test_ic0_quality_vs_reference(system, convergence_tolerances):
    """Verify IC0Preconditioner has zero fill-in and reasonable reconstruction quality.

    Theory:
        IC(0) computes sparse incomplete Cholesky: L @ L.T ≈ A
        Key IC(0) property: ZERO fill-in - L has same sparsity pattern as lower(A)
        The reference L matrix appears to be a different incomplete factorization.

    Validation:
        - Verify IC0 has ZERO fill-in (same sparsity as A)
        - Verify IC0 reconstruction error is reasonable (~10-20%)
        - Compare quality metrics vs reference L

    Args:
        system: Tuple (A, b, x_exact, L_reference) from fixture.
        convergence_tolerances: Tuple (rtol, atol).
    """
    A, b, x_exact, L_reference = system

    # Create IC0 preconditioner with centralized threshold
    ic0_precond = IC0Preconditioner(A, threshold=IC0_THRESHOLD)

    # Access the computed L matrix (private property)
    L_computed = ic0_precond._operator

    # Compute sparsity metrics using centralized threshold
    A_lower = np.tril(A)
    A_lower_nnz = np.count_nonzero(A_lower)
    L_computed_nnz = np.count_nonzero(np.abs(L_computed) > SPARSITY_THRESHOLD)
    L_ref_nnz = np.count_nonzero(np.abs(L_reference) > SPARSITY_THRESHOLD)

    # Check sparsity difference (allow small tolerance due to numerical precision)
    sparsity_diff = abs(L_computed_nnz - L_ref_nnz)
    assert sparsity_diff <= IC0_SPARSITY_COUNT_TOLERANCE, (
        f"IC(0) sparsity mismatch with reference exceeds tolerance! "
        f"IC0 nnz={L_computed_nnz}, Ref nnz={L_ref_nnz}, "
        f"diff={sparsity_diff}, max_allowed={IC0_SPARSITY_COUNT_TOLERANCE}"
    )

    # Verify L matrices are close in values (this is the real test)
    np.testing.assert_allclose(
        L_computed,
        L_reference,
        rtol=IC0_MATRIX_RTOL,
        atol=IC0_MATRIX_ATOL,
        err_msg="IC0 L matrix values differ from reference",
    )

    # Compute reconstructions
    A_ref_reconstructed = L_reference @ L_reference.T
    A_ic0_reconstructed = L_computed @ L_computed.T

    # Compute reconstruction errors
    ref_error = np.linalg.norm(A - A_ref_reconstructed, "fro") / np.linalg.norm(A, "fro")
    ic0_error = np.linalg.norm(A - A_ic0_reconstructed, "fro") / np.linalg.norm(A, "fro")

    # IC(0) should have reasonable reconstruction error (typically 5-20% for IC0)
    assert ic0_error < 0.5, (
        f"IC0 reconstruction error too large: {ic0_error:.6e} (expected < 0.5 for IC0)"
    )

    # Log quality comparison
    print(f"\nSparsity: A_lower={A_lower_nnz}, IC0={L_computed_nnz}, Ref={L_ref_nnz}")
    print(f"Reconstruction error: IC0={ic0_error:.4f}, Ref={ref_error:.4f}")


@pytest.mark.benchmark
def test_ic0_solver_performance_vs_reference(system, convergence_tolerances):
    """Compare CG solver performance using IC0 vs reference L preconditioner.

    Validates that IC0-computed L produces reasonable convergence.
    Both IC0 and reference have similar reconstruction errors (~10-11%),
    so iteration counts should be comparable.

    Args:
        system: Tuple (A, b, x_exact, L_reference) from fixture.
        convergence_tolerances: Tuple (rtol, atol).
    """
    A, b, x_exact, L_reference = system
    rtol, atol = convergence_tolerances

    # Create IC0 preconditioner with centralized threshold
    ic0_precond = IC0Preconditioner(A, threshold=IC0_THRESHOLD)

    # Create reference preconditioner from file
    ref_precond = ICholeskyPreconditioner(L_reference)

    # Solve with IC0
    x_ic0, result_ic0 = flexible_cg(
        A,
        b,
        preconditioner=ic0_precond,
        rtol=rtol,
        atol=atol,
        maxiter=1000,
    )

    # Solve with reference
    x_ref, result_ref = flexible_cg(
        A,
        b,
        preconditioner=ref_precond,
        rtol=rtol,
        atol=atol,
        maxiter=1000,
    )

    # Both should converge
    assert result_ic0.converged, (
        f"IC0 preconditioner failed to converge. "
        f"Iterations: {result_ic0.iterations}, "
        f"Final residual: {result_ic0.residual}"
    )
    assert result_ref.converged, (
        f"Reference preconditioner failed to converge. "
        f"Iterations: {result_ref.iterations}, "
        f"Final residual: {result_ref.residual}"
    )

    # Both should converge to same solution
    np.testing.assert_allclose(
        x_ic0,
        x_ref,
        rtol=IC0_SOLUTION_RTOL,
        atol=IC0_SOLUTION_ATOL,
        err_msg="IC0 and reference produce different solutions",
    )

    # CRITICAL: Iteration counts must match EXACTLY (zero tolerance for integer metrics)
    assert result_ic0.iterations == result_ref.iterations, (
        f"Iteration count mismatch - algorithm must converge in exact iterations! "
        f"IC0={result_ic0.iterations}, reference={result_ref.iterations}, "
        f"diff={abs(result_ic0.iterations - result_ref.iterations)}"
    )

    # Log performance
    print(f"\nIterations: IC0={result_ic0.iterations}, Ref={result_ref.iterations} (exact match ✓)")
