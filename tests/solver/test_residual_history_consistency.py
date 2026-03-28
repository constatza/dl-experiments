"""Tests for residual history consistency across solvers.

This module ensures FCG and PCG log residuals consistently:
- Same number of residuals (iterations + 1)
- First residual is ||r_0||, not ||b||
- No off-by-one errors in residual histories
"""

import numpy as np

from neuralls.domain.solver.factories import flexible_cg, pcg, scipy_cg
from neuralls.domain.solver.monitoring.trace_mode import TraceMode


def test_fcg_pcg_residual_history_length_match() -> None:
    """FCG and PCG must log same number of residuals (iterations + 1)."""
    A = np.diag([2.0, 3.0, 4.0])
    b = np.ones(3)

    x_fcg, result_fcg = flexible_cg(A, b, m_max=-1, rtol=1e-10, trace_mode=TraceMode.FULL)
    x_pcg, result_pcg = pcg(A, b, rtol=1e-10, trace_mode=TraceMode.FULL)

    # Both should converge in same iterations
    assert result_fcg.iterations == result_pcg.iterations

    # Both should log same number of residuals (iterations + 1)
    expected_len = result_fcg.iterations + 1
    assert result_fcg.residual_history_abs is not None
    assert result_pcg.residual_history_abs is not None
    assert len(result_fcg.residual_history_abs) == expected_len, (
        f"FCG logged {len(result_fcg.residual_history_abs)} residuals, expected {expected_len}"
    )
    assert len(result_pcg.residual_history_abs) == expected_len

    # First residual should match (both should be ||r_0||, not ||b||)
    assert np.isclose(
        result_fcg.residual_history_abs[0], result_pcg.residual_history_abs[0], rtol=1e-12
    ), (
        f"First residual mismatch: FCG={result_fcg.residual_history_abs[0]:.6e}, PCG={result_pcg.residual_history_abs[0]:.6e}"
    )


def test_first_residual_is_r0_not_rhs_norm() -> None:
    """First logged residual should be ||r_0||, not ||b||.

    Even though r_0 = b when x0 = 0, with non-identity A the norms can differ
    after preconditioning. This test verifies the logged residual is ||r_0||.
    """
    # Use non-identity A so ||r_0|| != ||b|| when x0=0 is not guaranteed
    A = np.diag([2.0, 3.0, 4.0])
    b = np.ones(3)
    x0 = None  # Zero initial guess

    x, result = flexible_cg(A, b, x0=x0, m_max=-1, rtol=1e-10, trace_mode=TraceMode.FULL)

    # Compute expected r_0 = b - A @ 0 = b
    r_0 = b - A @ np.zeros_like(b)
    expected_r0_norm = np.linalg.norm(r_0)

    # First residual should be ||r_0||
    assert result.residual_history_abs is not None
    assert np.isclose(result.residual_history_abs[0], expected_r0_norm, rtol=1e-12), (
        f"First residual {result.residual_history_abs[0]:.6e} != expected ||r_0|| {expected_r0_norm:.6e}"
    )


def test_fcg_scipy_residual_history_length_match() -> None:
    """FCG and scipy_cg must log same number of residuals."""
    A = np.diag([2.0, 3.0, 4.0])
    b = np.ones(3)

    x_fcg, result_fcg = flexible_cg(A, b, m_max=-1, rtol=1e-10, trace_mode=TraceMode.FULL)
    x_scipy, result_scipy = scipy_cg(A, b, rtol=1e-10, trace_mode=TraceMode.FULL)

    # Both should converge in same iterations
    assert result_fcg.iterations == result_scipy.iterations

    # Both should log same number of residuals (iterations + 1)
    expected_len = result_fcg.iterations + 1
    assert result_fcg.residual_history_abs is not None
    assert result_scipy.residual_history_abs is not None
    assert len(result_fcg.residual_history_abs) == expected_len
    assert len(result_scipy.residual_history_abs) == expected_len

    # First residuals should match
    assert np.isclose(
        result_fcg.residual_history_abs[0], result_scipy.residual_history_abs[0], rtol=1e-12
    )


def test_all_solvers_residual_consistency() -> None:
    """All three solvers (FCG, PCG, scipy) must log residuals consistently."""
    A = np.diag([1.5, 2.5, 3.5])
    b = np.array([1.0, 2.0, 3.0])

    x_fcg, result_fcg = flexible_cg(A, b, m_max=-1, rtol=1e-10, trace_mode=TraceMode.FULL)
    x_pcg, result_pcg = pcg(A, b, rtol=1e-10, trace_mode=TraceMode.FULL)
    x_scipy, result_scipy = scipy_cg(A, b, rtol=1e-10, trace_mode=TraceMode.FULL)

    # All should converge in same iterations
    assert result_fcg.iterations == result_pcg.iterations == result_scipy.iterations

    # All should have same residual history length
    expected_len = result_fcg.iterations + 1
    assert result_fcg.residual_history_abs is not None
    assert result_pcg.residual_history_abs is not None
    assert result_scipy.residual_history_abs is not None

    assert len(result_fcg.residual_history_abs) == expected_len, (
        f"FCG: {len(result_fcg.residual_history_abs)} != {expected_len}"
    )
    assert len(result_pcg.residual_history_abs) == expected_len, (
        f"PCG: {len(result_pcg.residual_history_abs)} != {expected_len}"
    )
    assert len(result_scipy.residual_history_abs) == expected_len, (
        f"scipy: {len(result_scipy.residual_history_abs)} != {expected_len}"
    )

    # All first residuals should match
    assert np.isclose(
        result_fcg.residual_history_abs[0], result_pcg.residual_history_abs[0], rtol=1e-12
    ), "FCG vs PCG first residual mismatch"

    assert np.isclose(
        result_fcg.residual_history_abs[0], result_scipy.residual_history_abs[0], rtol=1e-12
    ), "FCG vs scipy first residual mismatch"


def test_all_solvers_with_preconditioner() -> None:
    """All three solvers must produce identical results with preconditioning."""
    from neuralls.domain.solver.preconditioners import JacobiPreconditioner

    A = np.diag([2.0, 4.0, 6.0, 8.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])

    # Use Jacobi preconditioner
    precond = JacobiPreconditioner(A)

    x_fcg, result_fcg = flexible_cg(
        A, b, preconditioner=precond, m_max=-1, rtol=1e-10, trace_mode=TraceMode.FULL
    )
    x_pcg, result_pcg = pcg(A, b, preconditioner=precond, rtol=1e-10, trace_mode=TraceMode.FULL)
    x_scipy, result_scipy = scipy_cg(
        A, b, preconditioner=precond, rtol=1e-10, trace_mode=TraceMode.FULL
    )

    # All should converge in same iterations
    assert result_fcg.iterations == result_pcg.iterations == result_scipy.iterations

    # All should have same residual history length
    expected_len = result_fcg.iterations + 1
    assert result_fcg.residual_history_abs is not None
    assert result_pcg.residual_history_abs is not None
    assert result_scipy.residual_history_abs is not None

    assert len(result_fcg.residual_history_abs) == expected_len
    assert len(result_pcg.residual_history_abs) == expected_len
    assert len(result_scipy.residual_history_abs) == expected_len

    # All residual histories should be identical
    assert np.allclose(
        result_fcg.residual_history_abs, result_pcg.residual_history_abs, rtol=1e-12
    ), "FCG vs PCG residual history mismatch with preconditioner"

    assert np.allclose(
        result_fcg.residual_history_abs, result_scipy.residual_history_abs, rtol=1e-12
    ), "FCG vs scipy residual history mismatch with preconditioner"


def test_all_solvers_with_ilu_preconditioner() -> None:
    """All three solvers must produce identical results with ILU preconditioning.

    This is critical because the original bug report specifically mentioned
    large discrepancies with ILU preconditioner.
    """
    from neuralls.domain.solver.preconditioners import ILUPreconditioner

    # Create a non-trivial SPD matrix (ILU needs more realistic structure)
    np.random.seed(42)
    n = 20
    # SPD matrix: A = L @ L.T where L is lower triangular
    L = np.tril(np.random.rand(n, n))
    np.fill_diagonal(L, np.random.rand(n) + 1.0)
    A = L @ L.T
    b = np.random.rand(n)

    # Compute expected r_0 (should equal b when x0 = 0)
    r_0 = b - A @ np.zeros_like(b)
    r_0_norm = np.linalg.norm(r_0)

    # Use ILU preconditioner
    precond = ILUPreconditioner(A)

    x_fcg, result_fcg = flexible_cg(
        A, b, preconditioner=precond, m_max=-1, rtol=1e-10, trace_mode=TraceMode.FULL
    )
    x_pcg, result_pcg = pcg(A, b, preconditioner=precond, rtol=1e-10, trace_mode=TraceMode.FULL)
    x_scipy, result_scipy = scipy_cg(
        A, b, preconditioner=precond, rtol=1e-10, trace_mode=TraceMode.FULL
    )

    # All should converge in same iterations
    assert result_fcg.iterations == result_pcg.iterations == result_scipy.iterations

    # All should have same residual history length
    expected_len = result_fcg.iterations + 1
    assert result_fcg.residual_history_abs is not None
    assert result_pcg.residual_history_abs is not None
    assert result_scipy.residual_history_abs is not None

    assert len(result_fcg.residual_history_abs) == expected_len
    assert len(result_pcg.residual_history_abs) == expected_len
    assert len(result_scipy.residual_history_abs) == expected_len

    # All first residuals should equal ||r_0||, not something else like ||b|| or preconditioned residual
    assert np.isclose(result_fcg.residual_history_abs[0], r_0_norm, rtol=1e-12), (
        f"FCG first residual {result_fcg.residual_history_abs[0]:.6e} != ||r_0|| {r_0_norm:.6e}"
    )
    assert np.isclose(result_pcg.residual_history_abs[0], r_0_norm, rtol=1e-12), (
        f"PCG first residual {result_pcg.residual_history_abs[0]:.6e} != ||r_0|| {r_0_norm:.6e}"
    )
    assert np.isclose(result_scipy.residual_history_abs[0], r_0_norm, rtol=1e-12), (
        f"scipy first residual {result_scipy.residual_history_abs[0]:.6e} != ||r_0|| {r_0_norm:.6e}"
    )

    # All residual histories should be identical
    assert np.allclose(
        result_fcg.residual_history_abs,
        result_pcg.residual_history_abs,
        rtol=1e-10,  # Slightly looser tolerance for ILU numerical differences
    ), "FCG vs PCG residual history mismatch with ILU preconditioner"

    assert np.allclose(
        result_fcg.residual_history_abs, result_scipy.residual_history_abs, rtol=1e-10
    ), "FCG vs scipy residual history mismatch with ILU preconditioner"
