"""Test initial residual logging in scipy CG wrapper."""

import numpy as np
from src.solver.pcg_solver import cg_solve
from src.solver.trace_recorder import TraceRecorder


def test_vector_logger_prepend_scalar():
    """Test TraceRecorder.prepend() with scalar values."""
    logger = TraceRecorder()
    logger.log("residual_norm", 0.5)
    logger.log("residual_norm", 0.25)
    logger.prepend("residual_norm", 1.0)

    history = logger.get_history("residual_norm")
    assert history == [1.0, 0.5, 0.25]


def test_vector_logger_prepend_vector():
    """Test TraceRecorder.prepend() with vector values."""
    logger = TraceRecorder()
    logger.log("residual", np.array([0.5, 0.3]))
    logger.prepend("residual", np.array([1.0, 0.8]))

    residuals = logger.get_history("residual")
    assert len(residuals) == 2
    assert np.allclose(residuals[0], [1.0, 0.8])
    assert np.allclose(residuals[1], [0.5, 0.3])


def test_scipy_cg_initial_residual_zero_guess():
    """Test initial residual logging with zero initial guess."""
    n = 10
    A = np.eye(n) * 2.0 + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    b = np.ones(n)

    event_log = TraceRecorder()
    x, result = cg_solve(
        A,
        b,
        x0=None,  # Zero guess
        tol=1e-6,
        maxiter=20,
        callback_type="x",
        event_log=event_log,
    )

    # Check history includes iteration 0
    history = result.residual_history
    assert history is not None
    assert len(history) > 0

    # Initial residual should equal ||b|| when x0 = 0
    expected_r0 = np.linalg.norm(b)
    b_norm = np.linalg.norm(b)
    assert np.isclose(history[0], expected_r0 / b_norm)  # Relative residual


def test_scipy_cg_initial_residual_nonzero_guess():
    """Test initial residual logging with non-zero initial guess."""
    n = 10
    A = np.eye(n) * 2.0 + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    b = np.ones(n)
    x0 = np.random.rand(n) * 0.1

    event_log = TraceRecorder()
    x, result = cg_solve(
        A,
        b,
        x0=x0,
        tol=1e-6,
        maxiter=20,
        callback_type="x",
        event_log=event_log,
    )

    # Check history includes iteration 0
    history = result.residual_history
    assert history is not None
    assert len(history) > 0

    # Initial residual should equal ||b - A @ x0||
    r0 = b - A @ x0
    expected_r0 = np.linalg.norm(r0)
    b_norm = np.linalg.norm(b)
    assert np.isclose(history[0], expected_r0 / b_norm)


def test_scipy_cg_vs_fcg_consistency():
    """Test that scipy CG and FCG both capture iteration 0 with consistent formats.

    Both solvers should store:
    - residual_history: RELATIVE residuals (||r|| / ||b||)
    - residual_history_abs: ABSOLUTE residuals (||r||)
    """
    from src.solver.factories import flexible_cg, preconditioned_cg

    n = 10
    A = np.eye(n) * 2.0 + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    b = np.ones(n)
    x0 = np.zeros(n)

    # Solve with scipy CG
    x_scipy, result_scipy = preconditioned_cg(A, b, x0=x0, rtol=1e-6, max_iter=20)

    # Solve with FCG
    x_fcg, result_fcg = flexible_cg(A, b, x0=x0, rtol=1e-6, max_iterations=20)

    # Both should have iteration 0 in history
    assert result_scipy.residual_history is not None
    assert result_fcg.residual_history is not None
    assert len(result_scipy.residual_history) > 0
    assert len(result_fcg.residual_history) > 0

    # For x0=0, both should have initial relative residual = 1.0
    assert np.isclose(result_scipy.residual_history[0], 1.0, rtol=1e-10)
    assert np.isclose(result_fcg.residual_history[0], 1.0, rtol=1e-10)

    # Both should have absolute residual = ||b||
    assert np.isclose(
        result_scipy.residual_history_abs[0], np.linalg.norm(b), rtol=1e-10
    )
    assert np.isclose(result_fcg.residual_history_abs[0], np.linalg.norm(b), rtol=1e-10)
