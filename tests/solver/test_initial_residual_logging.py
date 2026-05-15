"""Test initial residual logging in CG solvers."""

import pytest
import numpy as np
from neuralls.domain.solver.factories import pcg, scipy_cg
from neuralls.domain.solver.monitoring.storage import ScalarHistory, VectorHistory


def test_scalar_history_prepend():
    """Test ScalarHistory.prepend() with scalar values."""
    history = ScalarHistory.empty()
    history = history.add(0.5)
    history = history.add(0.25)
    history = history.prepend(1.0)

    assert history.to_list() == [1.0, 0.5, 0.25]


def test_vector_history_prepend():
    """Test VectorHistory.prepend() with vector values."""
    history = VectorHistory.empty()
    history = history.add(np.array([0.5, 0.3]))
    history = history.prepend(np.array([1.0, 0.8]))

    vectors = history.to_array()
    assert len(history) == 2
    assert np.allclose(vectors[0], [1.0, 0.8])
    assert np.allclose(vectors[1], [0.5, 0.3])


@pytest.mark.parametrize("solver_factory", [pcg, scipy_cg])
def test_initial_residual_zero_guess(solver_factory):
    """Test initial residual logging with zero initial guess."""
    n = 10
    A = np.eye(n) * 2.0 + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    b = np.ones(n)

    x, result = solver_factory(
        A,
        b,
        x0=None,  # Zero guess
        rtol=1e-6,
        maxiter=20,
        trace_mode="minimal",
    )

    # Check history includes iteration 0
    assert result.iteration_history is not None
    history = result.iteration_history.residual_norms.to_list()
    assert len(history) > 0

    # Initial residual should equal ||b|| when x0 = 0
    expected_r0 = np.linalg.norm(b)
    assert np.isclose(history[0], expected_r0)


@pytest.mark.parametrize("solver_factory", [pcg, scipy_cg])
def test_initial_residual_nonzero_guess(solver_factory):
    """Test initial residual logging with non-zero initial guess."""
    n = 10
    A = np.eye(n) * 2.0 + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    b = np.ones(n)
    x0 = np.random.rand(n) * 0.1

    x, result = solver_factory(
        A,
        b,
        x0=x0,
        rtol=1e-6,
        maxiter=20,
        trace_mode="minimal",
    )

    # Check history includes iteration 0
    assert result.iteration_history is not None
    history = result.iteration_history.residual_norms.to_list()
    assert len(history) > 0

    # Initial residual should equal ||b - A @ x0||
    r0 = b - A @ x0
    expected_r0 = np.linalg.norm(r0)
    assert np.isclose(history[0], expected_r0, rtol=1e-5)


def test_scipy_cg_vs_fcg_consistency():
    """Test that scipy CG and FCG both capture iteration 0 with consistent formats.

    Both solvers should store:
    - residual_history_rel: RELATIVE residuals (||r|| / ||b||)
    - residual_history_abs: ABSOLUTE residuals (||r||)
    """
    from neuralls.domain.solver.factories import flexible_cg, pcg

    n = 10
    A = np.eye(n) * 2.0 + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    b = np.ones(n)
    x0 = np.zeros(n)

    # Solve with scipy CG
    x_scipy, result_scipy = pcg(A, b, x0=x0, rtol=1e-6, maxiter=20)

    # Solve with FCG
    x_fcg, result_fcg = flexible_cg(A, b, x0=x0, rtol=1e-6, maxiter=20)

    # Both should have iteration 0 in history
    assert result_scipy.residual_history_rel is not None
    assert result_fcg.residual_history_rel is not None
    assert len(result_scipy.residual_history_rel) > 0
    assert len(result_fcg.residual_history_rel) > 0

    # For x0=0, both should have initial relative residual = 1.0
    assert np.isclose(result_scipy.residual_history_rel[0], 1.0, rtol=1e-10)
    assert np.isclose(result_fcg.residual_history_rel[0], 1.0, rtol=1e-10)

    # Both should have absolute residual = ||b||
    assert result_scipy.residual_history_abs is not None
    assert result_fcg.residual_history_abs is not None
    assert np.isclose(result_scipy.residual_history_abs[0], np.linalg.norm(b), rtol=1e-10)
    assert np.isclose(result_fcg.residual_history_abs[0], np.linalg.norm(b), rtol=1e-10)
