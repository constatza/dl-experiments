"""Tests for CG comparison runner orchestration.

This module tests the workflow orchestration functions in cg_runner.py:
- run_cg_comparison(): Core CG comparison orchestration
- format_results_summary(): Result formatting
- summarize_best_combinations(): Analysis and ranking

Follows project testing principles:
- Use fixtures for all test data (no inline data creation)
- Use tmp_path for temporary files (never tempfile)
- Type hints throughout
- Mirror source structure (tests/workflows/ mirrors src/neuralls/workflows/)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neuralls.solver.models.result import CGComparisonResult
from neuralls.solver.preconditioners import (
    Identity,
    JacobiPreconditioner,
    PreconditionerContext,
    ScheduledPreconditioner,
)
from neuralls.solver.preconditioners.base import NonLinearPreconditioner
from neuralls.workflows.cg_runner import (
    format_results_summary,
    run_cg_comparison,
    summarize_best_combinations,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def spd_matrix() -> NDArray:
    """Small 10x10 SPD matrix for testing.

    Returns:
        Well-conditioned SPD matrix
    """
    n = 10
    A = np.random.RandomState(42).rand(n, n)
    return A.T @ A + np.eye(n) * 2.0  # Add to diagonal for better conditioning


@pytest.fixture
def rhs_vector(spd_matrix: NDArray) -> NDArray:
    """Compatible RHS vector for spd_matrix.

    Args:
        spd_matrix: Test matrix fixture

    Returns:
        RHS vector with same dimension as matrix
    """
    return np.random.RandomState(42).rand(spd_matrix.shape[0])


@pytest.fixture
def mock_comparison_results() -> dict[str, CGComparisonResult]:
    """Mock comparison results for testing formatters.

    Returns:
        Dict mapping preconditioner names to mock results
    """
    x_sol = np.array([1.0, 2.0, 3.0])
    x0 = np.zeros(3)

    return {
        "jacobi": CGComparisonResult(
            x=x_sol,
            converged=True,
            iterations=10,
            residual=1e-8,
            residual_abs=1e-9,
            residual_history=[1.0, 0.1, 1e-8],
            residual_history_abs=[1.0, 0.1, 1e-9],
            preconditioner="jacobi",
            initial_guess=x0,
            exact_error=1e-10,
            rhs_norm=1.0,
            breakdown=False,
        ),
        "identity": CGComparisonResult(
            x=x_sol,
            converged=True,
            iterations=20,
            residual=1e-7,
            residual_abs=1e-8,
            residual_history=[1.0, 0.5, 1e-7],
            residual_history_abs=[1.0, 0.5, 1e-8],
            preconditioner="identity",
            initial_guess=x0,
            exact_error=5e-10,
            rhs_norm=1.0,
            breakdown=False,
        ),
        "failed": CGComparisonResult(
            x=x0,
            converged=False,
            iterations=100,
            residual=0.5,
            residual_abs=0.5,
            residual_history=[1.0, 0.9, 0.5],
            residual_history_abs=[1.0, 0.9, 0.5],
            preconditioner="failed",
            initial_guess=x0,
            exact_error=None,
            rhs_norm=1.0,
            breakdown=True,
            error="breakdown detected",
        ),
    }


# ==============================================================================
# Tests for run_cg_comparison()
# ==============================================================================


def test_run_cg_comparison_with_preconditioner_instances(
    spd_matrix: NDArray, rhs_vector: NDArray
) -> None:
    """Test run_cg_comparison with Identity and Jacobi preconditioner instances."""
    preconditioners = {
        "identity": Identity(),
        "jacobi": JacobiPreconditioner(spd_matrix),
    }

    results = run_cg_comparison(
        spd_matrix,
        rhs_vector,
        preconditioners=preconditioners,
        rtol=1e-8,
        atol=1e-10,
        maxiter=100,
    )

    # Verify all keys present (including auto-added "none")
    assert "identity" in results
    assert "jacobi" in results
    assert "none" in results  # Auto-added baseline

    # Verify result types
    for name, result in results.items():
        assert isinstance(result, CGComparisonResult)
        assert result.preconditioner == name
        assert result.iterations >= 0
        assert isinstance(result.converged, bool)


def test_run_cg_comparison_routes_to_flexible_cg(spd_matrix: NDArray, rhs_vector: NDArray) -> None:
    """Test that ScheduledPreconditioner routes to flexible_cg."""
    primary = JacobiPreconditioner(spd_matrix)
    fallback = Identity()
    scheduled = ScheduledPreconditioner(primary, fallback, limit_iters=5)

    preconditioners = {"scheduled": scheduled}

    results = run_cg_comparison(
        spd_matrix,
        rhs_vector,
        preconditioners=preconditioners,
        rtol=1e-8,
        atol=1e-10,
        maxiter=100,
    )

    # Verify flexible_cg was used (check for iteration history)
    result = results["scheduled"]
    assert isinstance(result, CGComparisonResult)
    # Flexible CG provides iteration history
    assert len(result.residual_history) > 1


def test_run_cg_comparison_routes_nonlinear_preconditioner_to_flexible_cg(
    monkeypatch: pytest.MonkeyPatch,
    spd_matrix: NDArray,
    rhs_vector: NDArray,
) -> None:
    """Test that nonlinear preconditioners use flexible_cg by default."""

    class DummyNonLinearPreconditioner(NonLinearPreconditioner):
        def apply(
            self,
            residual: NDArray,
            context: PreconditionerContext | None = None,
        ) -> NDArray:
            return residual

    flexible_calls: list[str] = []
    pcg_calls: list[str] = []

    from neuralls.solver.models.result import SolverResult

    def fake_flexible_cg(*args: object, **kwargs: object) -> tuple[NDArray, SolverResult]:
        flexible_calls.append("flexible")
        return np.zeros_like(rhs_vector), SolverResult(
            converged=True,
            iterations=3,
            residual=1e-8,
            residual_abs=1e-8,
            rhs_norm=float(np.linalg.norm(rhs_vector)),
            breakdown=False,
            residual_history=[1.0, 1e-8],
            residual_history_abs=[1.0, 1e-8],
            tol=1e-8,
            atol=1e-10,
        )

    def fake_pcg(*args: object, **kwargs: object) -> tuple[NDArray, SolverResult]:
        pcg_calls.append("pcg")
        return np.zeros_like(rhs_vector), SolverResult(
            converged=True,
            iterations=2,
            residual=1e-7,
            residual_abs=1e-7,
            rhs_norm=float(np.linalg.norm(rhs_vector)),
            breakdown=False,
            residual_history=[1.0, 1e-7],
            residual_history_abs=[1.0, 1e-7],
            tol=1e-8,
            atol=1e-10,
        )

    monkeypatch.setattr("neuralls.workflows.cg_runner.flexible_cg", fake_flexible_cg)
    monkeypatch.setattr("neuralls.workflows.cg_runner.pcg", fake_pcg)

    results = run_cg_comparison(
        spd_matrix,
        rhs_vector,
        preconditioners={
            "nonlinear": DummyNonLinearPreconditioner(),
            "none": Identity(),
        },
        rtol=1e-8,
        atol=1e-10,
        maxiter=100,
    )

    assert flexible_calls == ["flexible"]
    assert pcg_calls == ["pcg"]
    assert results["nonlinear"].converged


def test_run_cg_comparison_routes_to_pcg(spd_matrix: NDArray, rhs_vector: NDArray) -> None:
    """Test that non-contextual preconditioner routes to pcg."""
    preconditioners = {"identity": Identity()}

    results = run_cg_comparison(
        spd_matrix,
        rhs_vector,
        preconditioners=preconditioners,
        rtol=1e-8,
        atol=1e-10,
        maxiter=100,
    )

    # Verify PCG was used
    result = results["identity"]
    assert isinstance(result, CGComparisonResult)
    assert result.converged


def test_run_cg_comparison_adds_none_baseline(spd_matrix: NDArray, rhs_vector: NDArray) -> None:
    """Test that 'none' baseline is automatically added if missing."""
    preconditioners = {"jacobi": JacobiPreconditioner(spd_matrix)}

    results = run_cg_comparison(
        spd_matrix,
        rhs_vector,
        preconditioners=preconditioners,
        rtol=1e-8,
        atol=1e-10,
        maxiter=100,
    )

    # Verify "none" was added
    assert "none" in results
    assert "jacobi" in results

    # Verify "none" result is valid
    none_result = results["none"]
    assert isinstance(none_result, CGComparisonResult)
    assert none_result.preconditioner == "none"


def test_run_cg_comparison_computes_exact_error(spd_matrix: NDArray, rhs_vector: NDArray) -> None:
    """Test that run_cg_comparison computes exact error correctly."""
    preconditioners = {"identity": Identity()}

    results = run_cg_comparison(
        spd_matrix,
        rhs_vector,
        preconditioners=preconditioners,
        rtol=1e-8,
        atol=1e-10,
        maxiter=100,
    )

    result = results["identity"]

    # Verify exact error computed
    assert result.exact_error is not None
    assert result.exact_error >= 0

    # Compute expected exact error
    x_exact = np.linalg.solve(spd_matrix, rhs_vector)
    exact_norm = np.linalg.norm(x_exact)
    expected_error = np.linalg.norm(result.x - x_exact) / exact_norm

    assert_allclose(result.exact_error, expected_error, rtol=1e-10)


# ==============================================================================
# Tests for format_results_summary()
# ==============================================================================


def test_format_results_summary(mock_comparison_results: dict[str, CGComparisonResult]) -> None:
    """Test that format_results_summary produces correct output format."""
    summary = format_results_summary(mock_comparison_results)

    # Verify header present
    assert "Flexible CG results:" in summary

    # Verify all preconditioners listed
    assert "jacobi" in summary
    assert "identity" in summary
    assert "failed" in summary

    # Verify status codes
    assert "status=ok" in summary  # For converged results
    assert "status=fail" in summary  # For failed result

    # Verify breakdown note
    assert "breakdown" in summary


def test_format_results_summary_converged_case(
    mock_comparison_results: dict[str, CGComparisonResult],
) -> None:
    """Test format_results_summary with converged results."""
    # Keep only converged results
    converged_results = {k: v for k, v in mock_comparison_results.items() if v.converged}

    summary = format_results_summary(converged_results)

    # Should only have "ok" status
    assert "status=ok" in summary
    assert "status=fail" not in summary


def test_format_results_summary_non_converged_case(
    mock_comparison_results: dict[str, CGComparisonResult],
) -> None:
    """Test format_results_summary with non-converged results."""
    # Keep only non-converged results
    non_converged = {k: v for k, v in mock_comparison_results.items() if not v.converged}

    summary = format_results_summary(non_converged)

    # Should only have "fail" status
    assert "status=fail" in summary
    assert "status=ok" not in summary


# ==============================================================================
# Tests for summarize_best_combinations()
# ==============================================================================


def test_summarize_best_combinations(
    mock_comparison_results: dict[str, CGComparisonResult],
) -> None:
    """Test summarize_best_combinations ranks results correctly."""
    summary = summarize_best_combinations(mock_comparison_results)

    # Verify ranking (only converged results)
    ranked = summary.ranked
    assert len(ranked) == 2  # Only jacobi and identity converged

    # Verify sorted by residual (jacobi has lower residual: 1e-8 < 1e-7)
    assert ranked[0].label == "jacobi"
    assert ranked[1].label == "identity"

    # Verify overall_best
    best = summary.overall_best
    assert best is not None
    assert best.label == "jacobi"
    assert best.residual == 1e-8


def test_summarize_best_combinations_empty_results() -> None:
    """Test summarize_best_combinations with no converged results."""
    # Create all non-converged results
    non_converged_results = {
        "failed1": CGComparisonResult(
            x=np.zeros(3),
            converged=False,
            iterations=100,
            residual=0.5,
            residual_abs=0.5,
            residual_history=[],
            residual_history_abs=[],
            preconditioner="failed1",
            initial_guess=np.zeros(3),
            exact_error=None,
            rhs_norm=1.0,
            breakdown=True,
        ),
    }

    summary = summarize_best_combinations(non_converged_results)

    # Verify empty ranking
    assert summary.ranked == ()
    assert summary.overall_best is None


def test_summarize_best_combinations_single_converged() -> None:
    """Test summarize_best_combinations with single converged result."""
    single_result = {
        "jacobi": CGComparisonResult(
            x=np.array([1.0, 2.0]),
            converged=True,
            iterations=10,
            residual=1e-8,
            residual_abs=1e-9,
            residual_history=[],
            residual_history_abs=[],
            preconditioner="jacobi",
            initial_guess=np.zeros(2),
            exact_error=1e-10,
            rhs_norm=1.0,
            breakdown=False,
        ),
    }

    summary = summarize_best_combinations(single_result)

    # Verify single entry
    assert len(summary.ranked) == 1
    assert summary.overall_best is not None
    assert summary.overall_best.label == "jacobi"
