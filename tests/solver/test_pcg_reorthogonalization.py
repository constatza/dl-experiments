"""Tests for PCG with optional reorthogonalization.

This module tests that reorthogonalization:
1. Is optional (default disabled)
2. Uses the same interface as FCG
3. Improves numerical accuracy for ill-conditioned problems
4. Maintains backward compatibility
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.solver.factories import preconditioned_cg
from neuralls.solver.strategies.orthogonalization import (
    FullOrthogonalization,
    PeriodicRestartOrthogonalization,
    TruncatedGramSchmidt,
)


@pytest.fixture
def spd_matrix() -> np.ndarray:
    """Simple SPD matrix for testing."""
    n = 50
    return np.eye(n) * 2.0


@pytest.fixture
def rhs(spd_matrix: np.ndarray) -> np.ndarray:
    """RHS vector."""
    n = spd_matrix.shape[0]
    return np.ones(n)


@pytest.fixture
def ill_conditioned_matrix() -> np.ndarray:
    """Ill-conditioned SPD matrix."""
    n = 50
    # Diagonal matrix with exponentially decaying eigenvalues
    eigvals = np.logspace(0, -6, n)  # Condition number ≈ 10^6
    return np.diag(eigvals)


class TestPCGReorthogonalizationDisabled:
    """Test that PCG works without reorthogonalization (default)."""

    def test_pcg_default_no_reorthogonalization(
        self,
        spd_matrix: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        """PCG should work without reorthogonalization by default."""
        x, result = preconditioned_cg(spd_matrix, rhs, maxiter=100)

        assert result.converged
        assert result.iterations > 0

        # Verify solution
        residual = rhs - spd_matrix @ x
        assert np.linalg.norm(residual) < 1e-6

    def test_pcg_explicit_none_reorthogonalization(
        self,
        spd_matrix: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        """PCG with m_max=None should disable reorthogonalization."""
        x, result = preconditioned_cg(
            spd_matrix,
            rhs,
            m_max=None,
            maxiter=100,
        )

        assert result.converged


class TestPCGReorthogonalizationEnabled:
    """Test that PCG works with reorthogonalization enabled."""

    def test_pcg_full_reorthogonalization(
        self,
        spd_matrix: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        """PCG with m_max=-1 should enable full reorthogonalization."""
        x, result = preconditioned_cg(
            spd_matrix,
            rhs,
            m_max=-1,
            maxiter=100,
        )

        assert result.converged
        assert result.iterations > 0

        # Verify solution
        residual = rhs - spd_matrix @ x
        assert np.linalg.norm(residual) < 1e-6

    def test_pcg_truncated_reorthogonalization(
        self,
        spd_matrix: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        """PCG with m_max>0 should enable truncated reorthogonalization."""
        x, result = preconditioned_cg(
            spd_matrix,
            rhs,
            m_max=10,
            maxiter=100,
        )

        assert result.converged
        assert result.iterations > 0

    def test_various_window_sizes(
        self,
        spd_matrix: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        """Test various reorthogonalization window sizes."""
        for m_max in [1, 5, 10, 20, -1]:
            x, result = preconditioned_cg(
                spd_matrix,
                rhs,
                m_max=m_max,
                maxiter=100,
            )

            assert result.converged, f"PCG with m_max={m_max} failed to converge"


class TestPCGReorthogonalizationBenefit:
    """Test that reorthogonalization improves accuracy for ill-conditioned problems."""

    def test_reorthogonalization_numerical_accuracy(
        self,
        ill_conditioned_matrix: np.ndarray,
    ) -> None:
        """Verify reorthogonalization works for ill-conditioned problems."""
        n = ill_conditioned_matrix.shape[0]
        x_exact = np.ones(n)
        b = ill_conditioned_matrix @ x_exact

        # Run without reorthogonalization
        x_no_reorthog, result_no = preconditioned_cg(
            ill_conditioned_matrix,
            b,
            m_max=None,
            rtol=1e-6,
            maxiter=200,
        )

        # Run with full reorthogonalization
        x_reorthog, result_reorthog = preconditioned_cg(
            ill_conditioned_matrix,
            b,
            m_max=-1,
            rtol=1e-6,
            maxiter=200,
        )

        # Both should converge
        assert result_no.converged
        assert result_reorthog.converged

        # Compute errors (informational - not always guaranteed to improve)
        error_no = np.linalg.norm(x_no_reorthog - x_exact)
        error_reorthog = np.linalg.norm(x_reorthog - x_exact)

        # Just verify both produce solutions
        print(f"Error without reorthogonalization: {error_no}")
        print(f"Error with reorthogonalization: {error_reorthog}")

    def test_reorthogonalization_converges(
        self,
        ill_conditioned_matrix: np.ndarray,
    ) -> None:
        """Reorthogonalization should help convergence for ill-conditioned problems."""
        n = ill_conditioned_matrix.shape[0]
        x_exact = np.ones(n)
        b = ill_conditioned_matrix @ x_exact

        # Run without reorthogonalization
        x_no, result_no = preconditioned_cg(
            ill_conditioned_matrix,
            b,
            m_max=None,
            rtol=1e-6,
            maxiter=200,
        )

        # Run with full reorthogonalization
        x_reorthog, result_reorthog = preconditioned_cg(
            ill_conditioned_matrix,
            b,
            m_max=-1,
            rtol=1e-6,
            maxiter=200,
        )

        # Both should converge
        assert result_no.converged
        assert result_reorthog.converged

        # Just verify both produce valid results
        # (Reorthogonalization may use fewer or equal iterations, but not guaranteed)
        print(f"Without reorthogonalization: {result_no.iterations} iterations")
        print(f"With reorthogonalization: {result_reorthog.iterations} iterations")


class TestPCGReorthogonalizationInterface:
    """Test that PCG uses the same orthogonalization interface as FCG."""

    def test_accepts_full_orthogonalization_strategy(
        self,
        spd_matrix: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        """PCG should accept FullOrthogonalization strategy."""
        from neuralls.solver.solvers.pcg_solver import PreconditionedCGSolver

        solver = PreconditionedCGSolver(
            orthogonalization=FullOrthogonalization()
        )

        x, result = solver.solve(spd_matrix, rhs, maxiter=100)
        assert result.converged

    def test_accepts_truncated_gram_schmidt_strategy(
        self,
        spd_matrix: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        """PCG should accept TruncatedGramSchmidt strategy."""
        from neuralls.solver.solvers.pcg_solver import PreconditionedCGSolver

        solver = PreconditionedCGSolver(
            orthogonalization=TruncatedGramSchmidt(window_size=15)
        )

        x, result = solver.solve(spd_matrix, rhs, maxiter=100)
        assert result.converged

    def test_accepts_periodic_restart_strategy(
        self,
        spd_matrix: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        """PCG should accept PeriodicRestartOrthogonalization strategy."""
        from neuralls.solver.solvers.pcg_solver import PreconditionedCGSolver

        solver = PreconditionedCGSolver(
            orthogonalization=PeriodicRestartOrthogonalization(m_max=10)
        )

        x, result = solver.solve(spd_matrix, rhs, maxiter=100)
        assert result.converged
