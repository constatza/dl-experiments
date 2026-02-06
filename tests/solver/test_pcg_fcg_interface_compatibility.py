"""Test that PCG and FCG use the same orthogonalization interface.

Verifies that:
1. Both solvers accept the same orthogonalization strategies
2. Both use the same factory pattern (m_max parameter)
3. Interface is fully compatible
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.solver.factories import flexible_cg, pcg
from neuralls.solver.strategies.orthogonalization import create_fcg_orthogonalization


@pytest.fixture
def spd_matrix() -> np.ndarray:
    """Simple SPD matrix."""
    n = 30
    return np.eye(n) * 2.0


@pytest.fixture
def rhs(spd_matrix: np.ndarray) -> np.ndarray:
    """RHS vector."""
    n = spd_matrix.shape[0]
    return np.ones(n)


class TestPCGFCGInterfaceCompatibility:
    """Test that PCG and FCG use compatible interfaces."""

    def test_both_accept_m_max_parameter(
        self,
        spd_matrix: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        """Both solvers should accept m_max parameter in factory functions."""
        # PCG with m_max
        x_pcg, result_pcg = pcg(
            spd_matrix,
            rhs,
            m_max=10,
            maxiter=100,
        )

        # FCG with m_max
        x_fcg, result_fcg = flexible_cg(
            spd_matrix,
            rhs,
            m_max=10,
            maxiter=100,
        )

        # Both should converge
        assert result_pcg.converged
        assert result_fcg.converged

    def test_both_accept_unlimited_reorthogonalization(
        self,
        spd_matrix: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        """Both solvers should accept m_max=-1 for full reorthogonalization."""
        # PCG with full reorthogonalization
        x_pcg, result_pcg = pcg(
            spd_matrix,
            rhs,
            m_max=-1,
            maxiter=100,
        )

        # FCG with full reorthogonalization
        x_fcg, result_fcg = flexible_cg(
            spd_matrix,
            rhs,
            m_max=-1,
            maxiter=100,
        )

        # Both should converge
        assert result_pcg.converged
        assert result_fcg.converged

    def test_both_use_same_factory_function(
        self,
        spd_matrix: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        """Both PCG and FCG solvers should use create_fcg_orthogonalization factory."""
        from neuralls.solver import PCGSolver, FCGSolver

        # Create orthogonalization strategy
        orthog_strategy = create_fcg_orthogonalization(m_max=15)

        # PCG with reorthogonalization
        pcg_solver = PCGSolver(reorthogonalization=orthog_strategy)

        # FCG with orthogonalization
        fcg_solver = FCGSolver(orthogonalization=orthog_strategy)

        # Both should solve successfully
        x_pcg, result_pcg = pcg_solver.solve(spd_matrix, rhs, maxiter=100)
        x_fcg, result_fcg = fcg_solver.solve(spd_matrix, rhs, maxiter=100)

        assert result_pcg.converged
        assert result_fcg.converged

