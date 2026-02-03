"""Test that PCG and FCG use the same orthogonalization interface.

Verifies that:
1. Both solvers accept the same orthogonalization strategies
2. Both use the same factory pattern (m_max parameter)
3. Interface is fully compatible
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.solver.factories import flexible_cg, preconditioned_cg
from neuralls.solver.strategies.orthogonalization import (
    FullOrthogonalization,
    PeriodicRestartOrthogonalization,
    TruncatedGramSchmidt,
    create_fcg_orthogonalization,
)


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
        x_pcg, result_pcg = preconditioned_cg(
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
        x_pcg, result_pcg = preconditioned_cg(
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
        """Both solvers should use create_fcg_orthogonalization factory."""
        from neuralls.solver.solvers.pcg_solver import PreconditionedCGSolver
        from neuralls.solver.solvers.fcg_solver import FlexibleCGSolver

        # Create orthogonalization strategy
        orthog_strategy = create_fcg_orthogonalization(m_max=15)

        # Both solvers accept the same strategy
        pcg_solver = PreconditionedCGSolver(orthogonalization=orthog_strategy)
        fcg_solver = FlexibleCGSolver(orthogonalization=orthog_strategy)

        # Both should solve successfully
        x_pcg, result_pcg = pcg_solver.solve(spd_matrix, rhs, maxiter=100)
        x_fcg, result_fcg = fcg_solver.solve(spd_matrix, rhs, maxiter=100)

        assert result_pcg.converged
        assert result_fcg.converged

    def test_both_accept_full_orthogonalization(
        self,
        spd_matrix: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        """Both solvers should accept FullOrthogonalization strategy."""
        from neuralls.solver.solvers.pcg_solver import PreconditionedCGSolver
        from neuralls.solver.solvers.fcg_solver import FlexibleCGSolver

        strategy = FullOrthogonalization()

        pcg_solver = PreconditionedCGSolver(orthogonalization=strategy)
        fcg_solver = FlexibleCGSolver(orthogonalization=strategy)

        x_pcg, result_pcg = pcg_solver.solve(spd_matrix, rhs, maxiter=100)
        x_fcg, result_fcg = fcg_solver.solve(spd_matrix, rhs, maxiter=100)

        assert result_pcg.converged
        assert result_fcg.converged

    def test_both_accept_truncated_gram_schmidt(
        self,
        spd_matrix: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        """Both solvers should accept TruncatedGramSchmidt strategy."""
        from neuralls.solver.solvers.pcg_solver import PreconditionedCGSolver
        from neuralls.solver.solvers.fcg_solver import FlexibleCGSolver

        strategy = TruncatedGramSchmidt(window_size=20)

        pcg_solver = PreconditionedCGSolver(orthogonalization=strategy)
        fcg_solver = FlexibleCGSolver(orthogonalization=strategy)

        x_pcg, result_pcg = pcg_solver.solve(spd_matrix, rhs, maxiter=100)
        x_fcg, result_fcg = fcg_solver.solve(spd_matrix, rhs, maxiter=100)

        assert result_pcg.converged
        assert result_fcg.converged

    def test_both_accept_periodic_restart(
        self,
        spd_matrix: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        """Both solvers should accept PeriodicRestartOrthogonalization strategy."""
        from neuralls.solver.solvers.pcg_solver import PreconditionedCGSolver
        from neuralls.solver.solvers.fcg_solver import FlexibleCGSolver

        strategy = PeriodicRestartOrthogonalization(m_max=12)

        pcg_solver = PreconditionedCGSolver(orthogonalization=strategy)
        fcg_solver = FlexibleCGSolver(orthogonalization=strategy)

        x_pcg, result_pcg = pcg_solver.solve(spd_matrix, rhs, maxiter=100)
        x_fcg, result_fcg = fcg_solver.solve(spd_matrix, rhs, maxiter=100)

        assert result_pcg.converged
        assert result_fcg.converged
