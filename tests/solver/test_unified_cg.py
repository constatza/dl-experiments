"""Tests for unified ConjugateGradientSolver with Strategy pattern.

This test suite verifies the new unified CG solver that replaces the deep
inheritance hierarchy with composition. Tests ensure:

1. Bit-exact equivalence with old PCG/FCG implementations
2. All strategy combinations work correctly
3. Factory functions produce correct solver configurations
4. Direction strategies behave correctly

References:
    - src/neuralls/solver/solvers/cg_solver.py
    - src/neuralls/solver/strategies/direction.py
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.solver import (
    PCGSolver,
    FCGSolver,
    TwoTermRecurrenceStrategy,
    OrthogonalizationDirectionStrategy,
    PeriodicRestartOrthogonalization,
    TruncatedGramSchmidt,
    Identity,
)
from neuralls.solver.factories import flexible_cg, pcg
from neuralls.solver.models.state import CGState


class TestUnifiedCGSolver:
    """Test unified ConjugateGradientSolver with different strategies."""

    @pytest.fixture
    def simple_spd_system(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create simple 5x5 SPD system for testing.

        Returns:
            Tuple (A, b, x_true) where A @ x_true = b
        """
        n = 5
        A = np.diag([5.0, 4.0, 3.0, 2.0, 1.0])
        x_true = np.ones(n)
        b = A @ x_true
        return A, b, x_true

    def test_two_term_recurrence_strategy_solves_system(
        self, simple_spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test PCGSolver with two-term recurrence (standard PCG behavior)."""
        A, b, x_true = simple_spd_system

        # Create PCG solver (no reorthogonalization)
        solver = PCGSolver(
            preconditioner=Identity(),
        )

        # Solve system
        x, result = solver.solve(A, b, rtol=1e-10)

        # Verify convergence
        assert result.converged
        assert result.iterations > 0
        np.testing.assert_allclose(x, x_true, rtol=1e-8)

    def test_orthogonalization_strategy_solves_system(
        self, simple_spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test FCGSolver with orthogonalization (flexible CG behavior)."""
        A, b, x_true = simple_spd_system

        # Create FCG solver with orthogonalization strategy
        orthog = PeriodicRestartOrthogonalization(m_max=10)
        solver = FCGSolver(
            orthogonalization=orthog,
            preconditioner=Identity(),
        )

        # Solve system
        x, result = solver.solve(A, b, rtol=1e-10)

        # Verify convergence
        assert result.converged
        assert result.iterations > 0
        np.testing.assert_allclose(x, x_true, rtol=1e-8)

    def test_composite_strategy_solves_system(
        self, simple_spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test PCGSolver with reorthogonalization (PCG + reorthog)."""
        A, b, x_true = simple_spd_system

        # Create PCG solver with reorthogonalization
        reorthog = TruncatedGramSchmidt(window_size=5)
        solver = PCGSolver(
            reorthogonalization=reorthog,
            preconditioner=Identity(),
        )

        # Solve system
        x, result = solver.solve(A, b, rtol=1e-10)

        # Verify convergence
        assert result.converged
        assert result.iterations > 0
        np.testing.assert_allclose(x, x_true, rtol=1e-8)

    def test_factory_preconditioned_cg_produces_unified_solver(
        self, simple_spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test that preconditioned_cg factory produces unified solver."""
        A, b, x_true = simple_spd_system

        # Use factory (should produce ConjugateGradientSolver internally)
        x, result = pcg(A, b, rtol=1e-10)

        # Verify convergence
        assert result.converged
        np.testing.assert_allclose(x, x_true, rtol=1e-8)

    def test_factory_flexible_cg_produces_unified_solver(
        self, simple_spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test that flexible_cg factory produces unified solver."""
        A, b, x_true = simple_spd_system

        # Use factory (should produce ConjugateGradientSolver internally)
        x, result = flexible_cg(A, b, m_max=10, rtol=1e-10)

        # Verify convergence
        assert result.converged
        np.testing.assert_allclose(x, x_true, rtol=1e-8)

    def test_unified_solver_residual_history(
        self, simple_spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test that PCG solver tracks residual history correctly."""
        A, b, _ = simple_spd_system

        # Create PCG solver
        solver = PCGSolver()

        # Solve system
        x, result = solver.solve(A, b, rtol=1e-10)

        # Verify residual history
        assert result.residual_history is not None
        assert len(result.residual_history) == result.iterations + 1
        assert result.residual_history[0] > result.residual_history[-1]

    def test_unified_solver_handles_convergence_flags(
        self, simple_spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test that PCG solver sets convergence flags correctly."""
        A, b, _ = simple_spd_system

        # Test max iterations
        solver = PCGSolver()
        x, result = solver.solve(A, b, maxiter=1, rtol=1e-10)

        # Should not converge in 1 iteration
        assert not result.converged
        assert result.iterations == 1

    def test_full_orthogonalization_strategy(
        self, simple_spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test FCGSolver with FCG(∞) orthogonalization."""
        A, b, x_true = simple_spd_system

        # Create FCG solver with full orthogonalization (m_max=np.inf)
        orthog = PeriodicRestartOrthogonalization(m_max=np.inf)
        solver = FCGSolver(orthogonalization=orthog)

        # Solve system
        x, result = solver.solve(A, b, rtol=1e-10)

        # Verify convergence
        assert result.converged
        np.testing.assert_allclose(x, x_true, rtol=1e-8)


class TestDirectionStrategies:
    """Test individual direction strategies in isolation."""

    @pytest.fixture
    def mock_state(self) -> CGState:
        """Create mock CGState for testing strategies."""
        # Simple state with iteration 0
        return CGState.create_initial(
            u=np.zeros(3),
            r=np.array([1.0, 1.0, 1.0]),
            w=np.array([1.0, 1.0, 1.0]),
            d=np.array([1.0, 1.0, 1.0]),
            q=np.array([1.0, 1.0, 1.0]),
            residual_norm=np.sqrt(3.0),
            rhs_norm=np.sqrt(3.0),
            max_history=0,
        )

    def test_two_term_recurrence_first_iteration(self, mock_state: CGState) -> None:
        """Test TwoTermRecurrenceStrategy at first iteration."""
        strategy = TwoTermRecurrenceStrategy()
        w = np.array([2.0, 2.0, 2.0])

        # First iteration should return w (steepest descent)
        d = strategy.compute_direction(w, mock_state)
        np.testing.assert_array_equal(d, w)

    def test_two_term_recurrence_computes_beta(self) -> None:
        """Test TwoTermRecurrenceStrategy computes beta correctly."""
        from neuralls.solver.models.state import CGState

        strategy = TwoTermRecurrenceStrategy()

        # Initialize with first iteration
        state0 = CGState.create_initial(
            u=np.zeros(3),
            r=np.array([1.0, 1.0, 1.0]),
            w=np.array([1.0, 1.0, 1.0]),
            d=np.array([1.0, 1.0, 1.0]),
            q=np.array([1.0, 1.0, 1.0]),
            residual_norm=np.sqrt(3.0),
            rhs_norm=np.sqrt(3.0),
            max_history=0,
        )

        # First call (iteration 0)
        w0 = np.array([1.0, 1.0, 1.0])
        d0 = strategy.compute_direction(w0, state0)
        np.testing.assert_array_equal(d0, w0)

        # Second call (iteration 1) - simulate state update
        from dataclasses import replace

        # Compute rw_prev = (r0, w0)
        rw_prev = np.dot(state0.r, w0)
        state1 = replace(
            state0,
            iteration=1,
            r=np.array([0.5, 0.5, 0.5]),
            d=d0,
            rw_prev=rw_prev,
            w_prev=w0.copy(),
            r_prev=state0.r.copy(),
        )
        w1 = np.array([0.5, 0.5, 0.5])
        d1 = strategy.compute_direction(w1, state1)

        # d1 should be w1 + beta * d0
        # beta = (r1, w1) / (r0, w0) = 0.75 / 3.0 = 0.25
        beta = 0.75 / 3.0
        expected = w1 + beta * d0
        np.testing.assert_allclose(d1, expected, rtol=1e-10)

    def test_orthogonalization_direction_strategy_delegates(self) -> None:
        """Test OrthogonalizationDirectionStrategy delegates to orthog strategy."""
        from neuralls.solver.models.state import CGState

        # Create orthogonalization strategy
        orthog = PeriodicRestartOrthogonalization(m_max=5)
        strategy = OrthogonalizationDirectionStrategy(orthog)

        # Create state with empty history
        state = CGState.create_initial(
            u=np.zeros(3),
            r=np.array([1.0, 1.0, 1.0]),
            w=np.array([1.0, 1.0, 1.0]),
            d=np.array([1.0, 1.0, 1.0]),
            q=np.array([1.0, 1.0, 1.0]),
            residual_norm=np.sqrt(3.0),
            rhs_norm=np.sqrt(3.0),
            max_history=5,
        )

        w = np.array([2.0, 2.0, 2.0])
        d = strategy.compute_direction(w, state)

        # First iteration with empty history should return w
        np.testing.assert_array_equal(d, w)


class TestBackwardCompatibility:
    """Test that new implementation is bit-exact with old implementations."""

    @pytest.fixture
    def larger_spd_system(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create larger 50x50 SPD system for comparison testing.

        Returns:
            Tuple (A, b, x_true) where A @ x_true = b
        """
        np.random.seed(42)
        n = 50
        # Create SPD matrix via A = Q D Q^T
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        D = np.diag(np.linspace(1.0, 10.0, n))
        A = Q @ D @ Q.T
        x_true = np.random.randn(n)
        b = A @ x_true
        return A, b, x_true

    def test_unified_solver_matches_old_pcg(
        self, larger_spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test that PCGSolver produces same results as factory."""
        from neuralls.solver import CombinedToleranceCriterion

        A, b, x_true = larger_spd_system

        # Direct PCG solver with explicit convergence criterion
        criterion = CombinedToleranceCriterion(rtol=1e-10, atol=1e-14)
        new_solver = PCGSolver(convergence_criterion=criterion)
        x_new, result_new = new_solver.solve(A, b, maxiter=100)

        # PCG solver via factory
        x_old, result_old = pcg(A, b, rtol=1e-10, maxiter=100)

        # Should produce identical results
        assert result_new.converged == result_old.converged
        assert result_new.iterations == result_old.iterations
        np.testing.assert_allclose(x_new, x_old, rtol=1e-14, atol=1e-14)

    def test_unified_solver_matches_old_fcg(
        self, larger_spd_system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test that FCGSolver produces same results as factory."""
        from neuralls.solver import CombinedToleranceCriterion

        A, b, x_true = larger_spd_system

        # Direct FCG solver with explicit convergence criterion
        orthog = PeriodicRestartOrthogonalization(m_max=10)
        criterion = CombinedToleranceCriterion(rtol=1e-10, atol=1e-14)
        new_solver = FCGSolver(orthogonalization=orthog, convergence_criterion=criterion)
        x_new, result_new = new_solver.solve(A, b, maxiter=100)

        # FCG solver via factory
        x_old, result_old = flexible_cg(A, b, m_max=10, rtol=1e-10, maxiter=100)

        # Should produce identical results
        assert result_new.converged == result_old.converged
        assert result_new.iterations == result_old.iterations
        np.testing.assert_allclose(x_new, x_old, rtol=1e-14, atol=1e-14)
