"""Tests for FCG orthogonalization factory function.

This module tests the create_fcg_orthogonalization() factory that maps
m_max parameter values to appropriate OrthogonalizationStrategy instances.

Coverage:
- m_max > 0: Returns PeriodicRestartOrthogonalization with finite window
- m_max = -1 or np.inf: Returns PeriodicRestartOrthogonalization with infinite window
- m_max = 0: Raises ValueError
- m_max < -1: Raises ValueError
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.solver.strategies.orthogonalization import (
    ModifiedGramSchmidt,
    PeriodicRestartOrthogonalization,
    TruncatedGramSchmidt,
    create_fcg_orthogonalization,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_vector() -> np.ndarray:
    """Sample vector for orthogonalization tests."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def p_vectors_history() -> list[np.ndarray]:
    """Sample history of p vectors."""
    return [
        np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    ]


@pytest.fixture
def q_vectors_history() -> list[np.ndarray]:
    """Sample history of q vectors (A @ p)."""
    return [
        np.array([2.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 2.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 2.0, 0.0, 0.0]),
    ]


# =============================================================================
# Factory Return Type Tests
# =============================================================================


class TestFactoryReturnTypes:
    """Test that factory returns correct strategy types."""

    def test_m_max_positive_returns_periodic_restart(self) -> None:
        """m_max > 0 should return PeriodicRestartOrthogonalization."""
        strategy = create_fcg_orthogonalization(m_max=10)
        assert isinstance(strategy, PeriodicRestartOrthogonalization)

    def test_m_max_one_returns_periodic_restart(self) -> None:
        """m_max = 1 should return PeriodicRestartOrthogonalization."""
        strategy = create_fcg_orthogonalization(m_max=1)
        assert isinstance(strategy, PeriodicRestartOrthogonalization)

    def test_m_max_negative_one_returns_periodic_with_inf(self) -> None:
        """m_max = -1 should return PeriodicRestartOrthogonalization with np.inf."""
        strategy = create_fcg_orthogonalization(m_max=-1)
        assert isinstance(strategy, PeriodicRestartOrthogonalization)
        assert np.isinf(strategy.m_max)
        assert strategy.window_size is None

    def test_m_max_inf_returns_periodic_with_inf(self) -> None:
        """m_max = np.inf should return PeriodicRestartOrthogonalization."""
        strategy = create_fcg_orthogonalization(m_max=np.inf)
        assert isinstance(strategy, PeriodicRestartOrthogonalization)
        assert np.isinf(strategy.m_max)
        assert strategy.window_size is None

    def test_default_m_max_returns_periodic_restart(self) -> None:
        """Default m_max should return PeriodicRestartOrthogonalization."""
        strategy = create_fcg_orthogonalization()
        assert isinstance(strategy, PeriodicRestartOrthogonalization)


# =============================================================================
# Factory Validation Tests
# =============================================================================


class TestFactoryValidation:
    """Test factory input validation."""

    def test_m_max_zero_raises_value_error(self) -> None:
        """m_max = 0 should raise ValueError."""
        with pytest.raises(ValueError, match="m_max cannot be 0"):
            create_fcg_orthogonalization(m_max=0)

    def test_m_max_less_than_negative_one_raises_value_error(self) -> None:
        """m_max < -1 should raise ValueError."""
        with pytest.raises(ValueError, match="m_max must be >= -1"):
            create_fcg_orthogonalization(m_max=-2)

        with pytest.raises(ValueError, match="m_max must be >= -1"):
            create_fcg_orthogonalization(m_max=-10)


# =============================================================================
# Strategy Configuration Tests
# =============================================================================


class TestStrategyConfiguration:
    """Test that factory configures strategies correctly."""

    def test_periodic_restart_has_correct_m_max(self) -> None:
        """PeriodicRestartOrthogonalization should have correct m_max."""
        strategy = create_fcg_orthogonalization(m_max=5)
        assert isinstance(strategy, PeriodicRestartOrthogonalization)
        assert strategy.m_max == 5

    def test_periodic_restart_various_m_max_values(self) -> None:
        """Test various m_max values for PeriodicRestartOrthogonalization."""
        for m in [1, 2, 5, 10, 50, 100]:
            strategy = create_fcg_orthogonalization(m_max=m)
            assert isinstance(strategy, PeriodicRestartOrthogonalization)
            assert strategy.m_max == m


# =============================================================================
# Integration Tests
# =============================================================================


class TestFactoryIntegration:
    """Integration tests verifying factory-created strategies work correctly."""

    def test_periodic_restart_strategy_orthogonalizes(
        self,
        sample_vector: np.ndarray,
        p_vectors_history: list[np.ndarray],
        q_vectors_history: list[np.ndarray],
    ) -> None:
        """Factory-created PeriodicRestartOrthogonalization should orthogonalize."""
        strategy = create_fcg_orthogonalization(m_max=10)

        result, report = strategy.orthogonalize(
            sample_vector, p_vectors_history, q_vectors_history
        )

        # Result should differ from input (orthogonalization applied)
        assert not np.allclose(result, sample_vector)
        assert not report.breakdown
        assert len(report.coefficients) > 0

    def test_infinite_orthogonalization_strategy_orthogonalizes(
        self,
        sample_vector: np.ndarray,
        p_vectors_history: list[np.ndarray],
        q_vectors_history: list[np.ndarray],
    ) -> None:
        """Factory-created FCG(∞) should orthogonalize against all history."""
        strategy = create_fcg_orthogonalization(m_max=-1)

        result, report = strategy.orthogonalize(
            sample_vector, p_vectors_history, q_vectors_history
        )

        # Result should differ from input (orthogonalization applied)
        assert not np.allclose(result, sample_vector)
        assert not report.breakdown
        # Should orthogonalize against all history
        assert len(report.coefficients) == len(p_vectors_history)

    def test_empty_history_returns_unchanged_vector(
        self,
        sample_vector: np.ndarray,
    ) -> None:
        """With empty history, orthogonalization should return unchanged vector."""
        strategy = create_fcg_orthogonalization(m_max=10)

        result, report = strategy.orthogonalize(sample_vector, [], [])

        np.testing.assert_allclose(result, sample_vector)
        assert len(report.coefficients) == 0
        assert not report.breakdown


# =============================================================================
# Window Size Property Tests
# =============================================================================


class TestWindowSizeProperty:
    """Test that all strategies expose window_size property correctly."""

    def test_periodic_restart_window_size(self) -> None:
        """PeriodicRestartOrthogonalization should return m_max as window_size."""
        strategy = PeriodicRestartOrthogonalization(m_max=15)
        assert strategy.window_size == 15

    def test_truncated_gram_schmidt_window_size(self) -> None:
        """TruncatedGramSchmidt should return configured window_size."""
        strategy = TruncatedGramSchmidt(window_size=20)
        assert strategy.window_size == 20

    def test_modified_gram_schmidt_window_size(self) -> None:
        """ModifiedGramSchmidt should return None for unlimited."""
        strategy = ModifiedGramSchmidt()
        assert strategy.window_size is None

    def test_periodic_restart_inf_window_size(self) -> None:
        """PeriodicRestartOrthogonalization with m_max=np.inf should return None."""
        strategy = PeriodicRestartOrthogonalization(m_max=np.inf)
        assert strategy.window_size is None

    def test_factory_bounded_window_size(self) -> None:
        """Factory-created bounded strategy should have correct window_size."""
        strategy = create_fcg_orthogonalization(m_max=10)
        assert strategy.window_size == 10

    def test_factory_unlimited_window_size(self) -> None:
        """Factory-created unlimited strategy should have None window_size."""
        strategy = create_fcg_orthogonalization(m_max=-1)
        assert strategy.window_size is None

    def test_various_window_sizes(self) -> None:
        """Test various window_size values through factory."""
        for m in [1, 5, 10, 50, 100]:
            strategy = create_fcg_orthogonalization(m_max=m)
            assert strategy.window_size == m
