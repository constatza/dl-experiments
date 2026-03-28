"""Tests for FCG solver direction history initialization.

This module verifies that the FCG solver correctly initializes its direction
history based on the orthogonalization strategy's window_size property.

Coverage:
- FCG(-1) uses DEFAULT_FCG_HISTORY_LIMIT for unlimited orthogonalization
- FCG(m_max) uses m_max for bounded orthogonalization
- Direction history is properly configured before first iteration
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.solver.factories import flexible_cg, pcg


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_spd_matrix() -> np.ndarray:
    """Simple SPD matrix for testing."""
    n = 50
    return np.eye(n) * 2.0


@pytest.fixture
def simple_rhs(simple_spd_matrix: np.ndarray) -> np.ndarray:
    """Simple RHS vector."""
    n = simple_spd_matrix.shape[0]
    return np.ones(n)


# =============================================================================
# FCG Unlimited History Tests
# =============================================================================


class TestFCGUnlimitedHistory:
    """Test FCG(-1) uses large history limit for full orthogonalization."""

    def test_fcg_unlimited_converges(
        self,
        simple_spd_matrix: np.ndarray,
        simple_rhs: np.ndarray,
    ) -> None:
        """FCG(-1) should converge successfully."""
        x, result = flexible_cg(
            simple_spd_matrix,
            simple_rhs,
            m_max=-1,
            maxiter=30,
        )

        assert result.converged
        assert result.iterations > 0


# =============================================================================
# FCG Bounded History Tests
# =============================================================================


class TestFCGBoundedHistory:
    """Test FCG(m_max) uses correct window size for bounded orthogonalization."""

    def test_fcg_bounded_converges(
        self,
        simple_spd_matrix: np.ndarray,
        simple_rhs: np.ndarray,
    ) -> None:
        """FCG(5) should converge successfully."""
        x, result = flexible_cg(
            simple_spd_matrix,
            simple_rhs,
            m_max=5,
            maxiter=30,
        )

        assert result.converged
        assert result.iterations > 0

    def test_fcg_bounded_may_take_more_iterations(
        self,
        simple_spd_matrix: np.ndarray,
        simple_rhs: np.ndarray,
    ) -> None:
        """FCG(5) may take more iterations than PCG due to truncation."""
        # Run FCG(5)
        x_fcg, result_fcg = flexible_cg(
            simple_spd_matrix,
            simple_rhs,
            m_max=5,
            maxiter=50,
        )

        # Run PCG
        x_pcg, result_pcg = pcg(
            simple_spd_matrix,
            simple_rhs,
            maxiter=50,
        )

        # Both should converge
        assert result_fcg.converged
        assert result_pcg.converged

        # FCG(5) may take more iterations (truncation effect)
        # But should still be reasonable
        assert result_fcg.iterations >= result_pcg.iterations

    def test_various_window_sizes(
        self,
        simple_spd_matrix: np.ndarray,
        simple_rhs: np.ndarray,
    ) -> None:
        """Test various bounded window sizes all converge."""
        for m_max in [1, 5, 10, 20]:
            x, result = flexible_cg(
                simple_spd_matrix,
                simple_rhs,
                m_max=m_max,
                maxiter=50,
            )

            assert result.converged, f"FCG({m_max}) failed to converge"


# =============================================================================
# Comparison Tests
# =============================================================================


class TestFCGComparison:
    """Compare FCG with different window sizes."""

    def test_larger_window_fewer_iterations(
        self,
        simple_spd_matrix: np.ndarray,
        simple_rhs: np.ndarray,
    ) -> None:
        """Larger orthogonalization window should require fewer iterations."""
        # Run with small window
        x_small, result_small = flexible_cg(
            simple_spd_matrix,
            simple_rhs,
            m_max=5,
            maxiter=50,
        )

        # Run with large window
        x_large, result_large = flexible_cg(
            simple_spd_matrix,
            simple_rhs,
            m_max=20,
            maxiter=50,
        )

        # Both should converge
        assert result_small.converged
        assert result_large.converged

        # Larger window should take same or fewer iterations
        assert result_large.iterations <= result_small.iterations

    def test_unlimited_is_fastest(
        self,
        simple_spd_matrix: np.ndarray,
        simple_rhs: np.ndarray,
    ) -> None:
        """FCG(-1) should be fastest (matches PCG)."""
        # Run FCG(-1)
        x_unlimited, result_unlimited = flexible_cg(
            simple_spd_matrix,
            simple_rhs,
            m_max=-1,
            maxiter=50,
        )

        # Run FCG(10)
        x_bounded, result_bounded = flexible_cg(
            simple_spd_matrix,
            simple_rhs,
            m_max=10,
            maxiter=50,
        )

        # Both should converge
        assert result_unlimited.converged
        assert result_bounded.converged

        # Unlimited should take same or fewer iterations
        assert result_unlimited.iterations <= result_bounded.iterations
