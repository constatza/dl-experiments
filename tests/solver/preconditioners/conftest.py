"""Shared fixtures for preconditioner tests.

This module provides fixtures specific to preconditioner testing.
General solver fixtures remain in tests/solver/conftest.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from neuralls.domain.solver.preconditioners.implementations.jacobi import JacobiPreconditioner
from neuralls.domain.solver.preconditioners.ports import ExtraInputPredictorPort, PredictorPort

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture
def small_diagonal_matrix() -> NDArray:
    """Small 3x3 diagonal matrix for base/bind_inputs tests.

    Returns:
        3x3 diagonal matrix with entries [2, 4, 8]
    """
    return np.diag([2.0, 4.0, 8.0])


@pytest.fixture
def jacobi(small_diagonal_matrix: NDArray) -> JacobiPreconditioner:
    """JacobiPreconditioner built from small_diagonal_matrix.

    Args:
        small_diagonal_matrix: 3x3 diagonal matrix fixture

    Returns:
        JacobiPreconditioner instance
    """
    return JacobiPreconditioner(small_diagonal_matrix)


@pytest.fixture
def well_conditioned_matrix() -> NDArray:
    """Well-conditioned 4x4 SPD matrix for testing.

    Returns:
        4x4 diagonal matrix with condition number ~2
    """
    return np.diag([4.0, 3.0, 2.0, 2.0])


@pytest.fixture
def dense_spd_matrix() -> NDArray:
    """Dense SPD matrix for ILU testing.

    Returns:
        5x5 dense SPD tridiagonal matrix
    """
    n = 5
    A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    return A


@pytest.fixture
def residual_vector() -> NDArray:
    """Test residual vector.

    Returns:
        4D residual vector
    """
    return np.array([1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def residual() -> NDArray:
    """4-element residual vector for port contract tests.

    Returns:
        4D float64 residual vector of ones
    """
    return np.ones(4, dtype=np.float64)


class _MinimalPredictor(PredictorPort):
    """Minimal implementation of base PredictorPort — ignores extra inputs."""

    def apply(self, residual: NDArray, **extra_inputs: NDArray) -> NDArray:
        """Return a copy of the residual unchanged.

        Args:
            residual: Input residual vector.
            **extra_inputs: Accepted but ignored.

        Returns:
            Copy of the residual.
        """
        return residual.copy()

    def cleanup(self) -> None:
        """No-op cleanup."""
        pass


class _CapturingPredictor(ExtraInputPredictorPort):
    """ExtraInputPredictorPort that records extra inputs for assertion."""

    def __init__(self) -> None:
        """Initialize with empty extra inputs record."""
        self.last_extra: dict[str, NDArray] = {}

    def apply(self, residual: NDArray, **extra_inputs: NDArray) -> NDArray:
        """Record extra inputs and return a copy of the residual.

        Args:
            residual: Input residual vector.
            **extra_inputs: Named extra arrays to capture.

        Returns:
            Copy of the residual.
        """
        self.last_extra = extra_inputs
        return residual.copy()

    def cleanup(self) -> None:
        """No-op cleanup."""
        pass


@pytest.fixture
def capturing_predictor() -> _CapturingPredictor:
    """ExtraInputPredictorPort that captures extra inputs passed to apply().

    Returns:
        Fresh _CapturingPredictor instance
    """
    return _CapturingPredictor()


@pytest.fixture
def minimal_predictor() -> _MinimalPredictor:
    """Minimal PredictorPort implementation — passes residual through unchanged.

    Returns:
        Fresh _MinimalPredictor instance
    """
    return _MinimalPredictor()
