"""Shared fixtures for generation strategy tests."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.generation.interfaces import TracingSolverCallable


@pytest.fixture
def spd_matrix() -> np.ndarray:
    """10x10 symmetric positive-definite matrix."""
    n = 10
    A = np.random.randn(n, n)
    return A.T @ A + np.eye(n)


@pytest.fixture
def sample_matrix(spd_matrix: np.ndarray) -> np.ndarray:
    """Alias for spd_matrix, used in validated_archive tests."""
    return spd_matrix


@pytest.fixture
def residual_solver() -> TracingSolverCallable:
    """Default tracing solver for residuals / residual_traces strategies."""
    from neuralls.generation.config_processing import _make_residual_solver

    return _make_residual_solver()


@pytest.fixture
def direction_solver() -> TracingSolverCallable:
    """Default tracing solver for search_directions strategy."""
    from neuralls.generation.config_processing import _make_direction_solver

    return _make_direction_solver()
