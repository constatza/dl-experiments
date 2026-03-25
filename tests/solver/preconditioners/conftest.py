"""Shared fixtures for preconditioner tests.

This module provides fixtures specific to preconditioner testing.
General solver fixtures remain in tests/solver/conftest.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
