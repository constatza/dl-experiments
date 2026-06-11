"""Test bind_inputs API and extra_input_names on base Preconditioner class."""

import numpy as np
import pytest

from neuralls.domain.solver.preconditioners.implementations.jacobi import (
    JacobiPreconditioner,
)


@pytest.fixture
def small_diagonal_matrix():
    """Small diagonal matrix for testing."""
    return np.diag([2.0, 4.0, 8.0])


@pytest.fixture
def jacobi(small_diagonal_matrix):
    """Jacobi preconditioner instance."""
    return JacobiPreconditioner(small_diagonal_matrix)


def test_extra_input_names_default_empty(jacobi):
    """Default extra_input_names should be empty tuple."""
    assert jacobi.extra_input_names == ()


def test_bind_inputs_noop_on_standard_preconditioners(jacobi):
    """bind_inputs should be no-op and not affect apply behavior."""
    jacobi.bind_inputs(matrix=np.eye(3), coords=np.zeros((3, 2)))
    r = np.array([2.0, 4.0, 8.0])
    result = jacobi.apply(r)
    np.testing.assert_allclose(result, np.array([1.0, 1.0, 1.0]))


def test_bind_inputs_returns_none(jacobi):
    """bind_inputs should return None."""
    assert jacobi.bind_inputs(matrix=np.eye(3)) is None
