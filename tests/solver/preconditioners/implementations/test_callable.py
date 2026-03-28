"""Tests for CallablePreconditioner wrapper.

Tests the adapter that wraps arbitrary callables as preconditioners.
"""

from __future__ import annotations

import numpy as np

from neuralls.domain.solver.preconditioners import CallablePreconditioner


def test_callable_preconditioner_wraps_function() -> None:
    """Verify CallablePreconditioner wraps arbitrary function."""

    def custom_precond(r):
        return r * 0.5  # Simple damping

    precond = CallablePreconditioner(custom_precond)
    r = np.array([2.0, 4.0, 6.0])
    z = precond.apply(r)

    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(z, expected)
