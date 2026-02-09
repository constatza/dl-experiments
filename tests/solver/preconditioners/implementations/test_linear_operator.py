"""Tests for LinearOperatorPreconditioner wrapper.

Tests the adapter that wraps scipy LinearOperators as preconditioners.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator

from neuralls.solver.preconditioners import LinearOperatorPreconditioner


def test_linear_operator_preconditioner_wraps_scipy() -> None:
    """Verify LinearOperatorPreconditioner wraps SciPy LinearOperator."""
    # Create simple diagonal operator
    n = 3
    diag = np.array([2.0, 4.0, 1.0])
    M_op = LinearOperator((n, n), matvec=lambda r: r / diag)

    precond = LinearOperatorPreconditioner(M_op)
    r = np.array([2.0, 4.0, 1.0])
    z = precond.apply(r)

    # Expected: z = r / diag = [2/2, 4/4, 1/1] = [1, 1, 1]
    expected = np.array([1.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(z, expected)
