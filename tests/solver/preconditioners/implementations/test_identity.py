"""Tests for Identity preconditioner.

Tests the noop Identity preconditioner to ensure:
- Returns copy of input (not reference)
- Doesn't modify input values
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neuralls.solver.preconditioners import Identity

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def test_identity_preconditioner_is_noop(
    identity_preconditioner: Callable[[NDArray, object], NDArray],
    rhs_ones_small: NDArray,
) -> None:
    """Verify identity preconditioner returns copy of input.

    Theory:
        Identity preconditioner: M = I, so z = M^{-1}r = r.
        Should return exact copy without modification.
    """
    r = rhs_ones_small.copy()
    z = identity_preconditioner(r, None)

    # Verify z == r
    np.testing.assert_array_equal(z, r)

    # Verify it's a copy (not same object)
    assert z is not r


def test_identity_preconditioner_class() -> None:
    """Verify Identity preconditioner returns copy of residual."""
    precond = Identity()
    r = np.array([1.0, 2.0, 3.0])
    z = precond.apply(r)

    # Should return copy, not reference
    assert np.array_equal(z, r)
    assert z is not r  # Different object

    # Modify original should not affect result
    r[0] = 999.0
    assert z[0] == 1.0
