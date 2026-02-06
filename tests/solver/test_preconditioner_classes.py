"""Tests for class-based preconditioner architecture.

This module tests the preconditioner classes with new ABC-based hierarchy:
- Identity, JacobiPreconditioner, ILUPreconditioner (direct instantiation)
- ScheduledPreconditioner (contextual)
- Preconditioners now accept matrix and compute what they need internally
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.solver.preconditioners import (
    Identity,
    ILUPreconditioner,
    JacobiPreconditioner,
    PreconditionerContext,
    ScheduledPreconditioner,
)


def test_identity_preconditioner() -> None:
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


def test_jacobi_preconditioner_from_matrix() -> None:
    """Verify Jacobi preconditioner computes diagonal inverse from matrix."""
    # Simple diagonal matrix
    A = np.diag([2.0, 4.0, 1.0])

    # Pass matrix - preconditioner computes diagonal inverse internally
    precond = JacobiPreconditioner(A)

    r = np.array([2.0, 4.0, 1.0])
    z = precond.apply(r)

    # Expected: z_i = r_i / A_ii = [2/2, 4/4, 1/1] = [1.0, 1.0, 1.0]
    expected = np.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(z, expected)


def test_jacobi_preconditioner_with_tridiagonal(tridiagonal_spd_small) -> None:
    """Verify Jacobi preconditioner from actual SPD matrix."""
    A = tridiagonal_spd_small

    # Pass matrix directly
    precond = JacobiPreconditioner(A)

    r = np.ones(A.shape[0])
    z = precond.apply(r)

    # z_i = r_i / A_ii for diagonal scaling
    expected = r / np.diag(A)
    np.testing.assert_allclose(z, expected)


def test_jacobi_handles_near_zero_diagonal() -> None:
    """Verify Jacobi preconditioner protects against near-zero diagonals."""
    # Matrix with near-zero diagonal element
    A = np.diag([2.0, 1e-15, 1.0])

    precond = JacobiPreconditioner(A)
    r = np.array([2.0, 4.0, 1.0])
    z = precond.apply(r)

    # Near-zero diagonal should be treated as 1.0
    # Expected: [2/2, 4/1, 1/1] = [1.0, 4.0, 1.0]
    expected = np.array([1.0, 4.0, 1.0])
    np.testing.assert_allclose(z, expected)


def test_ilu_preconditioner_from_matrix() -> None:
    """Verify ILU preconditioner computes factorization from matrix."""
    # Simple SPD tridiagonal matrix
    A = np.array([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 4.0]])

    # Pass matrix directly - preconditioner computes ILU internally
    precond = ILUPreconditioner(A)

    r = np.array([1.0, 2.0, 3.0])
    z = precond.apply(r)

    # Verify output is reasonable (not NaN/Inf)
    assert np.all(np.isfinite(z))
    assert z.shape == r.shape

    # ILU should approximate A^{-1}, so A @ z ≈ r
    result = A @ z
    np.testing.assert_allclose(result, r, rtol=1e-2)  # Approximate


def test_ilu_preconditioner_with_larger_matrix() -> None:
    """Verify ILU with larger tridiagonal matrix."""
    # 5x5 tridiagonal matrix
    n = 5
    A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)

    precond = ILUPreconditioner(A)
    r = np.ones(n)
    z = precond.apply(r)

    # Verify output is reasonable
    assert np.all(np.isfinite(z))
    assert z.shape == (n,)


def test_scheduled_preconditioner_limit_iters() -> None:
    """Verify ScheduledPreconditioner switches at limit_iters."""
    A = np.diag([2.0, 2.0])
    primary = JacobiPreconditioner(A)
    fallback = Identity()

    scheduled = ScheduledPreconditioner(primary, fallback, limit_iters=5)

    r = np.array([2.0, 4.0])

    # Before limit: use primary (Jacobi: z = r / diag = [2/2, 4/2] = [1, 2])
    ctx = PreconditionerContext(iteration=3, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(r, ctx)
    np.testing.assert_allclose(z, [1.0, 2.0])  # Primary (Jacobi)

    # At limit: switch to fallback (Identity: z = r = [2, 4])
    ctx = PreconditionerContext(iteration=5, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(r, ctx)
    np.testing.assert_allclose(z, [2.0, 4.0])  # Fallback (Identity)

    # After limit: still fallback
    ctx = PreconditionerContext(iteration=10, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(r, ctx)
    np.testing.assert_allclose(z, [2.0, 4.0])  # Fallback (Identity)


def test_scheduled_preconditioner_apply_every() -> None:
    """Verify ScheduledPreconditioner respects apply_every parameter."""
    A = np.diag([2.0, 2.0])
    primary = JacobiPreconditioner(A)
    fallback = Identity()

    scheduled = ScheduledPreconditioner(
        primary, fallback, limit_iters=None, apply_every=3
    )

    r = np.array([2.0, 4.0])

    # Iteration 0: apply primary (0 % 3 == 0)
    ctx = PreconditionerContext(iteration=0, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(r, ctx)
    np.testing.assert_allclose(z, [1.0, 2.0])  # Primary

    # Iteration 1: use fallback (1 % 3 != 0)
    ctx = PreconditionerContext(iteration=1, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(r, ctx)
    np.testing.assert_allclose(z, [2.0, 4.0])  # Fallback

    # Iteration 2: use fallback (2 % 3 != 0)
    ctx = PreconditionerContext(iteration=2, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(r, ctx)
    np.testing.assert_allclose(z, [2.0, 4.0])  # Fallback

    # Iteration 3: apply primary (3 % 3 == 0)
    ctx = PreconditionerContext(iteration=3, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(r, ctx)
    np.testing.assert_allclose(z, [1.0, 2.0])  # Primary


def test_scheduled_preconditioner_default_fallback() -> None:
    """Verify ScheduledPreconditioner uses Identity as default fallback."""
    A = np.diag([2.0, 2.0])
    primary = JacobiPreconditioner(A)

    # No fallback specified
    scheduled = ScheduledPreconditioner(primary, fallback=None, limit_iters=2)

    r = np.array([2.0, 4.0])

    # After limit: should use Identity (default fallback)
    ctx = PreconditionerContext(iteration=5, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(r, ctx)
    np.testing.assert_allclose(z, [2.0, 4.0])  # Identity fallback


def test_scheduled_preconditioner_combined_limits() -> None:
    """Verify combined limit_iters and apply_every scheduling."""
    A = np.diag([2.0, 2.0])
    primary = JacobiPreconditioner(A)
    fallback = Identity()

    scheduled = ScheduledPreconditioner(
        primary, fallback, limit_iters=6, apply_every=2
    )

    r = np.array([2.0, 4.0])

    # Iteration 0: primary (0 % 2 == 0 and 0 < 6)
    ctx = PreconditionerContext(iteration=0, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(r, ctx)
    np.testing.assert_allclose(z, [1.0, 2.0])

    # Iteration 1: fallback (1 % 2 != 0)
    ctx = PreconditionerContext(iteration=1, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(r, ctx)
    np.testing.assert_allclose(z, [2.0, 4.0])

    # Iteration 4: primary (4 % 2 == 0 and 4 < 6)
    ctx = PreconditionerContext(iteration=4, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(r, ctx)
    np.testing.assert_allclose(z, [1.0, 2.0])

    # Iteration 6: fallback (limit reached, even though 6 % 2 == 0)
    ctx = PreconditionerContext(iteration=6, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(r, ctx)
    np.testing.assert_allclose(z, [2.0, 4.0])


def test_preconditioner_context_immutable() -> None:
    """Verify PreconditionerContext is immutable (frozen dataclass)."""
    ctx = PreconditionerContext(iteration=5, residual_norm=1.0, rhs_norm=2.0)

    # Should not be able to modify
    with pytest.raises(AttributeError):
        ctx.iteration = 10  # type: ignore[misc]

    with pytest.raises(AttributeError):
        ctx.residual_norm = 5.0  # type: ignore[misc]
