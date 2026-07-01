"""Tests for scheduled preconditioner functionality.

This module tests the ScheduledPreconditioner class which provides:
- limit_iters: switch to fallback after N iterations

The apply_every and first_n parameters have been removed as redundant.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from neuralls.domain.solver.preconditioners import (
    Identity,
    JacobiPreconditioner,
    PreconditionerContext,
    ScheduledPreconditioner,
)


@pytest.fixture
def diagonal_matrix() -> NDArray:
    """Provide a diagonal matrix for Jacobi schedule tests."""
    return np.diag([2.0, 2.0])


@pytest.fixture
def residual_vector() -> NDArray:
    """Provide a residual vector for schedule tests."""
    return np.array([2.0, 4.0])


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


def test_scheduled_preconditioner_no_limit() -> None:
    """Verify ScheduledPreconditioner with no limit always uses primary."""
    A = np.diag([2.0, 2.0])
    primary = JacobiPreconditioner(A)
    fallback = Identity()

    scheduled = ScheduledPreconditioner(primary, fallback, limit_iters=None)

    r = np.array([2.0, 4.0])

    # All iterations should use primary when no limit set
    for i in [0, 5, 10, 100]:
        ctx = PreconditionerContext(iteration=i, residual_norm=1.0, rhs_norm=1.0)
        z = scheduled.apply(r, ctx)
        np.testing.assert_allclose(z, [1.0, 2.0])  # Primary (Jacobi)


def test_scheduled_preconditioner_requires_context() -> None:
    """Verify ScheduledPreconditioner raises error when context is None."""
    A = np.diag([2.0, 2.0])
    primary = JacobiPreconditioner(A)

    scheduled = ScheduledPreconditioner(primary, limit_iters=5)

    r = np.array([2.0, 4.0])

    # Should raise ValueError if context is None
    with pytest.raises(ValueError, match="requires context"):
        scheduled.apply(r, context=None)


def test_preconditioner_context_immutable() -> None:
    """Verify PreconditionerContext is immutable (frozen dataclass)."""
    ctx = PreconditionerContext(iteration=5, residual_norm=1.0, rhs_norm=2.0)

    # Should not be able to modify
    with pytest.raises(AttributeError):
        ctx.iteration = 10  # type: ignore[misc]

    with pytest.raises(AttributeError):
        ctx.residual_norm = 5.0  # type: ignore[misc]


def test_scheduled_preconditioner_iteration_zero() -> None:
    """Test that iteration=0 uses primary preconditioner when limit_iters > 0."""
    A = np.diag([2.0, 2.0])
    primary = JacobiPreconditioner(A)
    fallback = Identity()

    scheduled = ScheduledPreconditioner(primary, fallback, limit_iters=5)

    r = np.array([2.0, 4.0])

    # Iteration 0 should use primary
    ctx = PreconditionerContext(iteration=0, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(r, ctx)
    np.testing.assert_allclose(z, [1.0, 2.0])  # Primary (Jacobi)


def test_scheduled_preconditioner_uses_fallback_before_start_iter(
    diagonal_matrix: NDArray,
    residual_vector: NDArray,
) -> None:
    """Verify delayed schedules use fallback before start_iter."""
    primary = JacobiPreconditioner(diagonal_matrix)
    fallback = Identity()
    scheduled = ScheduledPreconditioner(
        primary,
        fallback,
        limit_iters=None,
        start_iter=3,
    )

    ctx = PreconditionerContext(iteration=2, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(residual_vector, ctx)

    np.testing.assert_allclose(z, residual_vector)


def test_scheduled_preconditioner_starts_primary_at_start_iter(
    diagonal_matrix: NDArray,
    residual_vector: NDArray,
) -> None:
    """Verify delayed schedules activate primary at start_iter."""
    primary = JacobiPreconditioner(diagonal_matrix)
    fallback = Identity()
    scheduled = ScheduledPreconditioner(
        primary,
        fallback,
        limit_iters=None,
        start_iter=3,
    )

    ctx = PreconditionerContext(iteration=3, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(residual_vector, ctx)

    np.testing.assert_allclose(z, [1.0, 2.0])


def test_scheduled_preconditioner_delayed_limit_switches_back_to_fallback(
    diagonal_matrix: NDArray,
    residual_vector: NDArray,
) -> None:
    """Verify limit_iters is measured from start_iter."""
    primary = JacobiPreconditioner(diagonal_matrix)
    fallback = Identity()
    scheduled = ScheduledPreconditioner(
        primary,
        fallback,
        limit_iters=2,
        start_iter=3,
    )

    active_ctx = PreconditionerContext(iteration=4, residual_norm=1.0, rhs_norm=1.0)
    inactive_ctx = PreconditionerContext(iteration=5, residual_norm=1.0, rhs_norm=1.0)

    np.testing.assert_allclose(scheduled.apply(residual_vector, active_ctx), [1.0, 2.0])
    np.testing.assert_allclose(scheduled.apply(residual_vector, inactive_ctx), residual_vector)


def test_scheduled_preconditioner_rejects_negative_start_iter() -> None:
    """Verify delayed schedules reject negative activation iterations."""
    with pytest.raises(ValueError, match="start_iter"):
        ScheduledPreconditioner(Identity(), start_iter=-1)
