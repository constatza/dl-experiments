"""Tests for scheduled preconditioner functionality.

This module tests the ScheduledPreconditioner class which provides:
- limit_iters: switch to fallback after N iterations
- apply_every: apply primary every N iterations
- Combined scheduling strategies

Tests cover both the class-based ScheduledPreconditioner and the legacy
_scheduled_preconditioner function wrapper.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.solver.comparison import _scheduled_preconditioner
from neuralls.solver.models.result import IterationContext
from neuralls.solver.preconditioners import (
    Identity,
    JacobiPreconditioner,
    PreconditionerContext,
    ScheduledPreconditioner,
)


# =============================================================================
# ScheduledPreconditioner Class Tests
# =============================================================================


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


# =============================================================================
# Legacy _scheduled_preconditioner Function Tests
# =============================================================================


def test_legacy_scheduled_preconditioner_limit_iters() -> None:
    """Verify limit_iters switches to fallback after N iterations."""
    call_log: list[str] = []

    def main_precond(r: np.ndarray) -> np.ndarray:
        call_log.append("main")
        return r * 2.0

    def fallback(r: np.ndarray) -> np.ndarray:
        call_log.append("fallback")
        return r * 0.5

    wrapped = _scheduled_preconditioner(
        main_precond,
        fallback=fallback,
        limit_iters=5,
        apply_every=1,
        first_n=None,
    )

    residual = np.array([1.0, 2.0, 3.0])

    # Iterations 0-4: use main (result = residual * 2.0)
    for i in range(5):
        ctx = IterationContext(
            iteration=i,
            residual=residual,
            solution=np.zeros(3),
            matrix=np.eye(3),
            rhs=residual,
        )
        result = wrapped(ctx)
        assert call_log[-1] == "main", f"Iteration {i} should use main"
        np.testing.assert_array_equal(result, residual * 2.0)

    # Iterations 5+: use fallback (result = residual * 0.5)
    for i in range(5, 10):
        ctx = IterationContext(
            iteration=i,
            residual=residual,
            solution=np.zeros(3),
            matrix=np.eye(3),
            rhs=residual,
        )
        result = wrapped(ctx)
        assert call_log[-1] == "fallback", f"Iteration {i} should use fallback"
        np.testing.assert_array_equal(result, residual * 0.5)


def test_legacy_scheduled_preconditioner_wrong_type_raises() -> None:
    """Passing wrong type should raise AttributeError (no 'iteration' or 'residual' attr)."""

    def main_precond(r: np.ndarray) -> np.ndarray:
        return r

    wrapped = _scheduled_preconditioner(
        main_precond,
        fallback=None,
        limit_iters=5,
        apply_every=1,
        first_n=None,
    )

    # Passing int instead of IterationContext should raise AttributeError
    with pytest.raises(AttributeError):
        wrapped(5)  # type: ignore[arg-type]

    # Passing dict should also raise AttributeError
    with pytest.raises(AttributeError):
        wrapped({"iteration": 3})  # type: ignore[arg-type]


def test_legacy_scheduled_preconditioner_apply_every() -> None:
    """Test apply_every parameter correctly skips iterations."""
    call_log: list[str] = []

    def main_precond(r: np.ndarray) -> np.ndarray:
        call_log.append("main")
        return r * 2.0

    def fallback(r: np.ndarray) -> np.ndarray:
        call_log.append("fallback")
        return r * 0.5

    wrapped = _scheduled_preconditioner(
        main_precond,
        fallback=fallback,
        limit_iters=None,
        apply_every=3,  # Apply main every 3rd iteration
        first_n=None,
    )

    residual = np.array([1.0, 2.0, 3.0])

    # Iterations: 0, 3, 6, 9, ... should use main
    # Others should use fallback
    for i in range(10):
        ctx = IterationContext(
            iteration=i,
            residual=residual,
            solution=np.zeros(3),
            matrix=np.eye(3),
            rhs=residual,
        )
        result = wrapped(ctx)

        if i % 3 == 0:
            assert call_log[-1] == "main", (
                f"Iteration {i} (divisible by 3) should use main"
            )
            np.testing.assert_array_equal(result, residual * 2.0)
        else:
            assert call_log[-1] == "fallback", (
                f"Iteration {i} (not divisible by 3) should use fallback"
            )
            np.testing.assert_array_equal(result, residual * 0.5)


def test_legacy_scheduled_preconditioner_first_n() -> None:
    """Test first_n parameter limits main preconditioner to first N iterations."""
    call_log: list[str] = []

    def main_precond(r: np.ndarray) -> np.ndarray:
        call_log.append("main")
        return r * 2.0

    def fallback(r: np.ndarray) -> np.ndarray:
        call_log.append("fallback")
        return r * 0.5

    wrapped = _scheduled_preconditioner(
        main_precond,
        fallback=fallback,
        limit_iters=None,
        apply_every=1,
        first_n=3,  # Only use main for first 3 iterations
    )

    residual = np.array([1.0, 2.0, 3.0])

    # Iterations 0-2: use main
    for i in range(3):
        ctx = IterationContext(
            iteration=i,
            residual=residual,
            solution=np.zeros(3),
            matrix=np.eye(3),
            rhs=residual,
        )
        wrapped(ctx)
        assert call_log[-1] == "main", f"Iteration {i} < first_n=3 should use main"

    # Iterations 3+: use fallback
    for i in range(3, 10):
        ctx = IterationContext(
            iteration=i,
            residual=residual,
            solution=np.zeros(3),
            matrix=np.eye(3),
            rhs=residual,
        )
        wrapped(ctx)
        assert call_log[-1] == "fallback", (
            f"Iteration {i} >= first_n=3 should use fallback"
        )


def test_legacy_scheduled_preconditioner_iteration_zero() -> None:
    """Test that iteration=0 uses main preconditioner with first_n scheduling."""
    call_log: list[str] = []

    def main_precond(r: np.ndarray) -> np.ndarray:
        call_log.append("main")
        return r * 2.0

    def fallback(r: np.ndarray) -> np.ndarray:
        call_log.append("fallback")
        return r * 0.5

    # With first_n=5, iteration=0 should use main
    wrapped = _scheduled_preconditioner(
        main_precond,
        fallback=fallback,
        limit_iters=None,
        apply_every=1,
        first_n=5,
    )

    residual = np.array([1.0, 2.0, 3.0])
    ctx = IterationContext(
        iteration=0,
        residual=residual,
        solution=np.zeros(3),
        matrix=np.eye(3),
        rhs=residual,
    )

    result = wrapped(ctx)
    assert call_log[-1] == "main", "iteration=0 < first_n=5 should use main"
    np.testing.assert_array_equal(result, residual * 2.0)


def test_legacy_scheduled_preconditioner_combined_parameters() -> None:
    """Test combination of limit_iters, apply_every, and first_n parameters."""
    call_log: list[str] = []

    def main_precond(r: np.ndarray) -> np.ndarray:
        call_log.append("main")
        return r * 2.0

    def fallback(r: np.ndarray) -> np.ndarray:
        call_log.append("fallback")
        return r * 0.5

    # Complex scheduling: limit_iters=10, apply_every=2, first_n=3
    # Iteration 0: < first_n=3, divisible by 2, < limit_iters=10 → main
    # Iteration 1: < first_n=3, NOT divisible by 2 → fallback
    # Iteration 2: < first_n=3, divisible by 2, < limit_iters=10 → main
    # Iteration 3: >= first_n=3 → fallback
    # Iteration 4: >= first_n=3 → fallback
    # ...
    wrapped = _scheduled_preconditioner(
        main_precond,
        fallback=fallback,
        limit_iters=10,
        apply_every=2,
        first_n=3,
    )

    residual = np.array([1.0, 2.0, 3.0])

    expected = {
        0: "main",  # < first_n, divisible by 2, < limit
        1: "fallback",  # < first_n, NOT divisible by 2
        2: "main",  # < first_n, divisible by 2, < limit
        3: "fallback",  # >= first_n
        4: "fallback",  # >= first_n
        5: "fallback",  # >= first_n
        10: "fallback",  # >= limit_iters
        11: "fallback",  # >= limit_iters
    }

    for i, expected_call in expected.items():
        ctx = IterationContext(
            iteration=i,
            residual=residual,
            solution=np.zeros(3),
            matrix=np.eye(3),
            rhs=residual,
        )
        wrapped(ctx)
        assert call_log[-1] == expected_call, (
            f"Iteration {i} should use {expected_call}"
        )


def test_legacy_scheduled_preconditioner_no_fallback() -> None:
    """Test that when fallback is None, identity preconditioner is used."""
    call_log: list[str] = []

    def main_precond(r: np.ndarray) -> np.ndarray:
        call_log.append("main")
        return r * 2.0

    # No fallback specified
    wrapped = _scheduled_preconditioner(
        main_precond,
        fallback=None,  # Should default to identity
        limit_iters=2,
        apply_every=1,
        first_n=None,
    )

    residual = np.array([1.0, 2.0, 3.0])

    # Iteration 0-1: use main
    for i in range(2):
        ctx = IterationContext(
            iteration=i,
            residual=residual,
            solution=np.zeros(3),
            matrix=np.eye(3),
            rhs=residual,
        )
        result = wrapped(ctx)
        assert call_log[-1] == "main"
        np.testing.assert_array_equal(result, residual * 2.0)

    # Iteration 2+: use identity (returns residual unchanged)
    ctx = IterationContext(
        iteration=2,
        residual=residual,
        solution=np.zeros(3),
        matrix=np.eye(3),
        rhs=residual,
    )
    result = wrapped(ctx)
    # main_precond should not be called after limit_iters
    assert len(call_log) == 2  # Only 2 main calls from iterations 0-1
    np.testing.assert_array_equal(
        result, residual
    )  # Identity returns residual unchanged


def test_legacy_scheduled_preconditioner_iteration_is_int() -> None:
    """Test that iteration attribute must be an int for proper scheduling."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class BadContext:
        """Context with wrong iteration type."""

        iteration: str  # Should be int!
        residual: np.ndarray
        solution: np.ndarray
        matrix: np.ndarray
        rhs: np.ndarray

    def main_precond(r: np.ndarray) -> np.ndarray:
        return r * 2.0

    wrapped = _scheduled_preconditioner(
        main_precond,
        fallback=None,
        limit_iters=5,
        apply_every=1,
        first_n=None,
    )

    residual = np.array([1.0, 2.0])

    # BadContext with string iteration will cause TypeError in comparison
    bad_ctx = BadContext(
        iteration="five",  # type: ignore[arg-type]
        residual=residual,
        solution=np.zeros(2),
        matrix=np.eye(2),
        rhs=residual,
    )

    # The comparison "five" >= 5 will raise TypeError
    with pytest.raises(TypeError):
        wrapped(bad_ctx)  # type: ignore[arg-type]
