"""Unit tests for scheduled preconditioner functionality."""

from __future__ import annotations

import pytest
import numpy as np

from src.solver.comparison import _scheduled_preconditioner
from src.solver.info import IterationContext


def test_scheduled_preconditioner_limit_iters() -> None:
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


def test_scheduled_preconditioner_wrong_type_raises() -> None:
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


def test_scheduled_preconditioner_apply_every() -> None:
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


def test_scheduled_preconditioner_first_n() -> None:
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


def test_scheduled_preconditioner_iteration_zero() -> None:
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


def test_scheduled_preconditioner_combined_parameters() -> None:
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


def test_scheduled_preconditioner_no_fallback() -> None:
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


def test_scheduled_preconditioner_iteration_is_int() -> None:
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
