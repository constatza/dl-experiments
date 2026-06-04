"""Tests for multi-matrix orchestration allocation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.generation.orchestration import (
    _allocate_strategy_counts_across_bindings,
    _resolve_binding_strategy_counts,
)
from neuralls.domain.generation.source_streams import SystemBinding


class _FakeRng:
    """Deterministic stand-in for allocation tests."""

    def __init__(self, draws: list[int]) -> None:
        self._draws = np.asarray(draws, dtype=np.int64)

    def integers(self, low: int, high: int | None = None, size: int | None = None) -> np.ndarray:
        assert low == 0
        assert high is not None
        assert size == self._draws.size
        return self._draws


@pytest.fixture
def three_bindings() -> list[SystemBinding]:
    """Three independent matrix bindings."""
    return [
        SystemBinding(sample_id=0, matrix_sample_id=0),
        SystemBinding(sample_id=1, matrix_sample_id=1),
        SystemBinding(sample_id=2, matrix_sample_id=2),
    ]


def test_allocate_strategy_counts_without_replacement_is_even(
    three_bindings: list[SystemBinding],
) -> None:
    allocations = _allocate_strategy_counts_across_bindings(
        count=5,
        bindings=three_bindings,
        replacement=False,
        rng=np.random.default_rng(0),
    )

    assert allocations == [2, 2, 1]


def test_allocate_strategy_counts_with_replacement_uses_sampled_bindings(
    three_bindings: list[SystemBinding],
) -> None:
    allocations = _allocate_strategy_counts_across_bindings(
        count=5,
        bindings=three_bindings,
        replacement=True,
        rng=_FakeRng([2, 2, 0, 2, 1]),
    )

    assert allocations == [1, 1, 3]


def test_resolve_binding_strategy_counts_rejects_unsupported_replacement(
    three_bindings: list[SystemBinding],
) -> None:
    with pytest.raises(ValueError, match="does not support matrix replacement"):
        _resolve_binding_strategy_counts(
            bindings=three_bindings,
            counts={"neutral_ones": 5},
            mix=None,
            total=None,
            replacement=True,
            seed=0,
            strategy_overrides=None,
            has_rhs_source=False,
            num_matrix_samples=3,
        )


def test_resolve_binding_strategy_counts_rejects_finite_trace_replacement(
    three_bindings: list[SystemBinding],
) -> None:
    with pytest.raises(ValueError, match="finite external source"):
        _resolve_binding_strategy_counts(
            bindings=three_bindings,
            counts={"residual_traces": 5},
            mix=None,
            total=None,
            replacement=True,
            seed=0,
            strategy_overrides=None,
            has_rhs_source=True,
            num_matrix_samples=3,
        )
