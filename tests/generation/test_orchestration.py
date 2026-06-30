"""Tests for multi-matrix orchestration allocation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.generation.orchestration import (
    _allocate_strategy_counts_across_bindings,
    _generate_mixture_with_metadata,
    _resolve_binding_strategy_counts,
)
from neuralls.domain.generation.source_streams import SystemBinding


@pytest.fixture
def spd_matrix() -> np.ndarray:
    """Small SPD matrix for orchestration tests."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 8))
    return A.T @ A + np.eye(8)


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


def test_generate_mixture_rhs_kind_codes_length_matches_trace_rows_after_shuffle(
    spd_matrix: np.ndarray,
) -> None:
    """rhs_kind_codes must align with error_traces.residuals, not with base-system count.

    gaussian_residuals produces N base systems but N*rows_per_system trace pairs.
    With shuffle=True the shuffle used to index rhs_kind_codes (trace-level, final_rows entries)
    with base-system-level indices (referenced_samples entries), silently truncating it.
    _finalize_payload then caught the mismatch: rhs_kind_codes.shape[0] != rhs_all.shape[0].
    """
    # cg_iters=3 → rows_per_system=4; samples=8 → 2 base systems, 8 trace pairs.
    # Bug: shuffle indexed rhs_kind_codes (len=8) with 2 base-system indices → truncated to len=2.
    result = _generate_mixture_with_metadata(
        spd_matrix,
        counts={"gaussian_residuals": 8},
        seed=0,
        shuffle=True,
        strategy_overrides={"gaussian_residuals": {"cg_iters": 3}},
    )

    assert result.error_traces is not None
    assert result.rhs_kind_codes.shape[0] == result.error_traces.residuals.shape[0]
