"""Tests for SearchDirectionsStrategy (search_directions)."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.generation import run_generation
from neuralls.generation.runner import SingleRhsStrategyRegistration, _registry


@pytest.fixture
def single_rhs(spd_matrix: np.ndarray) -> np.ndarray:
    """Deterministic RHS vector for the SPD system."""
    rng = np.random.default_rng(55)
    return rng.standard_normal(spd_matrix.shape[0])


def test_search_directions_registered_as_single_rhs_strategy() -> None:
    """SearchDirectionsStrategy is registered with explicit single-RHS capability."""
    registration = _registry._strategies["search_directions"]
    assert isinstance(registration, SingleRhsStrategyRegistration)


def test_search_directions_config_accepts_every_n(spd_matrix: np.ndarray) -> None:
    """Config accepts `every_n` and yields traced direction/product pairs."""
    cfg = {"samples": 2, "cg_iters": 4, "seed": 0, "every_n": 2}

    result = run_generation(
        "search_directions",
        spd_matrix,
        cfg=cfg,
        single_rhs=np.ones(spd_matrix.shape[0]),
    )

    assert result.residual_traces is not None
    assert result.residual_traces.search_directions is not None
    assert result.residual_traces.search_direction_products is not None
    assert result.residual_traces.search_directions.shape == result.residual_traces.search_direction_products.shape
    assert result.residual_traces.search_directions.shape[0] > 0


def test_search_directions_collects_repeated_single_rhs(
    spd_matrix: np.ndarray,
    single_rhs: np.ndarray,
) -> None:
    """Single-RHS mode keeps the same RHS while collecting CG search directions."""
    cfg = {"samples": 3, "cg_iters": 4, "seed": 0}

    result = run_generation("search_directions", spd_matrix, cfg=cfg, single_rhs=single_rhs)

    assert result.rhs is not None
    assert result.rhs.shape[1] == single_rhs.shape[0]
    for row in result.rhs:
        np.testing.assert_allclose(row, single_rhs)

    assert result.residual_traces is not None
    assert result.residual_traces.search_directions is not None
    assert result.residual_traces.search_direction_products is not None


def test_search_direction_products_match_matrix_action(
    spd_matrix: np.ndarray,
    single_rhs: np.ndarray,
) -> None:
    """Stored products equal `A @ p_k` for the traced directions."""
    cfg = {"samples": 1, "cg_iters": 3, "seed": 0}

    result = run_generation("search_directions", spd_matrix, cfg=cfg, single_rhs=single_rhs)
    traces = result.residual_traces

    assert traces is not None
    assert traces.search_directions is not None
    assert traces.search_direction_products is not None

    expected = np.array([spd_matrix @ direction for direction in traces.search_directions])
    np.testing.assert_allclose(traces.search_direction_products, expected)
