"""Tests for SearchDirectionsStrategy (search_directions).

This strategy collects (A @ p_k, p_k) pairs from CG iterations.
Currently raises RuntimeError because scipy.sparse.linalg.cg does not expose
internal search direction vectors. Tests verify registration and the expected
error behaviour until a custom CG implementation is available.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.generation import run_generation


@pytest.fixture
def single_rhs(spd_matrix: np.ndarray) -> np.ndarray:
    """Deterministic RHS vector for the SPD system."""
    rng = np.random.default_rng(55)
    return rng.standard_normal(spd_matrix.shape[0])


def test_search_directions_registered() -> None:
    """SearchDirectionsStrategy is registered under 'search_directions'."""
    from neuralls.generation.runner import _registry

    assert "search_directions" in _registry._strategies


def test_search_directions_config_accepts_every_n(spd_matrix: np.ndarray) -> None:
    """Config accepts every_n field without validation error (raises RuntimeError, not ValueError)."""
    cfg = {"samples": 2, "cg_iters": 4, "seed": 0, "every_n": 2}

    # Should fail with RuntimeError (no search directions from scipy), NOT ValidationError
    with pytest.raises(RuntimeError, match="search directions"):
        run_generation("search_directions", spd_matrix, cfg=cfg, single_rhs=np.ones(spd_matrix.shape[0]))


def test_search_directions_raises_not_implemented(
    spd_matrix: np.ndarray, single_rhs: np.ndarray
) -> None:
    """Attempting to collect search directions raises RuntimeError with helpful message."""
    cfg = {"samples": 2, "cg_iters": 4, "seed": 0}

    with pytest.raises(RuntimeError, match="search directions"):
        run_generation("search_directions", spd_matrix, cfg=cfg, single_rhs=single_rhs)


def test_search_directions_error_message_helpful(
    spd_matrix: np.ndarray, single_rhs: np.ndarray
) -> None:
    """RuntimeError message mentions scipy and suggests alternatives."""
    cfg = {"samples": 1, "cg_iters": 2, "seed": 0}

    with pytest.raises(RuntimeError) as exc_info:
        run_generation("search_directions", spd_matrix, cfg=cfg, single_rhs=single_rhs)

    msg = str(exc_info.value).lower()
    assert "scipy" in msg or "search direction" in msg
