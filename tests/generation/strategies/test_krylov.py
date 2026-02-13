"""Tests for KrylovStrategy."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.generation import run_generation


def test_krylov_registered() -> None:
    """KrylovStrategy is registered under 'krylov'."""
    from neuralls.generation.runner import _registry

    assert "krylov" in _registry._strategies


def test_krylov_shapes(spd_matrix: np.ndarray) -> None:
    """Output arrays have correct shapes."""
    n = spd_matrix.shape[0]
    cfg = {"samples": 4, "seed": 0}

    result = run_generation("krylov", spd_matrix, cfg=cfg)

    assert result.rhs is not None
    assert result.rhs.shape == (4, n)
    assert result.solutions is not None
    assert result.solutions.shape == (4, n)


def test_krylov_sample_count(spd_matrix: np.ndarray) -> None:
    """Different sample counts produce correct output sizes."""
    for count in (1, 5, 8):
        cfg = {"samples": count, "seed": 0}
        result = run_generation("krylov", spd_matrix, cfg=cfg)
        assert result.rhs is not None
        assert result.rhs.shape[0] == count


def test_krylov_computes_rhs(spd_matrix: np.ndarray) -> None:
    """RHS = A @ x for all generated samples."""
    cfg = {"samples": 5, "seed": 42}
    result = run_generation("krylov", spd_matrix, cfg=cfg)

    assert result.rhs is not None
    assert result.solutions is not None
    for i in range(result.solutions.shape[0]):
        expected = spd_matrix @ result.solutions[i]
        np.testing.assert_allclose(result.rhs[i], expected, rtol=1e-12)


def test_krylov_deterministic(spd_matrix: np.ndarray) -> None:
    """Same seed produces identical output across two calls."""
    cfg = {"samples": 4, "seed": 7}

    result1 = run_generation("krylov", spd_matrix, cfg=cfg)
    result2 = run_generation("krylov", spd_matrix, cfg=cfg)

    assert result1.rhs is not None and result2.rhs is not None
    np.testing.assert_array_equal(result1.rhs, result2.rhs)
    assert result1.solutions is not None and result2.solutions is not None
    np.testing.assert_array_equal(result1.solutions, result2.solutions)


def test_krylov_iters_default(spd_matrix: np.ndarray) -> None:
    """Strategy runs successfully with default krylov_iters."""
    cfg = {"samples": 3, "seed": 0}

    result = run_generation("krylov", spd_matrix, cfg=cfg)

    assert result.rhs is not None
    assert result.solutions is not None
    assert result.rhs.shape[0] == 3
