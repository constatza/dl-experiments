"""Tests for ResidualTraceStrategy and every_n downsampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.generation import run_generation
from neuralls.generation.helpers import trace_rows_per_system


@pytest.fixture
def single_rhs(spd_matrix: np.ndarray) -> np.ndarray:
    """A single deterministic RHS vector for the SPD system."""
    rng = np.random.default_rng(99)
    return rng.standard_normal(spd_matrix.shape[0])


@pytest.fixture
def solution_files(tmp_path: Path, spd_matrix: np.ndarray) -> tuple[list[Path], str]:
    """Save random solution vectors to disk and return (files, glob_pattern)."""
    rng = np.random.default_rng(7)
    n = spd_matrix.shape[0]
    num_files = 4
    files = []
    for i in range(num_files):
        x = rng.standard_normal(n)
        p = tmp_path / f"sol_{i:03d}.txt"
        np.savetxt(p, x)
        files.append(p)
    return files, str(tmp_path / "sol_*.txt")


def test_residual_trace_registered() -> None:
    """ResidualTraceStrategy is registered under 'cg_residual'."""
    from neuralls.generation.runner import _registry

    assert "cg_residual" in _registry._strategies


def test_residual_alias_registered() -> None:
    """ResidualAliasStrategy is registered under 'residual'."""
    from neuralls.generation.runner import _registry

    assert "residual" in _registry._strategies


def test_residual_trace_shapes(spd_matrix: np.ndarray, single_rhs: np.ndarray) -> None:
    """RHS output has correct shape and residual_traces is populated."""
    n = spd_matrix.shape[0]
    cfg = {"samples": 2, "cg_iters": 4, "seed": 0}

    result = run_generation("cg_residual", spd_matrix, cfg=cfg, single_rhs=single_rhs)

    assert result.rhs is not None
    expected_systems = max(1, 2 // trace_rows_per_system(4))
    assert result.rhs.shape == (expected_systems, n)
    assert result.residual_traces is not None


def test_residual_trace_trace_count(spd_matrix: np.ndarray, single_rhs: np.ndarray) -> None:
    """Trace strategies size systems from the requested row budget using integer division."""
    requested_rows = 12
    cg_iters = 5
    cfg = {"samples": requested_rows, "cg_iters": cg_iters, "seed": 0}

    result = run_generation("cg_residual", spd_matrix, cfg=cfg, single_rhs=single_rhs)

    assert result.residual_traces is not None
    total = result.residual_traces.residuals.shape[0]
    rows_per_system = trace_rows_per_system(cg_iters)
    expected_systems = max(1, requested_rows // rows_per_system)
    assert total == expected_systems * rows_per_system


def test_residual_trace_every_n_basic(spd_matrix: np.ndarray, single_rhs: np.ndarray) -> None:
    """every_n=2 yields half the trace entries compared to every_n=1.

    Uses cg_iters=5 so each sample has 6 residuals (divisible by 2).
    """
    cg_iters = 5  # 6 total residuals per sample (0..5), divisible by 2
    cfg_full = {"samples": 12, "cg_iters": cg_iters, "seed": 0, "every_n": 1}
    cfg_half = {"samples": 6, "cg_iters": cg_iters, "seed": 0, "every_n": 2}

    result_full = run_generation("cg_residual", spd_matrix, cfg=cfg_full, single_rhs=single_rhs)
    result_half = run_generation("cg_residual", spd_matrix, cfg=cfg_half, single_rhs=single_rhs)

    assert result_full.residual_traces is not None
    assert result_half.residual_traces is not None
    full_count = result_full.residual_traces.residuals.shape[0]
    half_count = result_half.residual_traces.residuals.shape[0]
    assert half_count == full_count // 2


def test_residual_trace_every_n_indices(spd_matrix: np.ndarray, single_rhs: np.ndarray) -> None:
    """every_n=2 produces iteration_indices 0, 2, 4, ... not 0, 1, 2, ...

    Uses cg_iters=5 (6 total residuals), so indices are 0, 2, 4.
    """
    cg_iters = 5  # 6 residuals per sample: indices 0..5
    cfg = {"samples": 1, "cg_iters": cg_iters, "seed": 0, "every_n": 2}

    result = run_generation("cg_residual", spd_matrix, cfg=cfg, single_rhs=single_rhs)

    assert result.residual_traces is not None
    iidx = result.residual_traces.iteration_indices
    expected = np.array([0, 2, 4], dtype=np.int64)
    np.testing.assert_array_equal(iidx, expected)


# ---------------------------------------------------------------------------
# Multiple-RHS mode (solutions loaded from archive files)
# ---------------------------------------------------------------------------


def test_residual_trace_multi_rhs_shapes(
    spd_matrix: np.ndarray,
    solution_files: tuple[list[Path], str],
) -> None:
    """Multiple-RHS mode: shapes are correct when solutions are loaded from files."""
    n = spd_matrix.shape[0]
    files, glob_pattern = solution_files
    requested_rows = 8
    cfg = {
        "samples": requested_rows,
        "cg_iters": 4,
        "seed": 0,
        "solutions_glob": glob_pattern,
    }

    result = run_generation("cg_residual", spd_matrix, cfg=cfg)

    assert result.rhs is not None
    expected_systems = max(1, requested_rows // trace_rows_per_system(4))
    assert result.rhs.shape == (expected_systems, n)
    assert result.residual_traces is not None


def test_residual_trace_multi_rhs_trace_count(
    spd_matrix: np.ndarray,
    solution_files: tuple[list[Path], str],
) -> None:
    """Multiple-RHS mode: total trace entries = num_systems × (cg_iters + 1)."""
    files, glob_pattern = solution_files
    requested_rows = 8
    cg_iters = 3
    cfg = {
        "samples": requested_rows,
        "cg_iters": cg_iters,
        "seed": 0,
        "solutions_glob": glob_pattern,
    }

    result = run_generation("cg_residual", spd_matrix, cfg=cfg)

    assert result.residual_traces is not None
    total = result.residual_traces.residuals.shape[0]
    rows_per_system = trace_rows_per_system(cg_iters)
    expected_systems = max(1, requested_rows // rows_per_system)
    assert total == expected_systems * rows_per_system


def test_residual_trace_multi_rhs_every_n(
    spd_matrix: np.ndarray,
    solution_files: tuple[list[Path], str],
) -> None:
    """Multiple-RHS mode: every_n downsampling works for archive-loaded solutions."""
    files, glob_pattern = solution_files
    cg_iters = 5  # 6 residuals per sample (0..5), divisible by 2
    cfg_full = {
        "samples": 12,
        "cg_iters": cg_iters,
        "seed": 0,
        "solutions_glob": glob_pattern,
        "every_n": 1,
    }
    cfg_half = {
        "samples": 6,
        "cg_iters": cg_iters,
        "seed": 0,
        "solutions_glob": glob_pattern,
        "every_n": 2,
    }

    result_full = run_generation("cg_residual", spd_matrix, cfg=cfg_full)
    result_half = run_generation("cg_residual", spd_matrix, cfg=cfg_half)

    assert result_full.residual_traces is not None
    assert result_half.residual_traces is not None
    assert result_half.residual_traces.residuals.shape[0] == result_full.residual_traces.residuals.shape[0] // 2


def test_residual_trace_samples_minus_one_requires_finite_source(
    spd_matrix: np.ndarray, single_rhs: np.ndarray
) -> None:
    """samples=-1 is rejected for single-RHS mode because the source is unbounded."""
    with pytest.raises(ValueError, match="samples=-1"):
        run_generation(
            "cg_residual",
            spd_matrix,
            cfg={"samples": -1, "cg_iters": 4, "seed": 0},
            single_rhs=single_rhs,
        )
