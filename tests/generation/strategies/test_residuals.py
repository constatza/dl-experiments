"""Thorough tests for residual-error trace strategies.

Mathematical properties under test:
    - e_k = x_true - x_k  (error = true solution minus current iterate)
    - r_k = b - A @ x_k   (CG residual definition)
    - r_k = A @ e_k        (holds exactly when b = A @ x_true)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.generation import run_generation
from neuralls.generation.helpers import trace_rows_per_system
from neuralls.generation.interfaces import TracingSolverCallable


@pytest.fixture
def single_rhs(spd_matrix: np.ndarray) -> np.ndarray:
    """Deterministic RHS vector b for the SPD system."""
    rng = np.random.default_rng(17)
    return rng.standard_normal(spd_matrix.shape[0])


@pytest.fixture
def solution_files(tmp_path: Path, spd_matrix: np.ndarray) -> tuple[list[Path], str]:
    """Save random solution vectors to disk and return (files, glob_pattern).

    Each stored x is a random vector; RHS b = A @ x is computed by the strategy.
    """
    rng = np.random.default_rng(31)
    n = spd_matrix.shape[0]
    files = []
    for i in range(4):
        x = rng.standard_normal(n)
        p = tmp_path / f"sol_{i:03d}.txt"
        np.savetxt(p, x)
        files.append(p)
    return files, str(tmp_path / "sol_*.txt")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_residuals_strategy_registered() -> None:
    """The primary residual-error trace strategy is registered as 'residuals'."""
    from neuralls.generation.runner import _registry

    assert "residuals" in _registry._strategies


def test_gaussian_residuals_strategy_registered() -> None:
    """Gaussian residual traces are available as an explicit strategy."""
    from neuralls.generation.runner import _registry

    assert "gaussian_residuals" in _registry._strategies


# ---------------------------------------------------------------------------
# Shapes – single-RHS mode
# ---------------------------------------------------------------------------


def test_residuals_single_rhs_shapes(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """Single-RHS mode: all output arrays have the correct shapes."""
    n = spd_matrix.shape[0]
    requested_rows = 10
    cg_iters = 4
    cfg = {"samples": requested_rows, "cg_iters": cg_iters, "seed": 0}

    result = run_generation(
        "residuals", spd_matrix, cfg=cfg, solver=residual_solver, single_rhs=single_rhs
    )

    rows_per_system = trace_rows_per_system(cg_iters)
    expected_systems = max(1, requested_rows // rows_per_system)
    total = expected_systems * rows_per_system

    assert result.rhs is not None
    assert result.rhs.shape == (expected_systems, n)
    assert result.solutions is not None
    assert result.solutions.shape == (expected_systems, n)

    et = result.error_traces
    assert et is not None

    assert et.residuals.shape == (total, n)
    assert et.solutions_current.shape == (total, n)
    assert et.errors.shape == (total, n)
    assert et.true_solutions.shape == (expected_systems, n)
    assert et.sample_indices.shape == (total,)
    assert et.iteration_indices.shape == (total,)


# ---------------------------------------------------------------------------
# Shapes – multiple-RHS mode
# ---------------------------------------------------------------------------


def test_residuals_multi_rhs_shapes(
    spd_matrix: np.ndarray,
    solution_files: tuple[list[Path], str],
) -> None:
    """Multiple-RHS mode: all output arrays have correct shapes."""
    files, glob_pattern = solution_files
    n = spd_matrix.shape[0]
    requested_rows = 8
    cg_iters = 3
    cfg = {
        "samples": requested_rows,
        "cg_iters": cg_iters,
        "seed": 0,
        "solutions_glob": glob_pattern,
    }

    result = run_generation("residuals", spd_matrix, cfg=cfg)

    rows_per_system = trace_rows_per_system(cg_iters)
    expected_systems = max(1, requested_rows // rows_per_system)
    total = expected_systems * rows_per_system

    assert result.rhs is not None
    assert result.rhs.shape == (expected_systems, n)

    et = result.error_traces
    assert et is not None

    assert et.residuals.shape == (total, n)
    assert et.solutions_current.shape == (total, n)
    assert et.errors.shape == (total, n)
    assert et.true_solutions.shape == (expected_systems, n)


def test_gaussian_residuals_multi_rhs_shapes(spd_matrix: np.ndarray) -> None:
    """Gaussian residuals do not require archive solutions or solution globs."""
    n = spd_matrix.shape[0]
    requested_rows = 8
    cg_iters = 3
    cfg = {"samples": requested_rows, "cg_iters": cg_iters, "seed": 0}

    result = run_generation("gaussian_residuals", spd_matrix, cfg=cfg)

    rows_per_system = trace_rows_per_system(cg_iters)
    expected_systems = max(1, requested_rows // rows_per_system)
    total = expected_systems * rows_per_system

    assert result.rhs is not None
    assert result.rhs.shape == (expected_systems, n)

    et = result.error_traces
    assert et is not None
    assert et.residuals.shape == (total, n)
    assert et.solutions_current.shape == (total, n)
    assert et.errors.shape == (total, n)
    assert et.true_solutions.shape == (expected_systems, n)


def test_gaussian_residuals_math_holds(spd_matrix: np.ndarray) -> None:
    """Gaussian residuals still satisfy e_k = x_true - x_k and r_k = A @ e_k."""
    cfg = {"samples": 10, "cg_iters": 4, "seed": 0}

    result = run_generation("gaussian_residuals", spd_matrix, cfg=cfg)

    et = result.error_traces
    assert et is not None

    for i in range(et.errors.shape[0]):
        s_idx = et.sample_indices[i]
        x_true = et.true_solutions[s_idx]
        x_k = et.solutions_current[i]
        np.testing.assert_allclose(et.errors[i], x_true - x_k, rtol=1e-12)
        np.testing.assert_allclose(et.residuals[i], spd_matrix @ et.errors[i], rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Mathematical correctness
# ---------------------------------------------------------------------------


def test_error_equals_true_minus_current(
    spd_matrix: np.ndarray,
    solution_files: tuple[list[Path], str],
) -> None:
    """e_k = x_true - x_k holds for every recorded trace pair."""
    files, glob_pattern = solution_files
    cfg = {
        "samples": 10,
        "cg_iters": 4,
        "seed": 0,
        "solutions_glob": glob_pattern,
    }

    result = run_generation("residuals", spd_matrix, cfg=cfg)

    et = result.error_traces
    assert et is not None
    for i in range(et.errors.shape[0]):
        s_idx = et.sample_indices[i]
        x_true = et.true_solutions[s_idx]
        x_k = et.solutions_current[i]
        np.testing.assert_allclose(et.errors[i], x_true - x_k, rtol=1e-12)


def test_residual_equals_b_minus_ax(
    spd_matrix: np.ndarray,
    solution_files: tuple[list[Path], str],
) -> None:
    """r_k = b - A @ x_k: CG residual matches b - A @ current iterate."""
    files, glob_pattern = solution_files
    cfg = {
        "samples": 10,
        "cg_iters": 4,
        "seed": 0,
        "solutions_glob": glob_pattern,
    }

    result = run_generation("residuals", spd_matrix, cfg=cfg)

    et = result.error_traces
    assert et is not None
    assert result.rhs is not None

    for i in range(et.residuals.shape[0]):
        s_idx = et.sample_indices[i]
        b = result.rhs[s_idx]
        x_k = et.solutions_current[i]
        expected_residual = b - spd_matrix @ x_k
        np.testing.assert_allclose(et.residuals[i], expected_residual, rtol=1e-10, atol=1e-10)


def test_residual_equals_a_times_error(
    spd_matrix: np.ndarray,
    solution_files: tuple[list[Path], str],
) -> None:
    """r_k = A @ e_k: holds exactly because b = A @ x_true in archive mode."""
    files, glob_pattern = solution_files
    cfg = {
        "samples": 10,
        "cg_iters": 4,
        "seed": 0,
        "solutions_glob": glob_pattern,
    }

    result = run_generation("residuals", spd_matrix, cfg=cfg)

    et = result.error_traces
    assert et is not None

    for i in range(et.residuals.shape[0]):
        a_times_e = spd_matrix @ et.errors[i]
        np.testing.assert_allclose(et.residuals[i], a_times_e, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Trace count and indices
# ---------------------------------------------------------------------------


def test_residuals_trace_count(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """Total trace entries = num_systems × (cg_iters + 1) when every_n=1."""
    requested_rows = 10
    cg_iters = 4
    cfg = {"samples": requested_rows, "cg_iters": cg_iters, "seed": 0}

    result = run_generation(
        "residuals", spd_matrix, cfg=cfg, solver=residual_solver, single_rhs=single_rhs
    )

    et = result.error_traces
    assert et is not None
    rows_per_system = trace_rows_per_system(cg_iters)
    expected_systems = max(1, requested_rows // rows_per_system)
    assert et.residuals.shape[0] == expected_systems * rows_per_system


def test_residuals_sample_indices(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """sample_indices correctly identifies which base system each trace pair belongs to."""
    requested_rows = 15
    cg_iters = 4
    cfg = {"samples": requested_rows, "cg_iters": cg_iters, "seed": 0}

    result = run_generation(
        "residuals", spd_matrix, cfg=cfg, solver=residual_solver, single_rhs=single_rhs
    )

    et = result.error_traces
    assert et is not None

    entries_per_system = trace_rows_per_system(cg_iters)
    expected_systems = max(1, requested_rows // entries_per_system)
    for s in range(expected_systems):
        start = s * entries_per_system
        end = start + entries_per_system
        np.testing.assert_array_equal(et.sample_indices[start:end], s)


def test_residuals_iteration_indices(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """iteration_indices run 0..cg_iters for each system when every_n=1."""
    requested_rows = 10
    cg_iters = 4
    cfg = {"samples": requested_rows, "cg_iters": cg_iters, "seed": 0}

    result = run_generation(
        "residuals", spd_matrix, cfg=cfg, solver=residual_solver, single_rhs=single_rhs
    )

    et = result.error_traces
    assert et is not None

    entries_per_system = trace_rows_per_system(cg_iters)
    expected_iters = np.arange(entries_per_system, dtype=np.int64)
    expected_systems = max(1, requested_rows // entries_per_system)
    for s in range(expected_systems):
        start = s * entries_per_system
        end = start + entries_per_system
        np.testing.assert_array_equal(et.iteration_indices[start:end], expected_iters)


# ---------------------------------------------------------------------------
# every_n downsampling
# ---------------------------------------------------------------------------


def test_residuals_every_n_reduces_count(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """every_n=2 yields half the trace entries (cg_iters=5 → 6 residuals, divisible by 2)."""
    cg_iters = 5  # 6 residuals per sample (0..5)
    cfg_full = {"samples": 12, "cg_iters": cg_iters, "seed": 0, "every_n": 1}
    cfg_half = {"samples": 6, "cg_iters": cg_iters, "seed": 0, "every_n": 2}

    r_full = run_generation(
        "residuals", spd_matrix, cfg=cfg_full, solver=residual_solver, single_rhs=single_rhs
    )
    r_half = run_generation(
        "residuals", spd_matrix, cfg=cfg_half, solver=residual_solver, single_rhs=single_rhs
    )

    assert r_full.error_traces is not None
    assert r_half.error_traces is not None
    assert r_half.error_traces.residuals.shape[0] == r_full.error_traces.residuals.shape[0] // 2


def test_residuals_every_n_indices_correct(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """every_n=2 with cg_iters=5: iteration_indices are [0, 2, 4] not [0, 1, 2]."""
    cfg = {"samples": 1, "cg_iters": 5, "seed": 0, "every_n": 2}

    result = run_generation(
        "residuals", spd_matrix, cfg=cfg, solver=residual_solver, single_rhs=single_rhs
    )

    et = result.error_traces
    assert et is not None
    np.testing.assert_array_equal(et.iteration_indices, np.array([0, 2, 4], dtype=np.int64))


def test_residuals_every_n_math_still_holds(
    spd_matrix: np.ndarray,
    solution_files: tuple[list[Path], str],
) -> None:
    """e_k = x_true - x_k and r_k = A @ e_k hold even after every_n downsampling."""
    files, glob_pattern = solution_files
    cfg = {
        "samples": 6,
        "cg_iters": 5,
        "seed": 0,
        "solutions_glob": glob_pattern,
        "every_n": 2,
    }

    result = run_generation("residuals", spd_matrix, cfg=cfg)

    et = result.error_traces
    assert et is not None

    for i in range(et.errors.shape[0]):
        s_idx = et.sample_indices[i]
        x_true = et.true_solutions[s_idx]
        x_k = et.solutions_current[i]
        np.testing.assert_allclose(et.errors[i], x_true - x_k, rtol=1e-12)
        np.testing.assert_allclose(et.residuals[i], spd_matrix @ et.errors[i], rtol=1e-9, atol=1e-9)


def test_residuals_samples_minus_one_requires_finite_source(
    spd_matrix: np.ndarray, single_rhs: np.ndarray, residual_solver: TracingSolverCallable
) -> None:
    """samples=-1 is rejected for single-RHS mode because the source is unbounded."""
    with pytest.raises(ValueError, match="samples=-1"):
        run_generation(
            "residuals",
            spd_matrix,
            cfg={"samples": -1, "cg_iters": 4, "seed": 0},
            solver=residual_solver,
            single_rhs=single_rhs,
        )
