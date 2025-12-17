"""Tests for archive-based data generation strategies."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.generation import run_generation
from neuralls.generation.strategies.rhs_archive import RhsArchiveStrategy
from neuralls.generation.strategies.solution_archive import SolutionArchiveStrategy


@pytest.fixture
def spd_matrix() -> np.ndarray:
    """Create a small SPD matrix for testing."""
    n = 10
    A = np.random.randn(n, n)
    return A.T @ A + np.eye(n)


@pytest.fixture
def rhs_archive_files(tmp_path, spd_matrix):
    """Create temporary RHS archive files."""
    n = spd_matrix.shape[0]
    num_files = 5

    files = []
    for i in range(num_files):
        rhs = np.random.randn(n)
        file_path = tmp_path / f"rhs_{i:03d}.txt"
        np.savetxt(file_path, rhs)
        files.append(file_path)

    return files, tmp_path / "rhs_*.txt"


@pytest.fixture
def solution_archive_files(tmp_path, spd_matrix):
    """Create temporary solution archive files."""
    n = spd_matrix.shape[0]
    num_files = 5

    files = []
    for i in range(num_files):
        solution = np.random.randn(n)
        file_path = tmp_path / f"sol_{i:03d}.txt"
        np.savetxt(file_path, solution)
        files.append(file_path)

    return files, tmp_path / "sol_*.txt"


def test_rhs_archive_strategy_registered():
    """Test RhsArchiveStrategy is registered."""
    from neuralls.generation.runner import _registry

    assert "rhs_archive" in _registry._strategies
    assert isinstance(_registry._strategies["rhs_archive"], RhsArchiveStrategy)


def test_solution_archive_strategy_registered():
    """Test SolutionArchiveStrategy is registered."""
    from neuralls.generation.runner import _registry

    assert "solution_archive" in _registry._strategies
    assert isinstance(
        _registry._strategies["solution_archive"], SolutionArchiveStrategy
    )


def test_rhs_archive_load_files(spd_matrix, rhs_archive_files):
    """Test RhsArchiveStrategy loads RHS files correctly."""
    files, glob_pattern = rhs_archive_files

    cfg = {
        "rhs_glob": str(glob_pattern),
        "samples": 3,
        "solve_systems": False,
    }

    samples = run_generation("rhs_archive", spd_matrix, None, cfg=cfg)

    assert samples.rhs is not None
    assert samples.rhs.shape == (3, spd_matrix.shape[0])
    assert samples.solutions is not None
    assert samples.solutions.shape == (3, spd_matrix.shape[0])
    assert samples.residual_traces is None
    assert samples.error_traces is None


def test_rhs_archive_solve_systems(spd_matrix, rhs_archive_files):
    """Test RhsArchiveStrategy solves systems correctly."""
    files, glob_pattern = rhs_archive_files

    cfg = {
        "rhs_glob": str(glob_pattern),
        "samples": 2,
        "solve_systems": True,
        "cg_tolerance": 1e-10,
        "cg_max_iters": 100,
    }

    samples = run_generation("rhs_archive", spd_matrix, None, cfg=cfg)

    assert samples.rhs is not None
    assert samples.solutions is not None

    # Verify solutions satisfy Ax = b
    for i in range(samples.rhs.shape[0]):
        residual = spd_matrix @ samples.solutions[i] - samples.rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(samples.rhs[i])
        assert rel_residual < 1e-6, f"Solution {i} has large residual: {rel_residual}"


def test_rhs_archive_all_files(spd_matrix, rhs_archive_files):
    """Test RhsArchiveStrategy with samples=-1 loads all files."""
    files, glob_pattern = rhs_archive_files

    cfg = {
        "rhs_glob": str(glob_pattern),
        "samples": -1,  # All files
        "solve_systems": False,
    }

    samples = run_generation("rhs_archive", spd_matrix, None, cfg=cfg)

    assert samples.rhs is not None
    assert samples.rhs.shape[0] == len(files)


def test_rhs_archive_shuffle(spd_matrix, rhs_archive_files):
    """Test RhsArchiveStrategy shuffling is deterministic."""
    files, glob_pattern = rhs_archive_files

    cfg = {
        "rhs_glob": str(glob_pattern),
        "samples": 3,
        "shuffle": True,
        "seed": 42,
        "solve_systems": False,
    }

    samples1 = run_generation("rhs_archive", spd_matrix, None, cfg=cfg)
    samples2 = run_generation("rhs_archive", spd_matrix, None, cfg=cfg)

    assert samples1.rhs is not None
    assert samples2.rhs is not None
    np.testing.assert_array_equal(samples1.rhs, samples2.rhs)


def test_solution_archive_load_files(spd_matrix, solution_archive_files):
    """Test SolutionArchiveStrategy loads solution files correctly."""
    files, glob_pattern = solution_archive_files

    cfg = {
        "solutions_glob": str(glob_pattern),
        "samples": 3,
    }

    samples = run_generation("solution_archive", spd_matrix, None, cfg=cfg)

    assert samples.solutions is not None
    assert samples.solutions.shape == (3, spd_matrix.shape[0])
    assert samples.rhs is not None
    assert samples.rhs.shape == (3, spd_matrix.shape[0])
    assert samples.residual_traces is None
    assert samples.error_traces is None


def test_solution_archive_computes_rhs(spd_matrix, solution_archive_files):
    """Test SolutionArchiveStrategy computes RHS = A @ x correctly."""
    files, glob_pattern = solution_archive_files

    cfg = {
        "solutions_glob": str(glob_pattern),
        "samples": 2,
    }

    samples = run_generation("solution_archive", spd_matrix, None, cfg=cfg)

    assert samples.solutions is not None
    assert samples.rhs is not None

    # Verify RHS = A @ x
    for i in range(samples.solutions.shape[0]):
        expected_rhs = spd_matrix @ samples.solutions[i]
        np.testing.assert_allclose(
            samples.rhs[i],
            expected_rhs,
            rtol=1e-12,
            err_msg=f"RHS {i} doesn't match A @ x",
        )


def test_solution_archive_all_files(spd_matrix, solution_archive_files):
    """Test SolutionArchiveStrategy with samples=-1 loads all files."""
    files, glob_pattern = solution_archive_files

    cfg = {
        "solutions_glob": str(glob_pattern),
        "samples": -1,  # All files
    }

    samples = run_generation("solution_archive", spd_matrix, None, cfg=cfg)

    assert samples.solutions is not None
    assert samples.solutions.shape[0] == len(files)


def test_solution_archive_shuffle(spd_matrix, solution_archive_files):
    """Test SolutionArchiveStrategy shuffling is deterministic."""
    files, glob_pattern = solution_archive_files

    cfg = {
        "solutions_glob": str(glob_pattern),
        "samples": 3,
        "shuffle": True,
        "seed": 42,
    }

    samples1 = run_generation("solution_archive", spd_matrix, None, cfg=cfg)
    samples2 = run_generation("solution_archive", spd_matrix, None, cfg=cfg)

    assert samples1.solutions is not None
    assert samples2.solutions is not None
    np.testing.assert_array_equal(samples1.solutions, samples2.solutions)


def test_rhs_archive_missing_glob():
    """Test RhsArchiveStrategy raises error without rhs_glob."""
    matrix = np.eye(5)
    cfg = {"samples": 10}

    with pytest.raises(ValueError, match="rhs_glob"):
        run_generation("rhs_archive", matrix, None, cfg=cfg)


def test_solution_archive_missing_glob():
    """Test SolutionArchiveStrategy raises error without solutions_glob."""
    matrix = np.eye(5)
    cfg = {"samples": 10}

    with pytest.raises(ValueError, match="solutions_glob"):
        run_generation("solution_archive", matrix, None, cfg=cfg)


def test_rhs_archive_insufficient_files(spd_matrix, rhs_archive_files):
    """Test RhsArchiveStrategy raises error when insufficient files."""
    files, glob_pattern = rhs_archive_files

    cfg = {
        "rhs_glob": str(glob_pattern),
        "samples": 100,  # More than available
        "solve_systems": False,
    }

    with pytest.raises(ValueError, match="only.*available"):
        run_generation("rhs_archive", spd_matrix, None, cfg=cfg)


def test_solution_archive_insufficient_files(spd_matrix, solution_archive_files):
    """Test SolutionArchiveStrategy raises error when insufficient files."""
    files, glob_pattern = solution_archive_files

    cfg = {
        "solutions_glob": str(glob_pattern),
        "samples": 100,  # More than available
    }

    with pytest.raises(ValueError, match="only.*available"):
        run_generation("solution_archive", spd_matrix, None, cfg=cfg)


def test_mixed_archive_and_synthetic(spd_matrix, rhs_archive_files):
    """Test mixing archive and synthetic strategies using generate_mixture."""
    from neuralls.generation import generate_mixture

    files, glob_pattern = rhs_archive_files

    # Create mother RHS for synthetic strategies
    mother_rhs = np.random.randn(spd_matrix.shape[0])

    X, Y, res_traces, err_traces = generate_mixture(
        spd_matrix,
        mother_rhs,
        counts={
            "rhs_archive": 2,
            "random": 3,
        },
        seed=42,
        shuffle=False,
        strategy_overrides={
            "rhs_archive": {
                "rhs_glob": str(glob_pattern),
                "solve_systems": False,
            }
        },
    )

    assert X.shape == (5, spd_matrix.shape[0])  # 2 archive + 3 random
    assert Y.shape == (5, spd_matrix.shape[0])
