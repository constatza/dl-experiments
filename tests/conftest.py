"""Test fixtures for graph-cg tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def small_spd_matrix() -> np.ndarray:
    """Small 2x2 symmetric positive definite test matrix.

    Returns:
        2x2 SPD matrix with known properties
    """
    return np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)


@pytest.fixture
def small_rhs() -> np.ndarray:
    """Small 2D RHS vector for testing.

    Returns:
        2D RHS vector
    """
    return np.array([1.0, 0.0], dtype=np.float64)


@pytest.fixture
def archive_solutions() -> np.ndarray:
    """Pre-computed archive solutions for testing.

    Returns:
        Array of 3 solution vectors (2D each)
    """
    return np.array(
        [[0.5, 0.3], [0.2, 0.8], [0.1, 0.4]],
        dtype=np.float64,
    )


@pytest.fixture
def archive_rhs(
    small_spd_matrix: np.ndarray, archive_solutions: np.ndarray
) -> np.ndarray:
    """Pre-computed RHS vectors from archive solutions.

    Args:
        small_spd_matrix: Test matrix A
        archive_solutions: Archive solution vectors

    Returns:
        Array of RHS vectors b = A @ x
    """
    return np.array(
        [small_spd_matrix @ x for x in archive_solutions],
        dtype=np.float64,
    )


@pytest.fixture
def test_seed() -> int:
    """Deterministic seed for reproducible tests.

    Returns:
        Random seed value
    """
    return 42


@pytest.fixture
def linear_config_path() -> str:
    """Path to linear model config.

    Returns:
        Path to linear model configuration file
    """
    import os

    dlx_root = os.environ.get("DLX_ROOT", "/")
    return f"{dlx_root}/graph-cg/configs/linear.toml"


@pytest.fixture
def data_config_path() -> str:
    """Path to data collection config.

    Returns:
        Path to 504-solutions data collection configuration file
    """
    import os

    dlx_root = os.environ.get("DLX_ROOT", "/")
    return f"{dlx_root}/graph-cg/data-configs/collect-504-solutions.toml"


@pytest.fixture
def sample_dataset_npz(tmp_path, small_spd_matrix, archive_solutions, archive_rhs):
    """Create a sample dataset in new .npz format.

    Args:
        tmp_path: Pytest temporary directory fixture
        small_spd_matrix: Test matrix
        archive_solutions: Test solutions
        archive_rhs: Test RHS vectors

    Returns:
        Path to dataset directory containing normalized.npz
    """
    dataset_dir = tmp_path / "test-dataset"
    dataset_dir.mkdir()

    # Save in new .npz format
    np.savez(
        dataset_dir / "normalized.npz",
        matrix=small_spd_matrix,
        rhs=archive_rhs,
        solutions=archive_solutions,
    )

    return dataset_dir


@pytest.fixture
def sample_dataset_with_raw(tmp_path, small_spd_matrix, archive_solutions, archive_rhs):
    """Create a sample dataset with both normalized and raw data.

    Args:
        tmp_path: Pytest temporary directory fixture
        small_spd_matrix: Test matrix
        archive_solutions: Test solutions
        archive_rhs: Test RHS vectors

    Returns:
        Path to dataset directory containing normalized.npz and raw.npz
    """
    dataset_dir = tmp_path / "test-dataset-with-raw"
    dataset_dir.mkdir()

    # Save normalized data
    np.savez(
        dataset_dir / "normalized.npz",
        matrix=small_spd_matrix,
        rhs=archive_rhs,
        solutions=archive_solutions,
    )

    # Save raw data (same for testing)
    np.savez(
        dataset_dir / "raw.npz",
        matrix=small_spd_matrix,
        rhs=archive_rhs,
        solutions=archive_solutions,
    )

    return dataset_dir
