"""Test fixtures for graph-cg tests."""

from __future__ import annotations

import sys
from types import ModuleType
from pathlib import Path

import numpy as np
import pytest
from loguru import logger


def _patch_default_paths_for_loaded_modules(
    path_values: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch already-imported modules that cache default path constants."""
    str_values = {
        "DEFAULT_PROCESSED_DIR": str(path_values["DEFAULT_PROCESSED_DATA_DIR"]),
        "DEFAULT_RESULTS_DIR": str(path_values["DEFAULT_OUTPUT_DIR"]),
        "DEFAULT_FIGURES_DIR_STR": str(path_values["DEFAULT_FIGURES_DIR"]),
    }
    all_values: dict[str, object] = {**path_values, **str_values}
    all_values["DEFAULT_CHECKPOINTS_DIR"] = (
        path_values["DEFAULT_OUTPUT_DIR"] / "checkpoints"
    )
    for module in tuple(sys.modules.values()):
        if not isinstance(module, ModuleType):
            continue
        for name, value in all_values.items():
            if hasattr(module, name):
                monkeypatch.setattr(module, name, value, raising=False)


@pytest.fixture(autouse=True)
def isolate_default_paths_with_tmp_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force all default artifact paths into pytest temp storage."""
    runtime_root = tmp_path / "runtime"
    env_map = {
        "GRAPH_CG_OUTPUT_DIR": str(runtime_root / "output"),
        "GRAPH_CG_PROCESSED_DIR": str(runtime_root / "processed"),
        "GRAPH_CG_FIGURES_DIR": str(runtime_root / "figures"),
        "GRAPH_CG_MLRUNS_DIR": str(runtime_root / "mlruns"),
        "GRAPH_CG_MLARTIFACTS_DIR": str(runtime_root / "mlartifacts"),
        # Force DLKit test artifacts away from repository-level tests/artifacts/.
        "DLKIT_TEST_MODE": "1",
        "DLKIT_TEST_ARTIFACT_ROOT": str(runtime_root / "dlkit_artifacts"),
        # Keep DLKit internals local to the test temp root.
        "DLKIT_ROOT_DIR": str(runtime_root),
        "DLKIT_INTERNAL_DIR": str(runtime_root / ".dlkit"),
    }
    for key, value in env_map.items():
        monkeypatch.setenv(key, value)

    from neuralls import constants as neuralls_constants

    path_values = {
        "DEFAULT_OUTPUT_DIR": Path(env_map["GRAPH_CG_OUTPUT_DIR"]).resolve(),
        "DEFAULT_PROCESSED_DATA_DIR": Path(env_map["GRAPH_CG_PROCESSED_DIR"]).resolve(),
        "DEFAULT_FIGURES_DIR": Path(env_map["GRAPH_CG_FIGURES_DIR"]).resolve(),
        "DEFAULT_MLRUNS_DIR": Path(env_map["GRAPH_CG_MLRUNS_DIR"]).resolve(),
        "DEFAULT_MLARTIFACTS_DIR": Path(env_map["GRAPH_CG_MLARTIFACTS_DIR"]).resolve(),
    }
    for name, value in path_values.items():
        monkeypatch.setattr(neuralls_constants, name, value, raising=False)

    _patch_default_paths_for_loaded_modules(path_values, monkeypatch)


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure loguru to be safe against closed streams during cleanup."""

    def safe_sink(message):
        try:
            sys.stderr.write(message)
        except ValueError:
            pass  # Ignore "I/O operation on closed file"

    logger.remove()
    logger.add(safe_sink)


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
def sample_dataset_npz(tmp_path, small_spd_matrix, archive_solutions, archive_rhs):
    """Create a sample dataset using split-array sparse storage.

    Args:
        tmp_path: Pytest temporary directory fixture
        small_spd_matrix: Test matrix
        archive_solutions: Test solutions
        archive_rhs: Test RHS vectors

    Returns:
        Path to dataset directory containing manifest + arrays
    """
    dataset_dir = tmp_path / "test-dataset"
    dataset_dir.mkdir()
    from neuralls.io.dataset_storage import save_dataset

    save_dataset(
        dataset_dir=dataset_dir,
        rhs=archive_rhs,
        solutions=archive_solutions,
        matrix=small_spd_matrix,
        normalization_type="matrix",
        matrix_norm=float(np.linalg.norm(small_spd_matrix, ord=2)),
        matrix_norm_type="spectral",
        scale_metadata={},
    )

    return dataset_dir


@pytest.fixture
def sample_dataset_with_raw(tmp_path, small_spd_matrix, archive_solutions, archive_rhs):
    """Create a sample dataset with normalized artifacts and raw arrays.

    Args:
        tmp_path: Pytest temporary directory fixture
        small_spd_matrix: Test matrix
        archive_solutions: Test solutions
        archive_rhs: Test RHS vectors

    Returns:
        Path to dataset directory containing normalized artifacts and raw/ arrays
    """
    dataset_dir = tmp_path / "test-dataset-with-raw"
    dataset_dir.mkdir()
    from neuralls.io.dataset_storage import save_dataset

    save_dataset(
        dataset_dir=dataset_dir,
        rhs=archive_rhs,
        solutions=archive_solutions,
        matrix=small_spd_matrix,
        normalization_type="matrix",
        matrix_norm=float(np.linalg.norm(small_spd_matrix, ord=2)),
        matrix_norm_type="spectral",
        scale_metadata={},
    )

    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir()
    np.save(raw_dir / "matrix.npy", small_spd_matrix)
    np.save(raw_dir / "rhs.npy", archive_rhs)
    np.save(raw_dir / "solutions.npy", archive_solutions)

    return dataset_dir
