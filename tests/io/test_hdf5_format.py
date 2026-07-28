"""Tests for DenseHdf5Accumulator and Hdf5GenerationStorage."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.typing import NDArray

from neuralls.domain.generation.payloads import GeneratedDatasetPayload
from neuralls.platform.config.dataset_entries import entry_from_path
from neuralls.platform.storage.datasets import (
    DenseHdf5Accumulator,
    Hdf5GenerationStorage,
    load_dataset_manifest,
    resolve_dataset_artifacts,
)
from neuralls.platform.storage.training_artifacts import Hdf5ArraySource, load_training_arrays

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_matrix() -> NDArray[np.float64]:
    """3×3 dense matrix for accumulator tests.

    Returns:
        Array of shape (3, 3).
    """
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)


@pytest.fixture
def two_by_two() -> NDArray[np.float64]:
    """2×2 diagonal matrix.

    Returns:
        Array of shape (2, 2).
    """
    return np.array([[3.0, 0.0], [0.0, 7.0]], dtype=np.float64)


@pytest.fixture
def simple_payload(tmp_path: Path, two_by_two: NDArray[np.float64]) -> GeneratedDatasetPayload:
    """Minimal payload backed by an HDF5 accumulator.

    Args:
        tmp_path: Pytest-provided temporary directory.
        two_by_two: 2×2 fixture matrix.

    Returns:
        Payload with rhs/solutions of shape (4, 2) and matrix shape (1, 2, 2).
    """
    n = 4
    rhs = np.ones((n, 2), dtype=np.float64)
    solutions = np.full((n, 2), 0.5, dtype=np.float64)

    acc = DenseHdf5Accumulator(tmp_path / ".matrix-staging.h5")
    acc.append_dense_matrix(two_by_two, repeats=1)
    staging_path = acc.finalize()

    return GeneratedDatasetPayload(
        rhs=rhs,
        solutions=solutions,
        matrix_artifact_path=staging_path,
        matrix_size=(2, 2),
        normalization_type="none",
        matrix_norm=1.0,
        matrix_norm_type="spectral",
    )


# ---------------------------------------------------------------------------
# DenseHdf5Accumulator unit tests
# ---------------------------------------------------------------------------


def test_accumulator_single_append_shape(tmp_path: Path, small_matrix: NDArray[np.float64]) -> None:
    """append_dense_matrix(M, 1) produces dataset shape (1, n, n).

    Args:
        tmp_path: Pytest-provided temporary directory.
        small_matrix: 3×3 fixture matrix.
    """
    acc = DenseHdf5Accumulator(tmp_path / "staging.h5")
    acc.append_dense_matrix(small_matrix, repeats=1)
    path = acc.finalize()

    with h5py.File(str(path), "r") as f:
        assert f["matrix"].shape == (1, 3, 3)
        np.testing.assert_allclose(f["matrix"][0], small_matrix)


def test_accumulator_multiple_repeats(tmp_path: Path, small_matrix: NDArray[np.float64]) -> None:
    """repeats=N stores N identical rows.

    Args:
        tmp_path: Pytest-provided temporary directory.
        small_matrix: 3×3 fixture matrix.
    """
    acc = DenseHdf5Accumulator(tmp_path / "staging.h5")
    acc.append_dense_matrix(small_matrix, repeats=3)
    path = acc.finalize()

    with h5py.File(str(path), "r") as f:
        assert f["matrix"].shape == (3, 3, 3)
        for i in range(3):
            np.testing.assert_allclose(f["matrix"][i], small_matrix)

    assert acc.n_samples == 3


def test_accumulator_multi_append_concatenates(
    tmp_path: Path, small_matrix: NDArray[np.float64]
) -> None:
    """Two appends produce rows in order.

    Args:
        tmp_path: Pytest-provided temporary directory.
        small_matrix: 3×3 fixture matrix.
    """
    m2 = small_matrix * 2.0
    acc = DenseHdf5Accumulator(tmp_path / "staging.h5")
    acc.append_dense_matrix(small_matrix, repeats=1)
    acc.append_dense_matrix(m2, repeats=1)
    path = acc.finalize()

    with h5py.File(str(path), "r") as f:
        assert f["matrix"].shape == (2, 3, 3)
        np.testing.assert_allclose(f["matrix"][0], small_matrix)
        np.testing.assert_allclose(f["matrix"][1], m2)


def test_accumulator_matrix_size_property(
    tmp_path: Path, small_matrix: NDArray[np.float64]
) -> None:
    """matrix_size is None before first append and (n, n) after.

    Args:
        tmp_path: Pytest-provided temporary directory.
        small_matrix: 3×3 fixture matrix.
    """
    acc = DenseHdf5Accumulator(tmp_path / "staging.h5")
    assert acc.matrix_size is None
    acc.append_dense_matrix(small_matrix, repeats=1)
    assert acc.matrix_size == (3, 3)


def test_accumulator_finalize_removes_open_handle(
    tmp_path: Path, small_matrix: NDArray[np.float64]
) -> None:
    """finalize() closes the file handle so the path is openable again.

    Args:
        tmp_path: Pytest-provided temporary directory.
        small_matrix: 3×3 fixture matrix.
    """
    acc = DenseHdf5Accumulator(tmp_path / "staging.h5")
    acc.append_dense_matrix(small_matrix, repeats=1)
    path = acc.finalize()
    # Should not raise — handle is closed
    with h5py.File(str(path), "r"):
        pass


def test_accumulator_repeats_zero_raises(tmp_path: Path, small_matrix: NDArray[np.float64]) -> None:
    """repeats < 1 raises ValueError.

    Args:
        tmp_path: Pytest-provided temporary directory.
        small_matrix: 3×3 fixture matrix.
    """
    acc = DenseHdf5Accumulator(tmp_path / "staging.h5")
    with pytest.raises(ValueError, match="repeats must be >= 1"):
        acc.append_dense_matrix(small_matrix, repeats=0)


def test_accumulator_sparse_roundtrip(tmp_path: Path) -> None:
    """append_sparse_components converts to dense before storing.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    indices = np.array([[0, 1], [1, 0]])
    values = np.array([5.0, 3.0])
    acc = DenseHdf5Accumulator(tmp_path / "staging.h5")
    acc.append_sparse_components(indices=indices, values=values, size=(2, 2), repeats=1)
    path = acc.finalize()

    expected = np.array([[0.0, 5.0], [3.0, 0.0]], dtype=np.float64)
    with h5py.File(str(path), "r") as f:
        np.testing.assert_allclose(f["matrix"][0], expected)


# ---------------------------------------------------------------------------
# Hdf5GenerationStorage integration tests
# ---------------------------------------------------------------------------


def test_storage_creates_single_h5_file(
    tmp_path: Path, simple_payload: GeneratedDatasetPayload
) -> None:
    """write_dataset() produces dataset.h5 in the dataset directory.

    Args:
        tmp_path: Pytest-provided temporary directory.
        simple_payload: Fixture payload.
    """
    Hdf5GenerationStorage().write_dataset(tmp_path, simple_payload)
    assert (tmp_path / "dataset.h5").is_file()


def test_storage_staging_file_removed(
    tmp_path: Path, simple_payload: GeneratedDatasetPayload
) -> None:
    """Staging file is removed after write_dataset().

    Args:
        tmp_path: Pytest-provided temporary directory.
        simple_payload: Fixture payload.
    """
    Hdf5GenerationStorage().write_dataset(tmp_path, simple_payload)
    assert not (tmp_path / ".matrix-staging.h5").exists()


def test_storage_h5_contains_expected_datasets(
    tmp_path: Path, simple_payload: GeneratedDatasetPayload
) -> None:
    """dataset.h5 contains matrix, rhs, and solutions datasets.

    Args:
        tmp_path: Pytest-provided temporary directory.
        simple_payload: Fixture payload.
    """
    Hdf5GenerationStorage().write_dataset(tmp_path, simple_payload)
    with h5py.File(str(tmp_path / "dataset.h5"), "r") as f:
        assert "matrix" in f
        assert "rhs" in f
        assert "solutions" in f


def test_storage_manifest_schema_and_format(
    tmp_path: Path, simple_payload: GeneratedDatasetPayload
) -> None:
    """manifest.json has schema v2 and all artifact formats set to 'hdf5'.

    Args:
        tmp_path: Pytest-provided temporary directory.
        simple_payload: Fixture payload.
    """
    Hdf5GenerationStorage().write_dataset(tmp_path, simple_payload)
    manifest = load_dataset_manifest(tmp_path)
    assert str(manifest["schema"]).startswith("neuralls.dataset")
    assert manifest["matrix"]["format"] == "hdf5"
    assert manifest["rhs"]["format"] == "hdf5"
    assert manifest["solutions"]["format"] == "hdf5"


def test_storage_manifest_artifact_paths_and_keys(
    tmp_path: Path, simple_payload: GeneratedDatasetPayload
) -> None:
    """Manifest stores path='dataset.h5' and key='<dataset>' as separate fields.

    Args:
        tmp_path: Pytest-provided temporary directory.
        simple_payload: Fixture payload.
    """
    Hdf5GenerationStorage().write_dataset(tmp_path, simple_payload)
    manifest = load_dataset_manifest(tmp_path)
    assert manifest["matrix"]["path"] == "dataset.h5"
    assert manifest["matrix"]["key"] == "matrix"
    assert manifest["rhs"]["path"] == "dataset.h5"
    assert manifest["rhs"]["key"] == "rhs"
    assert manifest["solutions"]["path"] == "dataset.h5"
    assert manifest["solutions"]["key"] == "solutions"


def test_storage_roundtrip_shapes_and_values(
    tmp_path: Path, two_by_two: NDArray[np.float64]
) -> None:
    """Arrays written by Hdf5GenerationStorage round-trip with correct shapes and values.

    Args:
        tmp_path: Pytest-provided temporary directory.
        two_by_two: 2×2 fixture matrix.
    """
    n = 5
    rhs = np.arange(n * 2, dtype=np.float64).reshape(n, 2)
    solutions = rhs * 0.1

    acc = DenseHdf5Accumulator(tmp_path / ".matrix-staging.h5")
    acc.append_dense_matrix(two_by_two, repeats=n)
    staging = acc.finalize()

    payload = GeneratedDatasetPayload(
        rhs=rhs,
        solutions=solutions,
        matrix_artifact_path=staging,
        matrix_size=(2, 2),
        normalization_type="matrix",
        matrix_norm=7.0,
        matrix_norm_type="spectral",
    )
    Hdf5GenerationStorage().write_dataset(tmp_path, payload)

    arrays = load_training_arrays(tmp_path)
    loaded_rhs = arrays.rhs_source.load_sample(0)
    assert isinstance(arrays.rhs_source, Hdf5ArraySource)
    assert arrays.rhs_source.key == "rhs"
    assert arrays.sample_count == n
    np.testing.assert_allclose(loaded_rhs, rhs[0])


def test_storage_resolve_artifacts_hdf5_key(
    tmp_path: Path, simple_payload: GeneratedDatasetPayload
) -> None:
    """resolve_dataset_artifacts() exposes the key field from the manifest artifact.

    Args:
        tmp_path: Pytest-provided temporary directory.
        simple_payload: Fixture payload.
    """
    Hdf5GenerationStorage().write_dataset(tmp_path, simple_payload)
    artifacts = resolve_dataset_artifacts(tmp_path)
    assert artifacts.rhs.key == "rhs"
    assert artifacts.solutions.key == "solutions"
    assert artifacts.matrix.key == "matrix"
    assert artifacts.rhs.path == tmp_path / "dataset.h5"


def test_entry_from_path_returns_hdf5_entry(tmp_path: Path) -> None:
    """entry_from_path returns Hdf5Entry for .h5 suffix with correct key.

    Args:
        tmp_path: Pytest-provided temporary directory.
    """
    from dlkit.infrastructure.config.data_entries import Hdf5Entry

    path = tmp_path / "dataset.h5"
    path.touch()  # Hdf5Entry validates existence
    entry = entry_from_path(path, name="x", model_input=True, role="feature", key="rhs")
    assert isinstance(entry, Hdf5Entry)
    assert entry.key == "rhs"
    assert entry.path == path


def test_storage_with_parameter_arrays(tmp_path: Path, two_by_two: NDArray[np.float64]) -> None:
    """Parameter arrays are stored in dataset.h5 with '::' key in manifest.

    Args:
        tmp_path: Pytest-provided temporary directory.
        two_by_two: 2×2 fixture matrix.
    """
    params = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)

    acc = DenseHdf5Accumulator(tmp_path / ".matrix-staging.h5")
    acc.append_dense_matrix(two_by_two, repeats=3)
    staging = acc.finalize()

    payload = GeneratedDatasetPayload(
        rhs=np.ones((3, 2), dtype=np.float64),
        solutions=np.zeros((3, 2), dtype=np.float64),
        matrix_artifact_path=staging,
        matrix_size=(2, 2),
        normalization_type="none",
        matrix_norm=1.0,
        matrix_norm_type="spectral",
        parameters_arrays=(params,),
    )
    Hdf5GenerationStorage().write_dataset(tmp_path, payload)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    param_entries = manifest["params"]
    assert len(param_entries) == 1
    assert param_entries[0]["path"] == "dataset.h5"
    assert param_entries[0]["key"] == "parameters_0"
    assert param_entries[0]["format"] == "hdf5"

    with h5py.File(str(tmp_path / "dataset.h5"), "r") as f:
        np.testing.assert_allclose(f["parameters_0"][:], params)
