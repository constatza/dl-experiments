"""Tests for generated dataset storage overwrite behavior."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest
import zarr

from neuralls.domain.generation.payloads import GeneratedDatasetPayload
from neuralls.platform.storage.dataset_readers import (
    load_dense_training_arrays,
    load_matrix_dense_sample,
)
from neuralls.platform.storage.datasets import make_generation_dataset_storage
from neuralls.platform.storage.manifest_io import read_dataset_manifest
from neuralls.shared.constants import PARAMETERS_ZARR_PREFIX
from neuralls.shared.types import DatasetFormat, LayoutType


@pytest.fixture(params=["hdf5", "npy", "zarr"])
def dataset_format(request: pytest.FixtureRequest) -> DatasetFormat:
    """All generation storage formats."""
    return cast(DatasetFormat, request.param)


@pytest.fixture
def matrix_samples() -> tuple[np.ndarray, np.ndarray]:
    """Two matrix payloads with distinct sample counts and values."""
    first = np.array(
        [
            [[2.0, 0.0], [0.0, 2.0]],
            [[3.0, 0.0], [0.0, 3.0]],
        ],
        dtype=np.float64,
    )
    second = np.array(
        [
            [[5.0, 1.0], [1.0, 5.0]],
            [[6.0, 1.0], [1.0, 6.0]],
            [[7.0, 1.0], [1.0, 7.0]],
        ],
        dtype=np.float64,
    )
    return first, second


@pytest.fixture
def vector_samples() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """RHS and solution samples aligned with ``matrix_samples``."""
    first_rhs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    first_solutions = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    second_rhs = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float64)
    second_solutions = np.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]], dtype=np.float64)
    return (first_rhs, first_solutions), (second_rhs, second_solutions)


@pytest.fixture
def payload_factory(
    tmp_path: Path,
) -> Callable[
    [DatasetFormat, str, np.ndarray, np.ndarray, np.ndarray, bool],
    GeneratedDatasetPayload,
]:
    """Create storage-ready payloads with format-specific staged matrix artifacts."""

    def make_payload(
        dataset_format: DatasetFormat,
        name: str,
        matrix: np.ndarray,
        rhs: np.ndarray,
        solutions: np.ndarray,
        include_optional: bool,
    ) -> GeneratedDatasetPayload:
        staging_path = tmp_path / f"{name}-matrix"
        matrix_artifact_path = _write_staged_matrix(dataset_format, staging_path, matrix)
        row_kind_codes: np.ndarray | None = None
        matrix_sample_index: np.ndarray | None = None
        parameters_arrays: tuple[np.ndarray, ...] = ()
        if include_optional:
            row_kind_codes = np.arange(rhs.shape[0], dtype=np.uint8)
            matrix_sample_index = np.arange(rhs.shape[0], dtype=np.int64)
            parameters_arrays = (np.arange(rhs.shape[0], dtype=np.float64).reshape(-1, 1),)
        return GeneratedDatasetPayload(
            rhs=rhs,
            solutions=solutions,
            matrix_artifact_path=matrix_artifact_path,
            matrix_size=tuple(int(dim) for dim in matrix.shape[1:]),
            normalization_type="none",
            matrix_norm=1.0,
            matrix_norm_type="spectral",
            parameters_arrays=parameters_arrays,
            layout=LayoutType.MANY_MATRICES,
            row_kind_codes=row_kind_codes,
            matrix_sample_index=matrix_sample_index,
        )

    return make_payload


def _write_staged_matrix(
    dataset_format: DatasetFormat,
    staging_path: Path,
    matrix: np.ndarray,
) -> Path:
    match dataset_format:
        case "hdf5":
            h5_path = staging_path.with_suffix(".h5")
            with h5py.File(str(h5_path), "w") as out:
                out.create_dataset("matrix", data=matrix)
            return h5_path
        case "npy":
            npy_path = staging_path.with_suffix(".npy")
            np.save(npy_path, matrix)
            return npy_path
        case "zarr":
            zarr_path = staging_path.with_suffix(".zarr")
            arr = zarr.open_array(
                str(zarr_path),
                mode="w",
                shape=matrix.shape,
                chunks=(1, *matrix.shape[1:]),
                dtype="float64",
            )
            arr[:] = matrix
            return zarr_path


def test_same_format_regeneration_replaces_dataset_artifacts(
    tmp_path: Path,
    dataset_format: DatasetFormat,
    matrix_samples: tuple[np.ndarray, np.ndarray],
    vector_samples: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    payload_factory: Callable[
        [DatasetFormat, str, np.ndarray, np.ndarray, np.ndarray, bool],
        GeneratedDatasetPayload,
    ],
) -> None:
    dataset_dir = tmp_path / f"{dataset_format}-dataset"
    storage = make_generation_dataset_storage(dataset_format)
    first_matrix, second_matrix = matrix_samples
    (first_rhs, first_solutions), (second_rhs, second_solutions) = vector_samples

    storage.write_dataset(
        dataset_dir,
        payload_factory(
            dataset_format,
            "first",
            first_matrix,
            first_rhs,
            first_solutions,
            True,
        ),
    )
    storage.write_dataset(
        dataset_dir,
        payload_factory(
            dataset_format,
            "second",
            second_matrix,
            second_rhs,
            second_solutions,
            False,
        ),
    )

    rhs, solutions = load_dense_training_arrays(dataset_dir)
    np.testing.assert_allclose(rhs, second_rhs)
    np.testing.assert_allclose(solutions, second_solutions)
    np.testing.assert_allclose(load_matrix_dense_sample(dataset_dir, 0), second_matrix[0])
    manifest = read_dataset_manifest(dataset_dir)
    assert manifest.rhs.shape == second_rhs.shape
    assert manifest.solutions.shape == second_solutions.shape
    assert manifest.matrix.n_matrix_samples == second_matrix.shape[0]


def test_same_format_regeneration_removes_stale_optional_artifacts(
    tmp_path: Path,
    dataset_format: DatasetFormat,
    matrix_samples: tuple[np.ndarray, np.ndarray],
    vector_samples: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    payload_factory: Callable[
        [DatasetFormat, str, np.ndarray, np.ndarray, np.ndarray, bool],
        GeneratedDatasetPayload,
    ],
) -> None:
    dataset_dir = tmp_path / f"{dataset_format}-dataset"
    storage = make_generation_dataset_storage(dataset_format)
    first_matrix, second_matrix = matrix_samples
    (first_rhs, first_solutions), (second_rhs, second_solutions) = vector_samples

    storage.write_dataset(
        dataset_dir,
        payload_factory(
            dataset_format,
            "first",
            first_matrix,
            first_rhs,
            first_solutions,
            True,
        ),
    )
    storage.write_dataset(
        dataset_dir,
        payload_factory(
            dataset_format,
            "second",
            second_matrix,
            second_rhs,
            second_solutions,
            False,
        ),
    )

    manifest = read_dataset_manifest(dataset_dir)
    assert manifest.row_kind is None
    assert manifest.matrix_sample_index is None
    assert manifest.params == ()
    _assert_optional_artifacts_removed(dataset_format, dataset_dir)


def test_hdf5_replaces_existing_dataset_file(
    tmp_path: Path,
    matrix_samples: tuple[np.ndarray, np.ndarray],
    vector_samples: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    payload_factory: Callable[
        [DatasetFormat, str, np.ndarray, np.ndarray, np.ndarray, bool],
        GeneratedDatasetPayload,
    ],
) -> None:
    dataset_dir = tmp_path / "hdf5-dataset"
    storage = make_generation_dataset_storage("hdf5")
    first_matrix, second_matrix = matrix_samples
    (first_rhs, first_solutions), (second_rhs, second_solutions) = vector_samples

    storage.write_dataset(
        dataset_dir,
        payload_factory(
            "hdf5",
            "first",
            first_matrix,
            first_rhs,
            first_solutions,
            True,
        ),
    )
    storage.write_dataset(
        dataset_dir,
        payload_factory(
            "hdf5",
            "second",
            second_matrix,
            second_rhs,
            second_solutions,
            False,
        ),
    )

    with h5py.File(str(dataset_dir / "dataset.h5"), "r") as out:
        assert sorted(out.keys()) == ["matrix", "rhs", "solutions"]
        np.testing.assert_allclose(out["rhs"][:], second_rhs)


def _assert_optional_artifacts_removed(dataset_format: DatasetFormat, dataset_dir: Path) -> None:
    match dataset_format:
        case "hdf5":
            with h5py.File(str(dataset_dir / "dataset.h5"), "r") as out:
                assert "row_kind" not in out
                assert "matrix_sample_index" not in out
                assert PARAMETERS_ZARR_PREFIX + "0" not in out
        case "npy":
            assert not (dataset_dir / "row_kind.npy").exists()
            assert not (dataset_dir / "matrix_sample_index.npy").exists()
            assert not (dataset_dir / f"{PARAMETERS_ZARR_PREFIX}0.npy").exists()
        case "zarr":
            group_dir = dataset_dir / "dataset.zarr"
            assert not (group_dir / "row_kind").exists()
            assert not (group_dir / "matrix_sample_index").exists()
            assert not (group_dir / f"{PARAMETERS_ZARR_PREFIX}0").exists()
