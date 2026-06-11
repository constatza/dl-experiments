"""Tests for DenseZarrAccumulator and DenseDatasetWriter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray

from neuralls.domain.generation.payloads import GeneratedDatasetPayload
from neuralls.platform.storage.datasets import (
    DenseDatasetWriter,
    DenseZarrAccumulator,
    load_dataset_manifest,
    resolve_dataset_paths,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random-number generator for deterministic test data.

    Returns:
        A Generator seeded to 42.
    """
    return np.random.default_rng(seed=42)


@pytest.fixture
def small_matrix() -> NDArray[np.float64]:
    """4×4 sparse-ish dense matrix for single-append tests.

    Returns:
        Array of shape (4, 4) with a few nonzero entries.
    """
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 1] = 1.0
    m[1, 2] = 2.0
    m[2, 0] = 3.0
    m[3, 3] = 4.0
    return m


@pytest.fixture
def two_by_two_matrix() -> NDArray[np.float64]:
    """2×2 diagonal matrix.

    Returns:
        Array of shape (2, 2).
    """
    return np.array([[4.0, 0.0], [0.0, 5.0]], dtype=np.float64)


@pytest.fixture
def simple_payload(
    tmp_path: Path,
    two_by_two_matrix: NDArray[np.float64],
) -> GeneratedDatasetPayload:
    """A minimal GeneratedDatasetPayload backed by a zarr accumulator.

    Args:
        tmp_path: Pytest-provided temporary directory.
        two_by_two_matrix: 2×2 fixture matrix.

    Returns:
        Payload with rhs/solutions of shape (4, 2) and matrix shape (1, 2, 2).
    """
    rhs = np.tile(np.array([1.0, 2.0], dtype=np.float64), (4, 1))
    solutions = np.tile(np.array([0.1, 0.2], dtype=np.float64), (4, 1))

    acc = DenseZarrAccumulator(tmp_path / "matrix.zarr")
    acc.append_dense_matrix(two_by_two_matrix, repeats=1)
    zarr_path = acc.finalize()

    return GeneratedDatasetPayload(
        rhs=rhs,
        solutions=solutions,
        matrix_pack_path=zarr_path,
        matrix_size=(2, 2),
        normalization_type="none",
        matrix_norm=1.0,
        matrix_norm_type="spectral",
    )


# ---------------------------------------------------------------------------
# DenseZarrAccumulator — unit tests
# ---------------------------------------------------------------------------


def test_single_append_shape_and_values(tmp_path: Path, small_matrix: NDArray[np.float64]) -> None:
    """Single append_dense_matrix(M, repeats=1) produces zarr shape (1, n, n) with correct values.

    Args:
        tmp_path: Pytest-provided temporary directory.
        small_matrix: 4×4 fixture matrix.
    """
    acc = DenseZarrAccumulator(tmp_path / "matrix.zarr")
    acc.append_dense_matrix(small_matrix, repeats=1)
    path = acc.finalize()
    arr = zarr.open_array(str(path), mode="r")
    assert arr.shape == (1, 4, 4)
    np.testing.assert_allclose(np.asarray(arr[0]), small_matrix)


def test_multiple_appends_accumulate_rows(
    tmp_path: Path, small_matrix: NDArray[np.float64]
) -> None:
    """Multiple append_dense_matrix calls concatenate rows in order.

    Args:
        tmp_path: Pytest-provided temporary directory.
        small_matrix: 4×4 fixture matrix.
    """
    m2 = small_matrix * 2.0
    acc = DenseZarrAccumulator(tmp_path / "matrix.zarr")
    acc.append_dense_matrix(small_matrix, repeats=1)
    acc.append_dense_matrix(m2, repeats=1)
    arr = zarr.open_array(str(acc.finalize()), mode="r")
    assert arr.shape[0] == 2
    np.testing.assert_allclose(np.asarray(arr[0]), small_matrix)
    np.testing.assert_allclose(np.asarray(arr[1]), m2)


def test_broadcast_path_writes_one_row(
    tmp_path: Path, two_by_two_matrix: NDArray[np.float64]
) -> None:
    """Single matrix with repeats=N stores only one zarr slice; n_samples reflects N.

    The accumulator stores each repeat as a distinct row so matrix.zarr[i] maps
    to rhs[i] row-for-row, giving shape (N, n, n).  Broadcast detection (one
    unique matrix) happens at manifest-write time via n_matrix_samples == 1.
    This test instead checks that n_samples == N and that every stored row
    matches the original matrix.

    Args:
        tmp_path: Pytest-provided temporary directory.
        two_by_two_matrix: 2×2 fixture matrix.
    """
    n_repeats = 4
    acc = DenseZarrAccumulator(tmp_path / "matrix.zarr")
    acc.append_dense_matrix(two_by_two_matrix, repeats=n_repeats)
    path = acc.finalize()
    arr = zarr.open_array(str(path), mode="r")
    assert arr.shape[0] == n_repeats
    assert acc.n_samples == n_repeats
    for i in range(n_repeats):
        np.testing.assert_allclose(np.asarray(arr[i]), two_by_two_matrix)


def test_finalize_returns_zarr_path(tmp_path: Path, small_matrix: NDArray[np.float64]) -> None:
    """finalize() returns the configured zarr path and the directory exists.

    Args:
        tmp_path: Pytest-provided temporary directory.
        small_matrix: 4×4 fixture matrix.
    """
    zarr_path = tmp_path / "matrix.zarr"
    acc = DenseZarrAccumulator(zarr_path)
    acc.append_dense_matrix(small_matrix, repeats=1)
    result = acc.finalize()
    assert result == zarr_path
    assert result.exists()


def test_finalize_idempotency(tmp_path: Path, small_matrix: NDArray[np.float64]) -> None:
    """Calling finalize() twice returns identical paths without error.

    Args:
        tmp_path: Pytest-provided temporary directory.
        small_matrix: 4×4 fixture matrix.
    """
    acc = DenseZarrAccumulator(tmp_path / "matrix.zarr")
    acc.append_dense_matrix(small_matrix, repeats=1)
    p1 = acc.finalize()
    p2 = acc.finalize()
    assert p1 == p2


def test_matrix_size_property(tmp_path: Path, small_matrix: NDArray[np.float64]) -> None:
    """matrix_size is None before any append and (n, n) after first append.

    Args:
        tmp_path: Pytest-provided temporary directory.
        small_matrix: 4×4 fixture matrix.
    """
    acc = DenseZarrAccumulator(tmp_path / "matrix.zarr")
    assert acc.matrix_size is None
    acc.append_dense_matrix(small_matrix, repeats=1)
    assert acc.matrix_size == (4, 4)


def test_n_samples_property(tmp_path: Path, small_matrix: NDArray[np.float64]) -> None:
    """n_samples is cumulative sum of all repeats across multiple appends.

    Args:
        tmp_path: Pytest-provided temporary directory.
        small_matrix: 4×4 fixture matrix.
    """
    acc = DenseZarrAccumulator(tmp_path / "matrix.zarr")
    assert acc.n_samples == 0
    acc.append_dense_matrix(small_matrix, repeats=2)
    assert acc.n_samples == 2
    acc.append_dense_matrix(small_matrix * 0.5, repeats=3)
    assert acc.n_samples == 5


# ---------------------------------------------------------------------------
# DenseDatasetWriter — integration tests
# ---------------------------------------------------------------------------


def test_writer_creates_manifest(tmp_path: Path, simple_payload: GeneratedDatasetPayload) -> None:
    """write_dataset() creates manifest.json with schema neuralls.dataset.v2.

    Args:
        tmp_path: Pytest-provided temporary directory.
        simple_payload: Fixture payload ready for writing.
    """
    DenseDatasetWriter().write_dataset(tmp_path, simple_payload)
    paths = resolve_dataset_paths(tmp_path)
    assert paths.manifest_path.exists()
    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "neuralls.dataset.v2"


def test_writer_creates_rhs_and_solutions_zarr(
    tmp_path: Path,
    two_by_two_matrix: NDArray[np.float64],
) -> None:
    """write_dataset() produces rhs.zarr/ and solutions.zarr/ with correct shapes.

    Args:
        tmp_path: Pytest-provided temporary directory.
        two_by_two_matrix: 2×2 fixture matrix.
    """
    n_samples = 3
    rhs = np.ones((n_samples, 2), dtype=np.float64)
    solutions = np.zeros((n_samples, 2), dtype=np.float64)

    acc = DenseZarrAccumulator(tmp_path / "matrix.zarr")
    acc.append_dense_matrix(two_by_two_matrix, repeats=n_samples)
    zarr_path = acc.finalize()

    payload = GeneratedDatasetPayload(
        rhs=rhs,
        solutions=solutions,
        matrix_pack_path=zarr_path,
        matrix_size=(2, 2),
        normalization_type="matrix",
        matrix_norm=1.0,
        matrix_norm_type="spectral",
    )
    DenseDatasetWriter().write_dataset(tmp_path, payload)
    paths = resolve_dataset_paths(tmp_path)

    rhs_arr = zarr.open_array(str(paths.rhs_path), mode="r")
    sol_arr = zarr.open_array(str(paths.solutions_path), mode="r")
    assert rhs_arr.shape == (n_samples, 2)
    assert sol_arr.shape == (n_samples, 2)
    np.testing.assert_allclose(np.asarray(rhs_arr), rhs)
    np.testing.assert_allclose(np.asarray(sol_arr), solutions)


def test_load_dataset_manifest_roundtrip(
    tmp_path: Path, simple_payload: GeneratedDatasetPayload
) -> None:
    """Manifest written by DenseDatasetWriter is readable by load_dataset_manifest.

    Args:
        tmp_path: Pytest-provided temporary directory.
        simple_payload: Fixture payload ready for writing.
    """
    DenseDatasetWriter().write_dataset(tmp_path, simple_payload)
    manifest = load_dataset_manifest(tmp_path)
    assert isinstance(manifest, dict)
    assert str(manifest.get("schema", "")).startswith("neuralls.dataset")
    assert "matrix" in manifest
    assert "rhs" in manifest
    assert "solutions" in manifest
