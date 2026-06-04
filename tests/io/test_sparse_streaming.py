"""Tests for DiskBackedSparseAccumulator streaming sparse pack assembly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlkit.io import open_sparse_pack
from neuralls.platform.storage.datasets import (
    DiskBackedSparseAccumulator,
    SparsePackAccumulator,
    as_sparse_pack_reader,
)


@pytest.fixture
def small_matrix() -> np.ndarray:
    """Return a 4x4 sparse numpy array for use in tests."""
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 1] = 1.0
    m[1, 2] = 2.0
    m[2, 0] = 3.0
    m[3, 3] = 4.0
    return m


def test_disk_backed_accumulator_matches_in_memory(
    tmp_path: Path, small_matrix: np.ndarray
) -> None:
    """DiskBackedSparseAccumulator produces the same nnz_ptr and arrays as SparsePackAccumulator."""
    in_mem = SparsePackAccumulator()
    in_mem.append_dense_matrix(small_matrix, repeats=1)
    in_mem.append_dense_matrix(small_matrix, repeats=1)
    ref_idx, ref_val, ref_ptr, ref_size = in_mem.build_arrays()

    staging_dir = tmp_path / "staging"
    disk_acc = DiskBackedSparseAccumulator(staging_dir)
    disk_acc.append_dense_matrix(small_matrix, repeats=1)
    disk_acc.append_dense_matrix(small_matrix, repeats=1)
    got_idx, got_val, got_ptr, got_size = disk_acc.build_arrays()

    assert got_size == ref_size
    np.testing.assert_array_equal(got_ptr, ref_ptr)
    np.testing.assert_array_equal(np.sort(got_val), np.sort(ref_val))


def test_disk_backed_accumulator_round_trips_via_open_sparse_pack(
    tmp_path: Path, small_matrix: np.ndarray
) -> None:
    """Pack assembled by DiskBackedSparseAccumulator can be read back via open_sparse_pack."""
    staging_dir = tmp_path / "staging"
    disk_acc = DiskBackedSparseAccumulator(staging_dir)
    disk_acc.append_dense_matrix(small_matrix, repeats=2)
    disk_acc.build_arrays()

    pack = as_sparse_pack_reader(open_sparse_pack(disk_acc.pack_dir))
    assert pack.n_samples == 2
    for sample_idx in range(2):
        dense = pack.collect(sample_idx).to_dense().numpy()
        np.testing.assert_allclose(dense, small_matrix)


def test_disk_backed_accumulator_handles_repeats(tmp_path: Path, small_matrix: np.ndarray) -> None:
    """DiskBackedSparseAccumulator correctly expands repeats > 1 in the nnz_ptr."""
    staging_dir = tmp_path / "staging"
    disk_acc = DiskBackedSparseAccumulator(staging_dir)
    disk_acc.append_dense_matrix(small_matrix, repeats=3)
    _, _, nnz_ptr, size = disk_acc.build_arrays()

    rows, cols = np.nonzero(small_matrix)
    nnz_per_sample = int(rows.size)

    assert size == (4, 4)
    assert nnz_ptr.size == 4  # 3 samples + 1
    expected_ptr = np.array(
        [0, nnz_per_sample, 2 * nnz_per_sample, 3 * nnz_per_sample], dtype=np.int64
    )
    np.testing.assert_array_equal(nnz_ptr, expected_ptr)
