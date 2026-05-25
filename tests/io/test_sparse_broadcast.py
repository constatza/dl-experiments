from __future__ import annotations

from pathlib import Path

import numpy as np

from dlkit.io import open_sparse_pack
from neuralls.platform.storage.datasets import (
    SparsePackAccumulator,
    resolve_dataset_paths,
    save_dataset_from_sparse,
)


def test_sparse_accumulator_broadcasts_single_matrix_components(tmp_path: Path) -> None:
    matrix = np.array([[2.0, 0.0], [1.0, 3.0]], dtype=np.float64)

    acc = SparsePackAccumulator()
    acc.append_dense_matrix(matrix, repeats=3)
    indices, values, nnz_ptr, size = acc.build_arrays()

    assert size == (2, 2)
    assert indices.shape == (2, 9)
    assert values.shape == (9,)
    np.testing.assert_array_equal(nnz_ptr, np.array([0, 3, 6, 9], dtype=np.int64))


def test_save_dataset_from_sparse_keeps_broadcasted_matrix_samples(tmp_path: Path) -> None:
    matrix = np.array([[4.0, 0.0], [0.0, 5.0]], dtype=np.float64)
    acc = SparsePackAccumulator()
    acc.append_dense_matrix(matrix, repeats=4)
    indices, values, nnz_ptr, size = acc.build_arrays()

    rhs = np.tile(np.array([1.0, 2.0], dtype=np.float64), (4, 1))
    solutions = np.tile(np.array([0.1, 0.2], dtype=np.float64), (4, 1))
    save_dataset_from_sparse(
        dataset_dir=tmp_path,
        rhs=rhs,
        solutions=solutions,
        indices=indices,
        values=values,
        nnz_ptr=nnz_ptr,
        size=size,
        normalization_type="none",
        matrix_norm=1.0,
        matrix_norm_type="spectral",
        scale_metadata={},
    )

    pack = open_sparse_pack(resolve_dataset_paths(tmp_path).matrix_pack_dir)
    assert pack.n_samples == 4
    for sample_idx in range(4):
        dense = pack.build_torch_sparse(sample_idx).to_dense().numpy()
        np.testing.assert_allclose(dense, matrix)


def test_save_dataset_from_sparse_stores_raw_matrix(tmp_path: Path) -> None:
    raw_matrix = np.array([[8.0, 0.0], [2.0, 4.0]], dtype=np.float64)
    value_scale = 2.0
    normalized_matrix = raw_matrix / value_scale

    acc = SparsePackAccumulator()
    acc.append_dense_matrix(normalized_matrix, repeats=1)
    indices, values, nnz_ptr, size = acc.build_arrays()

    rhs = np.array([[1.0, 0.0]], dtype=np.float64)
    solutions = np.array([[0.0, 1.0]], dtype=np.float64)
    save_dataset_from_sparse(
        dataset_dir=tmp_path,
        rhs=rhs,
        solutions=solutions,
        indices=indices,
        values=values,
        nnz_ptr=nnz_ptr,
        size=size,
        normalization_type="matrix",
        matrix_norm=1.0,
        matrix_norm_type="spectral",
        matrix_value_scale=value_scale,
        scale_metadata={},
    )

    pack = open_sparse_pack(resolve_dataset_paths(tmp_path).matrix_pack_dir)
    dense = pack.build_torch_sparse(0).to_dense().numpy()
    np.testing.assert_allclose(dense, raw_matrix)
