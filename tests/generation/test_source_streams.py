from __future__ import annotations

from pathlib import Path

import numpy as np

from neuralls.generation.source_streams import (
    bind_sources,
    open_matrix_stream,
    open_vector_stream,
)
from neuralls.generation.orchestration import build_dataset
from neuralls.io.dataset_storage import load_dense_training_arrays, resolve_dataset_paths
from dlkit.tools.io.sparse import open_sparse_pack


def test_open_matrix_stream_from_npy_stack(tmp_path: Path) -> None:
    matrix_stack = np.array(
        [
            [[2.0, 0.0], [0.0, 3.0]],
            [[4.0, 1.0], [1.0, 5.0]],
        ],
        dtype=np.float64,
    )
    matrix_path = tmp_path / "matrices.npy"
    np.save(matrix_path, matrix_stack)

    stream = open_matrix_stream(str(matrix_path))
    assert stream.sample_ids == (0, 1)

    dense_one = stream.load_dense_sample(1)
    np.testing.assert_allclose(dense_one.matrix, matrix_stack[1])

    sparse_zero = stream.load_sparse_sample(0)
    reconstructed = np.zeros(sparse_zero.size, dtype=np.float64)
    reconstructed[sparse_zero.indices[0], sparse_zero.indices[1]] = sparse_zero.values
    np.testing.assert_allclose(reconstructed, matrix_stack[0])


def test_bind_sources_broadcasts_single_matrix_to_many_rhs() -> None:
    bindings = bind_sources(
        matrix_ids=(0,),
        rhs_ids=(0, 1, 2),
        solution_ids=None,
    )
    assert [item.matrix_sample_id for item in bindings] == [0, 0, 0]
    assert [item.rhs_sample_id for item in bindings] == [0, 1, 2]


def test_build_dataset_streams_matrix_stack_without_dense_batch(tmp_path: Path) -> None:
    matrix_stack = np.array(
        [
            [[3.0, 1.0], [1.0, 2.0]],
            [[6.0, 0.0], [0.0, 4.0]],
        ],
        dtype=np.float64,
    )
    matrix_path = tmp_path / "matrix_stack.npy"
    np.save(matrix_path, matrix_stack)

    out_dir = tmp_path / "dataset"
    build_dataset(
        matrix_path=str(matrix_path),
        dataset_dir=str(out_dir),
        counts={"neutral_ones": 1},
        normalize="none",
        shuffle=False,
        seed=42,
    )

    rhs, solutions = load_dense_training_arrays(out_dir)
    assert rhs.shape == (2, 2)
    assert solutions.shape == (2, 2)

    pack = open_sparse_pack(resolve_dataset_paths(out_dir).matrix_pack_dir)
    assert pack.n_samples == 2
    dense0 = pack.build_torch_sparse(0).to_dense().numpy()
    dense1 = pack.build_torch_sparse(1).to_dense().numpy()
    np.testing.assert_allclose(dense0, matrix_stack[0])
    np.testing.assert_allclose(dense1, matrix_stack[1])


def test_open_vector_stream_from_npy_stack(tmp_path: Path) -> None:
    rhs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    rhs_path = tmp_path / "rhs.npy"
    np.save(rhs_path, rhs)

    stream = open_vector_stream(str(rhs_path))
    assert stream.sample_ids == (0, 1)
    np.testing.assert_allclose(stream.load_sample(1).vector, rhs[1])


def test_single_matrix_not_broadcasted_in_sparse_pack(tmp_path: Path) -> None:
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    matrix_path = tmp_path / "matrix.npy"
    np.save(matrix_path, matrix)

    rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    rhs_path = tmp_path / "rhs.npy"
    np.save(rhs_path, rhs)

    out_dir = tmp_path / "dataset_single_matrix"
    build_dataset(
        matrix_path=str(matrix_path),
        dataset_dir=str(out_dir),
        counts={"neutral_ones": 1},
        rhs_path=str(rhs_path),
        normalize="none",
        shuffle=False,
        seed=7,
    )

    saved_rhs, saved_solutions = load_dense_training_arrays(out_dir)
    assert saved_rhs.shape == (3, 2)
    assert saved_solutions.shape == (3, 2)

    pack = open_sparse_pack(resolve_dataset_paths(out_dir).matrix_pack_dir)
    assert pack.n_samples == 1
    dense_last = pack.build_torch_sparse(sample_index=2).to_dense().numpy()
    np.testing.assert_allclose(dense_last, matrix)
