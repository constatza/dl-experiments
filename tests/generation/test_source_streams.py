from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation.source_streams import (
    bind_sources,
    open_matrix_stream,
    open_vector_stream,
)
from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.platform.storage.datasets import load_dense_training_arrays, resolve_dataset_paths
from dlkit.io import open_sparse_pack


@pytest.fixture
def three_spd_txt_matrices(tmp_path: Path) -> tuple[Path, int]:
    """Three 5×5 SPD matrices written to individual .txt files for glob tests."""
    rng = np.random.default_rng(7)
    mat_dir = tmp_path / "matrices"
    mat_dir.mkdir()
    n = 5
    for i in range(3):
        A = rng.standard_normal((n, n))
        A = A @ A.T + n * np.eye(n)
        np.savetxt(mat_dir / f"A_{i:03d}.txt", A)
    return mat_dir, n


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


def test_build_dataset_persists_residual_trace_pairs(tmp_path: Path) -> None:
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    matrix_path = tmp_path / "matrix.npy"
    np.save(matrix_path, matrix)

    rhs = np.array([[1.0, 2.0]], dtype=np.float64)
    rhs_path = tmp_path / "rhs.npy"
    np.save(rhs_path, rhs)

    out_dir = tmp_path / "residual_dataset"
    build_dataset(
        matrix_path=str(matrix_path),
        dataset_dir=str(out_dir),
        counts={"residual_traces": 4},
        rhs_path=str(rhs_path),
        normalize="none",
        shuffle=False,
        seed=7,
        strategy_overrides={"residual_traces": {"cg_iters": 1}},
    )

    saved_rhs, saved_solutions = load_dense_training_arrays(out_dir)
    assert saved_rhs.shape == (4, 2)
    assert saved_solutions.shape == (4, 2)


def test_build_dataset_persists_residuals_pairs(tmp_path: Path) -> None:
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    matrix_path = tmp_path / "matrix.npy"
    np.save(matrix_path, matrix)

    for idx in range(3):
        np.savetxt(tmp_path / f"sol_{idx:03d}.txt", np.array([1.0 + idx, 0.5 + idx]))

    out_dir = tmp_path / "residuals_dataset"
    build_dataset(
        matrix_path=str(matrix_path),
        dataset_dir=str(out_dir),
        counts={"residuals": 4},
        normalize="none",
        shuffle=False,
        seed=7,
        strategy_overrides={
            "residuals": {
                "cg_iters": 1,
                "solutions_glob": str(tmp_path / "sol_*.txt"),
            }
        },
    )

    saved_rhs, saved_solutions = load_dense_training_arrays(out_dir)
    assert saved_rhs.shape == (4, 2)
    assert saved_solutions.shape == (4, 2)
    np.testing.assert_allclose(saved_solutions @ matrix.T, saved_rhs, atol=1e-10)


def test_build_dataset_persists_gaussian_residual_pairs(tmp_path: Path) -> None:
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    matrix_path = tmp_path / "matrix.npy"
    np.save(matrix_path, matrix)

    out_dir = tmp_path / "gaussian_residual_dataset"
    build_dataset(
        matrix_path=str(matrix_path),
        dataset_dir=str(out_dir),
        counts={"gaussian_residuals": 4},
        normalize="none",
        shuffle=False,
        seed=7,
        strategy_overrides={"gaussian_residuals": {"cg_iters": 1}},
    )

    saved_rhs, saved_solutions = load_dense_training_arrays(out_dir)
    assert saved_rhs.shape == (4, 2)
    assert saved_solutions.shape == (4, 2)
    np.testing.assert_allclose(saved_solutions @ matrix.T, saved_rhs, atol=1e-10)


def test_glob_matrix_gaussian_generates_n_times_m_samples(
    three_spd_txt_matrices: tuple[Path, int],
    tmp_path: Path,
) -> None:
    """build_dataset with a matrix glob and gaussian_forward produces 3 × samples rows."""
    mat_dir, n = three_spd_txt_matrices
    samples_per_matrix = 4
    out_dir = tmp_path / "ds_gaussian"

    build_dataset(
        matrix_path=str(mat_dir / "A_*.txt"),
        dataset_dir=str(out_dir),
        counts={"gaussian_forward": samples_per_matrix},
        normalize="none",
        shuffle=False,
        seed=0,
    )

    rhs, solutions = load_dense_training_arrays(out_dir)
    assert rhs.shape == (3 * samples_per_matrix, n)
    assert solutions.shape == (3 * samples_per_matrix, n)

    pack = open_sparse_pack(resolve_dataset_paths(out_dir).matrix_pack_dir)
    assert pack.n_samples == 3 * samples_per_matrix


def test_glob_matrix_solution_archive_pairs_each_matrix_with_shared_vectors(
    three_spd_txt_matrices: tuple[Path, int],
    tmp_path: Path,
) -> None:
    """solution_archive strategy with a matrix glob applies each solution to every matrix."""
    mat_dir, n = three_spd_txt_matrices
    sol_dir = tmp_path / "solutions"
    sol_dir.mkdir()
    rng = np.random.default_rng(13)
    n_solutions = 5
    for j in range(n_solutions):
        np.savetxt(sol_dir / f"sol_{j:04d}.txt", rng.standard_normal(n))

    out_dir = tmp_path / "ds_archive"
    build_dataset(
        matrix_path=str(mat_dir / "A_*.txt"),
        dataset_dir=str(out_dir),
        counts={"solution_archive": n_solutions},
        normalize="none",
        shuffle=False,
        seed=0,
        strategy_overrides={
            "solution_archive": {
                "solutions_glob": str(sol_dir / "sol_*.txt"),
                "samples": n_solutions,
            }
        },
    )

    rhs, solutions = load_dense_training_arrays(out_dir)
    assert rhs.shape == (3 * n_solutions, n)
    assert solutions.shape == (3 * n_solutions, n)

    pack = open_sparse_pack(resolve_dataset_paths(out_dir).matrix_pack_dir)
    assert pack.n_samples == 3 * n_solutions
