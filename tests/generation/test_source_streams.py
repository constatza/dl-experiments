from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation.source_streams import (
    EnumerateBy,
    bind_sources,
    open_matrix_stream,
    open_vector_stream,
    GlobMatrixStream,
    GlobVectorStream,
    _enumerate_files,
)
from neuralls.composition.generation.dataset_builder import build_dataset
import zarr
from neuralls.platform.storage.datasets import (
    load_dense_training_arrays,
    load_matrix_dense_sample,
    resolve_dataset_paths,
)


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
        counts={"neutral_ones": 2},
        normalize="none",
        shuffle=False,
        seed=42,
    )

    rhs, solutions = load_dense_training_arrays(out_dir)
    assert rhs.shape == (2, 2)
    assert solutions.shape == (2, 2)

    mat_arr = zarr.open_array(str(resolve_dataset_paths(out_dir).matrix_zarr_dir), mode="r")
    assert mat_arr.shape[0] == 2
    dense0 = load_matrix_dense_sample(out_dir, 0)
    dense1 = load_matrix_dense_sample(out_dir, 1)
    np.testing.assert_allclose(dense0, matrix_stack[0])
    np.testing.assert_allclose(dense1, matrix_stack[1])


def test_open_vector_stream_from_npy_stack(tmp_path: Path) -> None:
    rhs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    rhs_path = tmp_path / "rhs.npy"
    np.save(rhs_path, rhs)

    stream = open_vector_stream(str(rhs_path))
    assert stream.sample_ids == (0, 1)
    np.testing.assert_allclose(stream.load_sample(1).vector, rhs[1])


def test_single_matrix_stored_once_in_zarr(tmp_path: Path) -> None:
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

    mat_arr = zarr.open_array(str(resolve_dataset_paths(out_dir).matrix_zarr_dir), mode="r")
    assert mat_arr.shape[0] == 1
    dense_last = load_matrix_dense_sample(out_dir, 0)
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


def test_glob_matrix_gaussian_uses_global_sample_budget(
    three_spd_txt_matrices: tuple[Path, int],
    tmp_path: Path,
) -> None:
    """Multi-matrix gaussian generation treats counts as a global budget."""
    mat_dir, n = three_spd_txt_matrices
    total_samples = 4
    out_dir = tmp_path / "ds_gaussian"

    build_dataset(
        matrix_path=str(mat_dir / "A_*.txt"),
        dataset_dir=str(out_dir),
        counts={"gaussian_forward": total_samples},
        normalize="none",
        shuffle=False,
        seed=0,
    )

    rhs, solutions = load_dense_training_arrays(out_dir)
    assert rhs.shape == (total_samples, n)
    assert solutions.shape == (total_samples, n)

    mat_arr = zarr.open_array(str(resolve_dataset_paths(out_dir).matrix_zarr_dir), mode="r")
    assert mat_arr.shape[0] == total_samples


def test_glob_matrix_solution_archive_uses_global_sample_budget(
    three_spd_txt_matrices: tuple[Path, int],
    tmp_path: Path,
) -> None:
    """Multi-matrix solution_archive treats counts as a global budget."""
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
    assert rhs.shape == (n_solutions, n)
    assert solutions.shape == (n_solutions, n)

    mat_arr = zarr.open_array(str(resolve_dataset_paths(out_dir).matrix_zarr_dir), mode="r")
    assert mat_arr.shape[0] == n_solutions


# ---------------------------------------------------------------------------
# enumerate_by tests
# ---------------------------------------------------------------------------


@pytest.fixture
def arbitrary_named_txt_matrices(tmp_path: Path) -> tuple[Path, int]:
    """Three 5×5 SPD matrices with parameter-encoded filenames (no sequential integer ID)."""
    rng = np.random.default_rng(99)
    mat_dir = tmp_path / "arbitrary_matrices"
    mat_dir.mkdir()
    n = 5
    for E1, E2 in [(3000, 78000), (5000, 100000), (1000, 50000)]:
        A = rng.standard_normal((n, n))
        A = A @ A.T + n * np.eye(n)
        np.savetxt(mat_dir / f"E1_{E1}_E2_{E2}_matrix.txt", A)
    return mat_dir, n


@pytest.fixture
def subdomain_named_txt_matrices(tmp_path: Path) -> tuple[Path, int]:
    """Three SPD matrices whose stems would collide under the default trailing-digit regex."""
    rng = np.random.default_rng(123)
    mat_dir = tmp_path / "subdomain_matrices"
    mat_dir.mkdir()
    n = 4
    parameter_sets = [
        (10554158, 20662154, 19907116, 20715669),
        (10846921, 25611823, 31377296, 28205425),
        (12000001, 22000002, 32000003, 42000004),
    ]
    for e1, e2, e3, e4 in parameter_sets:
        A = rng.standard_normal((n, n))
        A = A @ A.T + n * np.eye(n)
        np.savetxt(
            mat_dir / f"E1_{e1}_E2_{e2}_E3_{e3}_E4_{e4}_subdomain_1_Kaa.txt",
            A,
        )
    return mat_dir, n


def test_enumerate_files_assigns_sequential_ids_in_name_order(tmp_path: Path) -> None:
    paths = [tmp_path / f"file_z_{suffix}.txt" for suffix in ["c", "a", "b"]]
    for p in paths:
        p.touch()

    result = _enumerate_files(paths, EnumerateBy.NAME)

    assert sorted(result.keys()) == [0, 1, 2]
    assert [result[i].name for i in range(3)] == [
        "file_z_a.txt",
        "file_z_b.txt",
        "file_z_c.txt",
    ]


def test_enumerate_files_assigns_sequential_ids_by_mtime(tmp_path: Path) -> None:
    import time

    paths = []
    for suffix in ["c", "a", "b"]:
        p = tmp_path / f"file_{suffix}.txt"
        p.touch()
        paths.append(p)
        time.sleep(0.01)

    result = _enumerate_files(paths, EnumerateBy.MTIME)

    # Creation order was c, a, b — IDs should follow mtime order
    assert result[0].name == "file_c.txt"
    assert result[1].name == "file_a.txt"
    assert result[2].name == "file_b.txt"


def test_glob_matrix_stream_enumerate_by_name_with_arbitrary_filenames(
    arbitrary_named_txt_matrices: tuple[Path, int],
) -> None:
    mat_dir, n = arbitrary_named_txt_matrices
    stream = GlobMatrixStream(str(mat_dir / "E1_*_matrix.txt"), enumerate_by=EnumerateBy.NAME)

    assert stream.sample_ids == (0, 1, 2)
    sample = stream.load_dense_sample(0)
    assert sample.matrix.shape == (n, n)


def test_glob_vector_stream_enumerate_by_name_with_arbitrary_filenames(tmp_path: Path) -> None:
    vec_dir = tmp_path / "vecs"
    vec_dir.mkdir()
    rng = np.random.default_rng(42)
    for E1, E2 in [(3000, 78000), (5000, 100000), (1000, 50000)]:
        np.savetxt(vec_dir / f"E1_{E1}_E2_{E2}_rhs.txt", rng.standard_normal(5))

    stream = GlobVectorStream(str(vec_dir / "E1_*_rhs.txt"), enumerate_by=EnumerateBy.NAME)

    assert stream.sample_ids == (0, 1, 2)
    assert stream.load_sample(0).vector.shape == (5,)


def test_open_matrix_stream_glob_with_enumerate_by_name(
    arbitrary_named_txt_matrices: tuple[Path, int],
) -> None:
    mat_dir, _ = arbitrary_named_txt_matrices
    stream = open_matrix_stream(
        str(mat_dir / "E1_*_matrix.txt"),
        enumerate_by=EnumerateBy.NAME,
    )

    assert stream.sample_ids == (0, 1, 2)


def test_glob_matrix_stream_enumerate_by_name_is_reproducible(
    arbitrary_named_txt_matrices: tuple[Path, int],
) -> None:
    mat_dir, _ = arbitrary_named_txt_matrices
    expr = str(mat_dir / "E1_*_matrix.txt")

    stream1 = GlobMatrixStream(expr, enumerate_by=EnumerateBy.NAME)
    stream2 = GlobMatrixStream(expr, enumerate_by=EnumerateBy.NAME)

    assert stream1.sample_ids == stream2.sample_ids
    for sid in stream1.sample_ids:
        np.testing.assert_array_equal(
            stream1.load_dense_sample(sid).matrix,
            stream2.load_dense_sample(sid).matrix,
        )


def test_build_dataset_honors_enumerate_by_for_globbed_matrix_files(
    subdomain_named_txt_matrices: tuple[Path, int],
    tmp_path: Path,
) -> None:
    mat_dir, n = subdomain_named_txt_matrices
    out_dir = tmp_path / "enumerated_dataset"

    build_dataset(
        matrix_path=str(mat_dir / "*_subdomain_1_Kaa.txt"),
        dataset_dir=str(out_dir),
        counts={"neutral_ones": 3},
        enumerate_by=EnumerateBy.NAME,
        normalize="none",
        shuffle=False,
        seed=42,
    )

    rhs, solutions = load_dense_training_arrays(out_dir)
    assert rhs.shape == (3, n)
    assert solutions.shape == (3, n)

    mat_arr = zarr.open_array(str(resolve_dataset_paths(out_dir).matrix_zarr_dir), mode="r")
    assert mat_arr.shape[0] == 3


def test_source_config_rejects_both_sample_id_regex_and_enumerate_by() -> None:
    from neuralls.platform.config.models.data_models import SourceConfig

    with pytest.raises(Exception):
        SourceConfig(sample_id_regex=r"(\d+)", enumerate_by=EnumerateBy.NAME)


def test_source_config_accepts_enumerate_by_without_regex() -> None:
    from neuralls.platform.config.models.data_models import SourceConfig

    config = SourceConfig(enumerate_by=EnumerateBy.NAME)
    assert config.enumerate_by == EnumerateBy.NAME


def test_bind_sources_pairs_solution_ids_per_matrix() -> None:
    """solution_ids are bound per-matrix, mirroring the rhs_ids pattern."""
    bindings = bind_sources(
        matrix_ids=(0, 1, 2),
        solution_ids=(0, 1, 2),
    )
    assert [b.matrix_sample_id for b in bindings] == [0, 1, 2]
    assert [b.solution_sample_id for b in bindings] == [0, 1, 2]


def test_bind_sources_solution_mismatch_raises() -> None:
    """Mismatched solution_ids vs matrix_ids raises ValueError for multi-matrix case."""
    with pytest.raises(ValueError, match="solution IDs must match matrix IDs"):
        bind_sources(
            matrix_ids=(0, 1, 2),
            solution_ids=(0, 1),  # missing id 2
        )


def test_bind_sources_single_matrix_broadcasts_to_many_solution_ids() -> None:
    """Single matrix broadcasts across solution_ids (mirrors rhs_ids broadcast)."""
    bindings = bind_sources(
        matrix_ids=(0,),
        solution_ids=(0, 1, 2),
    )
    assert len(bindings) == 3
    assert all(b.matrix_sample_id == 0 for b in bindings)
    assert [b.solution_sample_id for b in bindings] == [0, 1, 2]
    assert all(b.rhs_sample_id is None for b in bindings)
