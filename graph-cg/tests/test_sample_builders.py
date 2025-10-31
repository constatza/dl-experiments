from __future__ import annotations

import numpy as np

from src.sample_builders import build_solution_archive_samples


def test_build_solution_archive_samples(tmp_path) -> None:
    matrix = np.array([[2.0, 0.0], [0.0, 3.0]])
    matrix_path = tmp_path / "matrix.txt"
    np.savetxt(matrix_path, matrix)

    solutions = [np.array([1.0, 2.0]), np.array([-1.0, 0.5])]
    solution_paths = []
    for idx, solution in enumerate(solutions):
        path = tmp_path / f"solution_{idx}.txt"
        np.savetxt(path, solution)
        solution_paths.append(path)

    samples = build_solution_archive_samples(
        matrix_path=matrix_path,
        solution_files=solution_paths,
        shuffle=False,
        seed=None,
    )

    assert samples.matrix.shape == (2, 2)
    assert np.allclose(samples.matrix, matrix)

    expected_rhs = np.array([matrix @ solution for solution in solutions])
    assert np.allclose(samples.rhs, expected_rhs)
    assert np.allclose(samples.solutions, np.stack(solutions))
    assert np.allclose(samples.mother_rhs, expected_rhs[0])


def test_build_solution_archive_samples_shuffle(tmp_path) -> None:
    matrix = np.eye(2)
    matrix_path = tmp_path / "matrix.txt"
    np.savetxt(matrix_path, matrix)

    solutions = [np.array([i, i + 1], dtype=float) for i in range(4)]
    solution_paths = []
    for idx, solution in enumerate(solutions):
        path = tmp_path / f"sol_{idx}.txt"
        np.savetxt(path, solution)
        solution_paths.append(path)

    samples_first = build_solution_archive_samples(
        matrix_path=matrix_path,
        solution_files=solution_paths,
        shuffle=True,
        seed=42,
    )

    samples_second = build_solution_archive_samples(
        matrix_path=matrix_path,
        solution_files=solution_paths,
        shuffle=True,
        seed=42,
    )

    samples_third = build_solution_archive_samples(
        matrix_path=matrix_path,
        solution_files=solution_paths,
        shuffle=True,
        seed=7,
    )

    original_stack = np.stack(solutions)

    # Fixed seed should yield deterministic ordering
    assert np.allclose(samples_first.solutions, samples_second.solutions)
    assert not np.allclose(samples_first.solutions, original_stack)
    assert not np.allclose(samples_first.solutions, samples_third.solutions)

    # Matrix is identity -> rhs matches solutions and mother rhs is first entry
    assert np.allclose(samples_first.rhs, samples_first.solutions)
    assert np.allclose(samples_first.mother_rhs, samples_first.solutions[0])
