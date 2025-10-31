"""Sample builders for different data acquisition strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse.linalg import cg as scipy_cg

from .common import load_system_data
from .data_generation import generate_mixture, rng_from_seed
from .data_pipeline import RawSamples, load_matrix, rhs_from_solutions
from .validation import validate_matrix, validate_rhs


def _load_rhs_samples(rhs_files: Sequence[Path], matrix: np.ndarray) -> np.ndarray:
    """Load RHS samples from text files and validate against the matrix."""

    rhs_list: list[np.ndarray] = []
    for rhs_file in rhs_files:
        rhs = np.loadtxt(rhs_file, dtype=np.float64)
        if rhs.ndim > 1:
            rhs = rhs.reshape(-1)
        validate_rhs(rhs, matrix)
        rhs_list.append(rhs)

    if not rhs_list:
        raise ValueError("No RHS samples provided")

    return np.stack(rhs_list, axis=0)


def build_rhs_archive_samples(
    matrix_path: Path,
    rhs_glob: Sequence[Path],
    *,
    solve_systems: bool,
    cg_tolerance: float,
    cg_max_iters: int,
) -> RawSamples:
    """Build raw samples by replaying RHS archives and solving systems."""

    matrix = load_matrix(matrix_path)
    validate_matrix(matrix)

    rhs_samples = _load_rhs_samples(rhs_glob, matrix)
    num_samples, dimension = rhs_samples.shape

    solutions = np.zeros((num_samples, dimension), dtype=np.float64)

    if solve_systems:
        absolute_tolerance = 0.0
        for idx, rhs in enumerate(rhs_samples):
            if (idx + 1) % 100 == 0 or idx == 0:
                print(f"  Solving system {idx + 1}/{num_samples}")

            solution, exit_code = scipy_cg(
                matrix,
                rhs,
                rtol=cg_tolerance,
                maxiter=cg_max_iters,
                atol=absolute_tolerance,
            )
            solutions[idx, :] = solution

            if exit_code != 0:
                residual = np.linalg.norm(matrix @ solution - rhs)
                print(
                    f"  Warning: system {idx + 1} exit_code={exit_code} "
                    f"(residual: {residual:.2e})"
                )

    return RawSamples(
        matrix=matrix,
        rhs=rhs_samples,
        solutions=solutions,
        mother_rhs=rhs_samples[0].copy(),
    )


def build_generated_samples(
    matrix_path: Path,
    rhs_path: Path,
    *,
    num_samples: int,
    mix: dict[str, float],
    krylov_iters: int,
    residual_iters: int,
    seed: int,
    shuffle: bool,
    normalize_mode: str,
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> RawSamples:
    """Build raw samples using synthetic generation strategies.

    Args:
        matrix_path: Path to system matrix file
        rhs_path: Path to reference RHS file
        num_samples: Number of samples to generate
        mix: Strategy proportions keyed by strategy name
        krylov_iters: Default Krylov iterations for "krylov" strategy
        residual_iters: Default CG iterations for residual-trace strategies
        seed: Random seed driving synthetic generation
        shuffle: Whether to shuffle combined samples
        normalize_mode: Normalization behaviour passed to generators
        strategy_overrides: Optional mapping of strategy name to override options
            (e.g., {"krylov": {"krylov_iters": 10}})
    """

    matrix, mother_rhs = load_system_data(matrix_path, rhs_path)
    validate_matrix(matrix)
    validate_rhs(mother_rhs, matrix)

    rhs_samples, solutions, residual_traces = generate_mixture(
        A=matrix,
        b=mother_rhs,
        mix=mix,
        total=num_samples,
        krylov_iters=krylov_iters,
        residual_iters=residual_iters,
        seed=seed,
        shuffle=shuffle,
        normalize=normalize_mode,
        strategy_overrides=strategy_overrides,
    )

    return RawSamples(
        matrix=matrix,
        rhs=rhs_samples,
        solutions=solutions,
        mother_rhs=mother_rhs.copy(),
        residual_traces=residual_traces,
    )


def _load_solution_samples(
    solution_files: Sequence[Path],
    expected_dimension: int,
) -> np.ndarray:
    """Load solution vectors from text files and validate dimensions."""

    solutions: list[np.ndarray] = []
    for path in solution_files:
        solution = np.loadtxt(path, dtype=np.float64)
        if solution.ndim > 1:
            solution = solution.reshape(-1)
        if solution.size != expected_dimension:
            raise ValueError(
                f"Solution {path} has dimension {solution.size}, expected {expected_dimension}"
            )
        solutions.append(solution)

    if not solutions:
        raise ValueError("No solution samples provided")

    return np.stack(solutions, axis=0)


def build_solution_archive_samples(
    matrix_path: Path,
    solution_files: Sequence[Path],
    *,
    shuffle: bool,
    seed: int | None,
) -> RawSamples:
    """Build raw samples from an archive of pre-computed solutions."""

    matrix = load_matrix(matrix_path)
    validate_matrix(matrix)

    dimension = matrix.shape[0]
    solutions = _load_solution_samples(solution_files, dimension)
    rhs_samples = rhs_from_solutions(matrix, solutions)

    if shuffle:
        rng = rng_from_seed(seed)
        indices = rng.permutation(len(rhs_samples))
        rhs_samples = rhs_samples[indices]
        solutions = solutions[indices]

    return RawSamples(
        matrix=matrix,
        rhs=rhs_samples,
        solutions=solutions,
        mother_rhs=rhs_samples[0].copy(),
    )


__all__ = [
    "build_rhs_archive_samples",
    "build_generated_samples",
    "build_solution_archive_samples",
]
