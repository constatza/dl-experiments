"""Sample builders for graph neural network models.

Graph models require both matrix and RHS as inputs, so these builders
wrap the standard sample builders and ensure proper data structure for
graph-based training and inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from .generation.types import RawSamples
from .normalization import IScale
from .sample_builders import (
    build_rhs_archive_samples,
    build_generated_samples,
    build_solution_archive_samples,
)


def build_graph_rhs_archive_samples(
    matrix_path: Path,
    rhs_glob: Sequence[Path],
    *,
    solve_systems: bool,
    cg_tolerance: float,
    cg_max_iters: int,
    normalize_type: Literal["none", "matrix", "diagonal", "spectral"] = "none",
    spectral_radius_bound: float | None = None,
) -> tuple[RawSamples, IScale | list[IScale] | None]:
    """Build graph samples from RHS archives and solved NORMALIZED systems.

    This is a wrapper around build_rhs_archive_samples that ensures the
    matrix is properly included for graph neural network training.

    Args:
        matrix_path: Path to system matrix file
        rhs_glob: Paths to RHS vector files
        solve_systems: Whether to solve systems for solutions
        cg_tolerance: CG solver tolerance
        cg_max_iters: CG solver max iterations
        normalize_type: Type of normalization ("none", "matrix", "diagonal", "spectral")
        spectral_radius_bound: For matrix normalization (computed if None)

    Returns:
        Tuple of (normalized_samples, scale):
            - normalized_samples: RawSamples with normalized matrix/RHS/solutions
            - scale: IScale object, list[IScale] (spectral), or None (no normalization)

    Note:
        For graph models, the matrix will be saved alongside the RHS samples
        during persistence, unlike standard models where matrix is saved once.
        Solutions are in normalized space: x_norm solves A_norm @ x_norm = b_norm.
    """
    # Reuse existing builder - graph vs non-graph difference is only in persistence
    return build_rhs_archive_samples(
        matrix_path=matrix_path,
        rhs_glob=rhs_glob,
        solve_systems=solve_systems,
        cg_tolerance=cg_tolerance,
        cg_max_iters=cg_max_iters,
        normalize_type=normalize_type,
        spectral_radius_bound=spectral_radius_bound,
    )


def build_graph_generated_samples(
    matrix_path: Path,
    rhs_path: Path,
    *,
    num_samples: int,
    mix: dict[str, float],
    krylov_iters: int,
    residual_iters: int,
    seed: int,
    shuffle: bool,
    normalize_type: Literal["none", "matrix", "rhs", "spectral", "diagonal"] = "none",
    spectral_radius_bound: float | None = None,
) -> tuple[RawSamples, IScale | list[IScale] | None]:
    """Build graph samples using synthetic generation strategies.

    This is a wrapper around build_generated_samples that ensures the
    matrix is properly included for graph neural network training.

    Args:
        matrix_path: Path to system matrix file
        rhs_path: Path to reference RHS file
        num_samples: Number of samples to generate
        mix: Dictionary of generation strategies and their proportions
        krylov_iters: Number of Krylov iterations for Krylov-based generation
        residual_iters: Number of CG iterations for residual trace strategies
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle samples
        normalize_type: Type of normalization ("none", "matrix", "diagonal", "spectral", "rhs")
        spectral_radius_bound: For matrix normalization (computed if None)

    Returns:
        Tuple of (normalized_samples, scale):
            - normalized_samples: RawSamples with normalized matrix/RHS/solutions
            - scale: IScale object, list[IScale] (spectral), or None (no normalization)

    Note:
        For graph models, the matrix will be saved alongside the RHS samples
        during persistence, unlike standard models where matrix is saved once.
    """
    # Reuse existing builder - graph vs non-graph difference is only in persistence
    return build_generated_samples(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        num_samples=num_samples,
        mix=mix,
        krylov_iters=krylov_iters,
        residual_iters=residual_iters,
        seed=seed,
        shuffle=shuffle,
        normalize_type=normalize_type,
        spectral_radius_bound=spectral_radius_bound,
    )


def build_graph_solution_archive_samples(
    matrix_path: Path,
    solution_files: Sequence[Path],
    *,
    shuffle: bool,
    seed: int | None,
    normalize_type: Literal["none", "matrix", "diagonal", "spectral"] = "none",
    spectral_radius_bound: float | None = None,
) -> tuple[RawSamples, IScale | list[IScale] | None]:
    """Build graph samples from an archive of pre-computed solutions.

    This is a wrapper around build_solution_archive_samples that ensures the
    matrix is properly included for graph neural network training.

    Args:
        matrix_path: Path to system matrix file
        solution_files: Paths to solution vector files
        shuffle: Whether to shuffle samples
        seed: Random seed for shuffling
        normalize_type: Type of normalization ("none", "matrix", "diagonal", "spectral")
        spectral_radius_bound: For matrix normalization (computed if None)

    Returns:
        Tuple of (normalized_samples, scale):
            - normalized_samples: RawSamples with normalized matrix/RHS/solutions
            - scale: IScale object, list[IScale] (spectral), or None (no normalization)

    Note:
        For graph models, the matrix will be saved alongside the RHS samples
        during persistence, unlike standard models where matrix is saved once.
    """
    # Reuse existing builder - graph vs non-graph difference is only in persistence
    return build_solution_archive_samples(
        matrix_path=matrix_path,
        solution_files=solution_files,
        shuffle=shuffle,
        seed=seed,
        normalize_type=normalize_type,
        spectral_radius_bound=spectral_radius_bound,
    )


def verify_graph_samples(samples: RawSamples) -> None:
    """Verify that raw samples are valid for graph neural network training.

    Args:
        samples: RawSamples to verify

    Raises:
        ValueError: If samples are invalid for graph models

    Note:
        Graph models require:
        - Matrix is square and 2D
        - RHS samples match matrix dimension
        - Solutions match matrix dimension (if present)
    """
    matrix = samples.matrix
    rhs = samples.rhs
    solutions = samples.solutions

    # Verify matrix
    if matrix.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got shape {matrix.shape}")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {matrix.shape}")

    n = matrix.shape[0]

    # Verify RHS
    if rhs.ndim != 2:
        raise ValueError(f"RHS must be 2D array of samples, got shape {rhs.shape}")
    if rhs.shape[1] != n:
        raise ValueError(
            f"RHS dimension {rhs.shape[1]} doesn't match matrix dimension {n}"
        )

    # Verify solutions (if present)
    if solutions.size > 0:
        if solutions.ndim != 2:
            raise ValueError(f"Solutions must be 2D array, got shape {solutions.shape}")
        if solutions.shape[1] != n:
            raise ValueError(
                f"Solutions dimension {solutions.shape[1]} doesn't match matrix dimension {n}"
            )
        if solutions.shape[0] != rhs.shape[0]:
            raise ValueError(
                f"Number of solutions {solutions.shape[0]} doesn't match number of RHS {rhs.shape[0]}"
            )


__all__ = [
    "build_graph_rhs_archive_samples",
    "build_graph_generated_samples",
    "build_graph_solution_archive_samples",
    "verify_graph_samples",
]
