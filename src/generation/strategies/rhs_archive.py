"""RHS Archive Strategy - Load RHS vectors from disk and optionally solve systems."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse.linalg import cg as scipy_cg

from ..interfaces import GeneratedSamples, IMatrixOnlyGenerationStrategy, ArchiveData
from ...validation import validate_rhs_vector as validate_rhs
from ..helpers import select_archive_files
from ..runner import register_strategy
from ..strategy_configs import RhsArchiveConfig


def _load_rhs_vectors(
    rhs_files: list[Path],
    matrix: np.ndarray,
) -> np.ndarray:
    """Load RHS vectors from text files and validate against matrix.

    Pure function (modulo I/O).

    Args:
        rhs_files: List of RHS file paths
        matrix: System matrix for dimension validation

    Returns:
        Stacked RHS vectors, shape (num_files, n)

    Raises:
        ValueError: If no files or dimension mismatch
    """
    if not rhs_files:
        raise ValueError("No RHS files provided")

    rhs_list: list[np.ndarray] = []
    matrix.shape[0]

    for rhs_file in rhs_files:
        rhs = np.loadtxt(rhs_file, dtype=np.float64)
        if rhs.ndim > 1:
            rhs = rhs.reshape(-1)
        validate_rhs(rhs, matrix)
        rhs_list.append(rhs)

    return np.stack(rhs_list, axis=0)


def _solve_systems(
    matrix: np.ndarray,
    rhs_vectors: np.ndarray,
    cg_tolerance: float,
    cg_max_iters: int,
) -> np.ndarray:
    """Solve linear systems Ax = b using Conjugate Gradient.

    Args:
        matrix: System matrix, shape (n, n)
        rhs_vectors: RHS vectors, shape (num_systems, n)
        cg_tolerance: Relative tolerance for CG
        cg_max_iters: Maximum CG iterations

    Returns:
        Solution vectors, shape (num_systems, n)
    """
    num_systems, n = rhs_vectors.shape
    solutions = np.zeros((num_systems, n), dtype=np.float64)
    absolute_tolerance = 0.0

    for idx, rhs in enumerate(rhs_vectors):
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
                f"Warning: RHS archive system {idx + 1} exit_code={exit_code} "
                f"(residual: {residual:.2e})"
            )

    return solutions


@register_strategy
class RhsArchiveStrategy(IMatrixOnlyGenerationStrategy):
    """Load RHS vectors from archive and optionally solve systems.

    This strategy treats archive collection as data generation by loading
    pre-computed RHS vectors from disk. It can optionally solve the linear
    systems to obtain solution vectors.

    Configuration:
        - rhs_glob (str): Glob pattern for RHS files (e.g., "/data/rhs_*.txt")
        - samples (int): Number of files to load (-1 for all)
        - shuffle (bool): Whether to shuffle file selection (default: False)
        - seed (int | None): Random seed for shuffling (required if shuffle=True)
        - solve_systems (bool): Whether to solve systems (default: True)
        - cg_tolerance (float): CG relative tolerance (default: 1e-12)
        - cg_max_iters (int): CG max iterations (default: 500)

    Examples:
        >>> # Load first 100 RHS files and solve
        >>> cfg = {
        ...     "rhs_glob": "/data/rhs_*.txt",
        ...     "samples": 100,
        ...     "solve_systems": True,
        ...     "cg_tolerance": 1e-12,
        ... }
        >>> samples = strategy.generate(matrix, None, cfg=cfg)

        >>> # Load all RHS files with shuffling
        >>> cfg = {
        ...     "rhs_glob": "/data/rhs_*.txt",
        ...     "samples": -1,
        ...     "shuffle": True,
        ...     "seed": 42,
        ...     "solve_systems": False,
        ... }
        >>> samples = strategy.generate(matrix, None, cfg=cfg)
    """

    name = "rhs_archive"
    ConfigType = RhsArchiveConfig

    def requires_rhs(self) -> bool:
        """Archive is self-sufficient, doesn't need mother RHS."""
        return False

    def generate(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray | None,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Load RHS vectors from archive and optionally solve systems.

        Args:
            matrix: System matrix (already normalized by orchestrator)
            rhs: Mother RHS (ignored, strategy is self-sufficient)
            cfg: Configuration dictionary with keys:
                - rhs_glob: Glob pattern for RHS files
                - samples: Number of files to load
                - shuffle: Whether to shuffle file selection
                - seed: Random seed for shuffling
                - solve_systems: Whether to solve systems
                - cg_tolerance: CG relative tolerance
                - cg_max_iters: CG max iterations
            archive: Optional pre-loaded archive data (ignored by this strategy as it loads from disk)

        Returns:
            GeneratedSamples with loaded RHS and optionally solved solutions

        Raises:
            ValueError: If rhs_glob not provided or insufficient files
            FileNotFoundError: If no files match pattern
        """
        # Validate and convert to typed config
        config = RhsArchiveConfig(**cfg)

        # Extract configuration
        rhs_glob = config.rhs_glob
        samples = config.samples
        shuffle = config.shuffle
        seed = config.seed
        solve_systems = config.solve_systems
        cg_tolerance = config.cg_tolerance
        cg_max_iters = config.cg_max_iters

        # Select files from archive
        rhs_files = select_archive_files(
            glob_pattern=rhs_glob,
            count=samples,
            shuffle=shuffle,
            seed=seed,
        )

        print(f"Loading {len(rhs_files)} RHS vectors from archive...")

        # Load RHS vectors
        rhs_vectors = _load_rhs_vectors(rhs_files, matrix)

        # Optionally solve systems
        if solve_systems:
            print(f"Solving {len(rhs_files)} linear systems...")
            solutions = _solve_systems(matrix, rhs_vectors, cg_tolerance, cg_max_iters)
        else:
            # Return zero solutions if not solving
            solutions = np.zeros_like(rhs_vectors)

        return GeneratedSamples(
            matrix=matrix,
            rhs=rhs_vectors,
            solutions=solutions,
            residual_traces=None,
            error_traces=None,
        )
