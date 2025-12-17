"""Solution Archive Strategy - Load solution vectors from disk and compute RHS."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..interfaces import GeneratedSamples, IMatrixOnlyGenerationStrategy, ArchiveData
from ..helpers import select_archive_files
from ..runner import register_strategy
from ..strategy_configs import SolutionArchiveConfig


def _load_solution_vectors(
    solution_files: list[Path],
    expected_dimension: int,
) -> np.ndarray:
    """Load solution vectors from text files and validate dimensions.

    Pure function (modulo I/O).

    Args:
        solution_files: List of solution file paths
        expected_dimension: Expected vector dimension

    Returns:
        Stacked solution vectors, shape (num_files, n)

    Raises:
        ValueError: If no files or dimension mismatch
    """
    if not solution_files:
        raise ValueError("No solution files provided")

    solutions: list[np.ndarray] = []

    for path in solution_files:
        solution = np.loadtxt(path, dtype=np.float64)
        if solution.ndim > 1:
            solution = solution.reshape(-1)
        if solution.size != expected_dimension:
            raise ValueError(
                f"Solution {path} has dimension {solution.size}, "
                f"expected {expected_dimension}"
            )
        solutions.append(solution)

    return np.stack(solutions, axis=0)


def _compute_rhs_from_solutions(
    matrix: np.ndarray,
    solutions: np.ndarray,
) -> np.ndarray:
    """Compute RHS vectors from solutions: b = A @ x.

    Pure function (matrix multiplication).

    Args:
        matrix: System matrix, shape (n, n)
        solutions: Solution vectors, shape (num_solutions, n)

    Returns:
        RHS vectors, shape (num_solutions, n)
    """
    return np.array([matrix @ x for x in solutions], dtype=np.float64)


@register_strategy
class SolutionArchiveStrategy(IMatrixOnlyGenerationStrategy):
    """Load solution vectors from archive and compute RHS = A @ x.

    This strategy treats archive collection as data generation by loading
    pre-computed solution vectors from disk and computing the corresponding
    RHS vectors through matrix multiplication.

    Configuration:
        - solutions_glob (str): Glob pattern for solution files (e.g., "/data/sol_*.txt")
        - samples (int): Number of files to load (-1 for all)
        - shuffle (bool): Whether to shuffle file selection (default: False)
        - seed (int | None): Random seed for shuffling (required if shuffle=True)

    Examples:
        >>> # Load first 50 solution files
        >>> cfg = {
        ...     "solutions_glob": "/data/solutions_*.txt",
        ...     "samples": 50,
        ... }
        >>> samples = strategy.generate(matrix, None, cfg=cfg)

        >>> # Load all solution files with shuffling
        >>> cfg = {
        ...     "solutions_glob": "/data/sol_*.npy",
        ...     "samples": -1,
        ...     "shuffle": True,
        ...     "seed": 42,
        ... }
        >>> samples = strategy.generate(matrix, None, cfg=cfg)
    """

    name = "solution_archive"
    ConfigType = SolutionArchiveConfig

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
        """Load solution vectors from archive and compute RHS = A @ x.

        Args:
            matrix: System matrix (already normalized by orchestrator)
            rhs: Mother RHS (ignored, strategy is self-sufficient)
            cfg: Configuration dictionary with keys:
                - solutions_glob: Glob pattern for solution files
                - samples: Number of files to load
                - shuffle: Whether to shuffle file selection
                - seed: Random seed for shuffling
            archive: Optional pre-loaded archive data (ignored by this strategy as it loads from disk)

        Returns:
            GeneratedSamples with loaded solutions and computed RHS

        Raises:
            ValueError: If solutions_glob not provided or insufficient files
            FileNotFoundError: If no files match pattern
        """
        # Validate and convert to typed config
        config = SolutionArchiveConfig(**cfg)

        # Extract configuration
        solutions_glob = config.solutions_glob
        samples = config.samples
        shuffle = config.shuffle
        seed = config.seed

        # Select files from archive
        solution_files = select_archive_files(
            glob_pattern=solutions_glob,
            count=samples,
            shuffle=shuffle,
            seed=seed,
        )

        print(f"Loading {len(solution_files)} solution vectors from archive...")

        # Load solution vectors
        expected_dimension = matrix.shape[0]
        solutions = _load_solution_vectors(solution_files, expected_dimension)

        # Compute RHS = A @ x
        print(f"Computing RHS for {len(solution_files)} solutions...")
        rhs_vectors = _compute_rhs_from_solutions(matrix, solutions)

        return GeneratedSamples(
            matrix=matrix,
            rhs=rhs_vectors,
            solutions=solutions,
            residual_traces=None,
            error_traces=None,
        )
