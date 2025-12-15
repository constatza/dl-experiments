"""Pure utility functions for data generation strategies."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from scipy.linalg import eigh, norm

from ..constants import (
    EIGENVECTOR_SELECT_SMALLEST,
    EIGENVECTOR_SELECT_LARGEST,
    EIGENVECTOR_SELECT_RANDOM,
)
from .interfaces import ArchiveData


def rng_from_seed(seed: int | None) -> np.random.Generator:
    """Create random number generator from seed.

    Args:
        seed: Random seed (None for random)

    Returns:
        Random number generator
    """
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()


def rounded_counts(total: int, proportions: Mapping[str, float]) -> dict[str, int]:
    """Convert strategy proportions to integer counts that sum exactly to total.

    The proportions do not need to sum to 1.0; they are normalized internally.

    Args:
        total: Total number of samples to generate
        proportions: Mapping of strategy names to proportions (non-negative)

    Returns:
        Dictionary mapping strategy names to integer counts

    Raises:
        ValueError: If total <= 0, proportions empty, or weights negative/zero sum
    """
    if total <= 0:
        raise ValueError(f"Total samples must be positive, got {total}")
    if not proportions:
        raise ValueError("At least one strategy proportion must be provided")

    weights: dict[str, float] = {}
    for key, value in proportions.items():
        weight = float(value)
        if weight < 0:
            raise ValueError(
                f"Mix weight for '{key}' must be non-negative, got {value}"
            )
        weights[key] = weight

    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Sum of mix weights must be positive")

    scaled = {key: (weight / total_weight) * total for key, weight in weights.items()}
    counts = {key: int(math.floor(amount)) for key, amount in scaled.items()}
    remainders = {key: scaled[key] - counts[key] for key in scaled}

    remaining = total - sum(counts.values())
    if remaining > 0:
        sorted_keys = sorted(
            remainders.keys(),
            key=lambda key: (-remainders[key], -weights[key], key),
        )
        idx = 0
        while remaining > 0 and sorted_keys:
            key = sorted_keys[idx % len(sorted_keys)]
            counts[key] += 1
            remaining -= 1
            idx += 1

    return counts


def _calculate_normalization_scale(
    A: np.ndarray,
    b: np.ndarray,
    normalize: str,
) -> float:
    """Calculate target RHS scale based on normalization strategy.

    Pure function.

    Args:
        A: System matrix
        b: Mother RHS vector
        normalize: Normalization strategy

    Returns:
        Target RHS scale for generation
    """
    if normalize == "rhs":
        n = A.shape[0]
        target_norm = float(norm(b))
        return target_norm / np.sqrt(n)
    return 1.0


def _normalize_matrix_for_generation(
    matrix: np.ndarray,
    normalize_type: Literal["none", "matrix", "diagonal", "spectral", "rhs"],
    spectral_radius_bound: float | None,
) -> tuple[np.ndarray, Any]:
    """Normalize matrix for synthetic generation (pure function).

    This function prepares a matrix for synthetic data generation by applying
    the appropriate normalization strategy. For most strategies (matrix, diagonal),
    it returns a complete scale object. For spectral normalization, it normalizes
    the matrix but defers per-sample RHS norm computations until after generation.

    CONTRACT:
        - Input: Raw matrix A
        - Output: Normalized matrix A_norm and optional scale metadata
        - For spectral: Only matrix is normalized; RHS norms computed later
        - Strategies receive normalized matrix and compute b_norm = A_norm @ x

    Args:
        matrix: Raw system matrix A
        normalize_type: Normalization strategy
            - "none": No normalization (identity)
            - "matrix": Scale by spectral_radius_bound * sqrt(d)
            - "diagonal": Jacobi-style normalization (diagonal scaling)
            - "spectral": Scale matrix by spectral norm (RHS norms deferred)
            - "rhs": Legacy, treated as "none" (RHS matching handled by caller)
        spectral_radius_bound: For matrix normalization (computed if None)

    Returns:
        Tuple of (normalized_matrix, scale_or_none):
            - normalized_matrix: Matrix in normalized space
            - scale_or_none: IScale object for matrix/diagonal, None for spectral/none/rhs

    Notes:
        - For "none" or "rhs": Returns defensive copy and None
        - For "matrix": Returns normalized matrix and MatrixScale object
        - For "diagonal": Returns normalized matrix and DiagonalScale object
        - For "spectral": Returns normalized matrix ONLY (scale created later)
        - Spectral strategy: Matrix normalized by spectral_norm * sqrt(d),
          but per-sample scales (with individual rhs_norm) created after generation

    Examples:
        >>> # Matrix normalization
        >>> A_norm, scale = _normalize_matrix_for_generation(A, "matrix", None)
        >>> isinstance(scale, MatrixScale)
        True

        >>> # Spectral normalization (deferred RHS norms)
        >>> A_norm, scale = _normalize_matrix_for_generation(A, "spectral", None)
        >>> scale is None  # RHS norms not yet available
        True
    """
    from ..normalization import create_scale_from_config

    # No normalization: return defensive copy
    if normalize_type in ("none", "rhs"):
        return matrix.copy(), None

    # Spectral: Normalize matrix only (defer per-sample RHS norms)
    if normalize_type == "spectral":
        # Use matrix normalization to get A_norm, but don't return scale yet
        # (we need RHS samples to compute per-sample rhs_norm values)
        dimension = matrix.shape[0]
        from ..math_utils import calculate_spectral_norm, compute_dim_scale

        spectral_norm = calculate_spectral_norm(matrix)
        dimension_scale = compute_dim_scale(dimension)
        composite_scale = spectral_norm * dimension_scale
        matrix_norm = matrix / composite_scale
        return matrix_norm, None

    # Matrix or Diagonal: Create full scale and normalize matrix
    scale = create_scale_from_config(
        normalize_type=normalize_type,
        matrix=matrix,
        spectral_radius_bound=spectral_radius_bound,
    )
    assert scale is not None, f"Expected scale for {normalize_type}"
    assert not isinstance(scale, list), (
        f"Expected single scale for {normalize_type}, got list"
    )
    matrix_norm = scale.scale_matrix(matrix)
    return matrix_norm, scale


def _resolve_strategy_counts(
    counts: Mapping[str, int] | None,
    mix: Mapping[str, float] | None,
    total: int | None,
) -> dict[str, int]:
    """Resolve strategy counts from explicit counts or mix/total pair.

    Pure function: no side effects.

    Args:
        counts: Optional explicit strategy counts
        mix: Optional strategy proportions
        total: Total samples (required if mix provided)

    Returns:
        Dictionary of strategy_name -> count (only positive counts)

    Raises:
        ValueError: If arguments are invalid or inconsistent
    """
    if counts is not None and mix is not None:
        raise ValueError("Specify either explicit counts or a mix/total pair, not both")

    if counts is None:
        if mix is None:
            raise ValueError("Either counts or mix must be provided")
        if total is None:
            raise ValueError("Parameter 'total' is required when using mix")
        resolved = rounded_counts(int(total), mix)
    else:
        resolved = {name: int(value) for name, value in counts.items()}

    # Filter out zero counts (but keep -1 which means "all available")
    nonzero = {name: value for name, value in resolved.items() if value != 0}
    if not nonzero:
        raise ValueError("No strategy counts were provided")

    return nonzero


def _load_or_generate_solutions(
    count: int,
    n: int,
    rng: np.random.Generator,
    target_rhs_scale: float,
    archive: ArchiveData | None,
) -> np.ndarray:
    """Load solutions from archive or generate random solutions.

    Pure function (modulo RNG): deterministic given RNG state.

    Args:
        count: Number of solutions needed
        n: Dimension of solution space
        rng: Random number generator
        target_rhs_scale: Scale for random solutions
        archive: Optional archive data

    Returns:
        Solution vectors, shape (count, n)

    Raises:
        ValueError: If archive has insufficient solutions
    """
    if archive is None:
        return rng.normal(size=(count, n), scale=target_rhs_scale).astype(
            np.float64, copy=False
        )

    if archive.solutions.shape[0] < count:
        raise ValueError(
            f"Not enough archive solutions: need {count}, "
            f"got {archive.solutions.shape[0]}"
        )

    return archive.solutions[:count].astype(np.float64, copy=True)


def _load_or_compute_rhs(
    A: np.ndarray,
    solutions: np.ndarray,
    archive: ArchiveData | None,
) -> np.ndarray:
    """Load RHS from archive or compute from solutions.

    Pure function (matrix multiplication).

    Args:
        A: System matrix
        solutions: Solution vectors
        archive: Optional archive data

    Returns:
        RHS vectors, shape (N, n)

    Raises:
        ValueError: If archive has insufficient RHS vectors
    """
    if archive is None or archive.rhs_vectors is None:
        return np.array([A @ x for x in solutions], dtype=np.float64)

    count = solutions.shape[0]
    if archive.rhs_vectors.shape[0] < count:
        raise ValueError(
            f"Not enough archive RHS vectors: need {count}, "
            f"got {archive.rhs_vectors.shape[0]}"
        )

    return archive.rhs_vectors[:count].astype(np.float64, copy=True)


def _build_trace_indices(
    num_pairs: int,
    sample_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sample and iteration index arrays for trace data.

    Pure function.

    Args:
        num_pairs: Number of (residual, solution) pairs
        sample_idx: Sample index for this trace

    Returns:
        Tuple of (sample_indices, iteration_indices)
    """
    return (
        np.full(num_pairs, sample_idx, dtype=np.int64),
        np.arange(num_pairs, dtype=np.int64),
    )


def _merge_strategy_outputs(
    features_list: list[np.ndarray],
    targets_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Merge feature and target arrays from multiple strategies.

    Pure function.

    Args:
        features_list: List of feature arrays
        targets_list: List of target arrays

    Returns:
        Tuple of (merged_features, merged_targets)
    """
    return np.vstack(features_list), np.vstack(targets_list)


# =============================================================================
# EIGENVECTOR STRATEGY HELPERS
# =============================================================================


def _compute_eigendecomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigendecomposition of symmetric matrix using scipy.

    Args:
        A: Symmetric matrix

    Returns:
        Tuple of (eigenvalues, eigenvectors)

    Raises:
        ValueError: If matrix is not symmetric within tolerance
    """
    if not np.allclose(A, A.T, rtol=1e-10, atol=1e-10):
        max_asymmetry = np.max(np.abs(A - A.T))
        raise ValueError(
            f"Eigenvector strategies require symmetric matrices. "
            f"Max asymmetry: {max_asymmetry:.2e}"
        )
    eigenvalues, eigenvectors = eigh(A)
    return eigenvalues, eigenvectors


def _select_eigenvectors(
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    count: int,
    which: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select subset of eigenvectors according to which eigenvalues to use.

    Args:
        eigenvectors: Eigenvector matrix, shape (n, n)
        eigenvalues: Eigenvalue array, shape (n,)
        count: Number of eigenvectors to select
        which: Which eigenvalues to select ("smallest", "largest", or "random")
            - "smallest": Select k eigenvectors with smallest eigenvalues
            - "largest": Select k eigenvectors with largest eigenvalues
            - "random": Random selection without replacement
        rng: Random number generator (used for "random" mode)

    Returns:
        Tuple of (selected_eigenvectors, selected_eigenvalues, indices)

    Raises:
        ValueError: If count invalid or which unknown
    """
    n = eigenvectors.shape[0]
    if count > n:
        raise ValueError(
            f"Requested {count} samples but matrix has only {n} eigenvectors. "
            f"Maximum samples: {n}"
        )
    if count <= 0:
        raise ValueError(f"Sample count must be positive, got {count}")
    if which == EIGENVECTOR_SELECT_SMALLEST:
        indices = np.arange(count)
    elif which == EIGENVECTOR_SELECT_LARGEST:
        indices = np.arange(n - count, n)
    elif which == EIGENVECTOR_SELECT_RANDOM:
        indices = rng.choice(n, size=count, replace=False)
    else:
        raise ValueError(
            f"Invalid which: '{which}'. Must be '{EIGENVECTOR_SELECT_SMALLEST}', "
            f"'{EIGENVECTOR_SELECT_LARGEST}', or '{EIGENVECTOR_SELECT_RANDOM}'"
        )
    return eigenvectors[:, indices], eigenvalues[indices], indices

def _generate_eigenvector_combinations(
    eigenvectors: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate random L2-normalized linear combinations of eigenvectors.

    Args:
        eigenvectors: Eigenvector matrix, shape (n, k)
        num_samples: Number of combinations to generate
        rng: Random number generator

    Returns:
        Linear combinations, shape (num_samples, n)
    """
    n, k = eigenvectors.shape
    coeffs = rng.standard_normal(size=(num_samples, k), dtype=np.float64)
    norms = np.linalg.norm(coeffs, axis=1, keepdims=True)
    coeffs_normalized = coeffs / norms
    return coeffs_normalized @ eigenvectors.T


def _verify_solution_accuracy(
    A: np.ndarray,
    rhs_vectors: np.ndarray,
    solutions: np.ndarray,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Verify solution accuracy by computing relative residuals.

    Args:
        A: System matrix
        rhs_vectors: Right-hand side vectors
        solutions: Solution vectors
        tolerance: Relative residual tolerance for warnings

    Returns:
        Array of relative residuals, shape (count,)

    Warns:
        RuntimeWarning: If any solution exceeds tolerance
    """
    count = rhs_vectors.shape[0]
    rel_residuals = np.zeros(count)
    for i in range(count):
        residual = A @ solutions[i] - rhs_vectors[i]
        rhs_norm = np.linalg.norm(rhs_vectors[i])
        rel_residuals[i] = (
            np.linalg.norm(residual) / rhs_norm
            if rhs_norm > 0
            else np.linalg.norm(residual)
        )
        if rel_residuals[i] > tolerance:
            warnings.warn(
                f"Sample {i}: Solution accuracy {rel_residuals[i]:.2e} "
                f"exceeds tolerance {tolerance:.2e}",
                RuntimeWarning,
                stacklevel=2,
            )
    return rel_residuals


def select_archive_files(
    glob_pattern: str,
    count: int,
    shuffle: bool,
    seed: int | None,
) -> list:
    """Select N files from glob pattern with optional shuffling.

    Pure function for deterministic file selection from archives.

    Args:
        glob_pattern: Path pattern like "/data/*.txt" or "/data/rhs_*.npy"
        count: Number of files to select (-1 means all available files)
        shuffle: Whether to shuffle before selection
        seed: Random seed for shuffling (required if shuffle=True)

    Returns:
        List of selected file paths (pathlib.Path objects)

    Raises:
        FileNotFoundError: If no files match pattern or directory doesn't exist
        ValueError: If count exceeds available files

    Examples:
        >>> # Select first 10 files
        >>> files = select_archive_files("/data/rhs_*.txt", 10, False, None)

        >>> # Select all files with shuffling
        >>> files = select_archive_files("/data/sol_*.npy", -1, True, 42)

        >>> # Select 50 shuffled files
        >>> files = select_archive_files("/data/vec_*.txt", 50, True, 42)
    """
    from pathlib import Path

    pattern_path = Path(glob_pattern)
    directory = pattern_path.parent
    pattern = pattern_path.name

    # Validate directory exists
    if not directory.exists():
        raise FileNotFoundError(f"Archive directory not found: {directory}")

    # Scan for matching files
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No files found matching pattern: {directory / pattern}"
        )

    # Handle "all files" case
    if count == -1:
        count = len(candidates)

    # Validate sufficient files available
    if count > len(candidates):
        raise ValueError(
            f"Requested {count} files but only {len(candidates)} available "
            f"matching pattern: {glob_pattern}"
        )

    # Select files with optional shuffling
    if shuffle:
        rng = rng_from_seed(seed)
        indices = rng.permutation(len(candidates))[:count]
        return [candidates[idx] for idx in indices]

    return candidates[:count]


__all__ = [
    "rng_from_seed",
    "rounded_counts",
    "select_archive_files",
    "_calculate_normalization_scale",
    "_normalize_matrix_for_generation",
    "_resolve_strategy_counts",
    "_load_or_generate_solutions",
    "_load_or_compute_rhs",
    "_build_trace_indices",
    "_merge_strategy_outputs",
    "_compute_eigendecomposition",
    "_select_eigenvectors",
    "_generate_eigenvector_combinations",
    "_verify_solution_accuracy",
]
