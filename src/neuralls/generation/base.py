"""Data generation strategies for training data creation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal
from collections.abc import Mapping

import warnings

import numpy as np
from scipy.linalg import eigh, norm, solve

from ..solver import SolverResult, flexible_cg
from ..normalization import ErrorTraceSamples, ResidualTraceSamples


# =============================================================================
# CONSTANTS
# =============================================================================

# Eigenvector strategy constants
EIGENVECTOR_SOLUTION_TOLERANCE = 1e-14  # Near machine precision for float64


# =============================================================================
# DATA LAYER: Immutable data structures
# =============================================================================


@dataclass(frozen=True)
class StrategyOutput:
    """Container for samples produced by a generation strategy."""

    rhs: np.ndarray
    solutions: np.ndarray
    residual_traces: ResidualTraceSamples | None = None
    error_traces: ErrorTraceSamples | None = None


@dataclass(frozen=True)
class ArchiveData:
    """Container for pre-computed solutions and RHS vectors.

    Used when loading from solution archives instead of generating random data.

    Attributes:
        solutions: Solution vectors, shape (N, n)
        rhs_vectors: Optional pre-computed RHS vectors, shape (N, n)
    """

    solutions: np.ndarray
    rhs_vectors: np.ndarray | None = None


# =============================================================================
# PURE HELPERS: Utilities with no side effects
# =============================================================================


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

    # Filter out zero counts
    nonzero = {name: value for name, value in resolved.items() if value > 0}
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
# STRATEGY LAYER: Data generation strategies
# =============================================================================


def normal_strategy(
    A: np.ndarray,
    b: np.ndarray,
    count: int,
    rng: np.random.Generator,
    match_rhs_norm: bool = False,
    target_rhs_scale: float = 1.0,
) -> StrategyOutput:
    """Generate samples using normal random solutions.

    CONTRACT:
        - Matrix A is already NORMALIZED by caller
        - Strategy generates solutions in normalized space
        - RHS computed as: b_norm = A_norm @ x
        - Returned samples are in NORMALIZED space

    Args:
        A: System matrix (NORMALIZED)
        b: RHS vector (mother RHS, used for norm scaling if match_rhs_norm=True)
        count: Number of samples to generate
        rng: Random number generator
        match_rhs_norm: If True, scale generated RHS to consistent target scale
        target_rhs_scale: Target RHS norm scale (default: 1.0)

    Returns:
        StrategyOutput with normalized RHS and solutions
    """
    n = A.shape[0]

    X = rng.normal(size=(count, n), scale=target_rhs_scale).astype(
        np.float64, copy=False
    )
    R = np.array([A @ x for x in X], dtype=np.float64)

    return StrategyOutput(rhs=R, solutions=X)


def krylov_strategy(
    A: np.ndarray,
    b: np.ndarray,
    count: int,
    krylov_iters: int,
    rng: np.random.Generator,
    match_rhs_norm: bool = False,
    target_rhs_scale: float = 1.0,
) -> StrategyOutput:
    """Generate samples using Lanczos-based Krylov subspace method."""
    n = A.shape[0]
    m = krylov_iters

    V = np.zeros((n, m), dtype=np.float64)
    alpha = np.zeros(m, dtype=np.float64)
    beta = np.zeros(m + 1, dtype=np.float64)

    v = rng.normal(size=n).astype(np.float64, copy=False)
    v = v / norm(v)
    V[:, 0] = v

    v_prev = np.zeros(n, dtype=np.float64)
    beta[0] = 0.0

    m_eff = m
    for j in range(m):
        w = A @ V[:, j] - beta[j] * v_prev
        alpha[j] = np.dot(V[:, j], w)
        w = w - alpha[j] * V[:, j]
        beta[j + 1] = norm(w)
        if beta[j + 1] <= 1e-14:
            m_eff = j + 1
            V = V[:, :m_eff]
            alpha = alpha[:m_eff]
            beta = beta[: m_eff + 1]
            break
        v_prev = V[:, j].copy()
        if j + 1 < m:
            V[:, j + 1] = w / beta[j + 1]

    T = (
        np.diag(alpha[:m_eff])
        + np.diag(beta[1:m_eff], k=-1)
        + np.diag(beta[1:m_eff], k=1)
    )

    Lambda, Q = np.linalg.eigh(T)
    Lambda_inv = 1.0 / Lambda

    R = []
    X = []

    for _ in range(count):
        eps = rng.normal(size=m_eff).astype(np.float64, copy=False)
        x = V @ (Q @ (Lambda_inv * eps))
        b_sample = A @ x
        R.append(b_sample)
        X.append(x)

    return StrategyOutput(
        rhs=np.array(R, dtype=np.float64),
        solutions=np.array(X, dtype=np.float64),
    )


def residual_trace_strategy(
    A: np.ndarray,
    b: np.ndarray,
    count: int,
    cg_iters: int,
    rng: np.random.Generator,
    match_rhs_norm: bool = False,
    target_rhs_scale: float = 1.0,
    solutions: np.ndarray | None = None,
    rhs_vectors: np.ndarray | None = None,
) -> StrategyOutput:
    """Generate samples augmented with CG residual traces for each system."""
    n = A.shape[0]
    archive = ArchiveData(solutions, rhs_vectors) if solutions is not None else None
    sols = _load_or_generate_solutions(count, n, rng, target_rhs_scale, archive)
    rhs_samples = _load_or_compute_rhs(A, sols, archive)

    residual_blocks: list[np.ndarray] = []
    solution_blocks: list[np.ndarray] = []
    search_direction_blocks: list[np.ndarray] = []
    search_direction_product_blocks: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    iteration_indices: list[np.ndarray] = []

    for sample_idx, rhs_vec in enumerate(rhs_samples):
        _, info_result = flexible_cg(
            A,
            rhs_vec,
            x0=np.zeros(n, dtype=np.float64),
            max_iter=cg_iters,
            preconditioner=None,
            stopping_criterion="fixed_iterations",
            tol=1e-12,
        )
        info = (
            SolverResult(**info_result)
            if isinstance(info_result, dict)
            else info_result
        )

        assert info.event_log is not None
        residual_list = info.event_log.get_history("residual")
        solution_list = info.event_log.get_history("solution")
        search_direction_list = info.event_log.get_history("search_direction")

        residual_seq = np.array(residual_list)
        solution_seq = np.array(solution_list)
        search_direction_seq = np.array(search_direction_list)
        search_direction_products_seq = np.array(
            [A @ search_direction for search_direction in search_direction_seq]
        )

        num_pairs = residual_seq.shape[0]

        residual_blocks.append(residual_seq)
        solution_blocks.append(solution_seq)
        search_direction_blocks.append(search_direction_seq)
        search_direction_product_blocks.append(search_direction_products_seq)
        sidx, iidx = _build_trace_indices(num_pairs, sample_idx)
        sample_indices.append(sidx)
        iteration_indices.append(iidx)

    residual_traces = ResidualTraceSamples(
        residuals=np.vstack(residual_blocks),
        solutions=np.vstack(solution_blocks),
        sample_indices=np.concatenate(sample_indices),
        iteration_indices=np.concatenate(iteration_indices),
        search_directions=np.vstack(search_direction_blocks),
        search_direction_products=np.vstack(search_direction_product_blocks),
    )

    return StrategyOutput(
        rhs=rhs_samples,
        solutions=sols,
        residual_traces=residual_traces,
    )


def residual_error_strategy(
    A: np.ndarray,
    b: np.ndarray,
    count: int,
    cg_iters: int,
    rng: np.random.Generator,
    match_rhs_norm: bool = False,
    target_rhs_scale: float = 1.0,
    solutions: np.ndarray | None = None,
    rhs_vectors: np.ndarray | None = None,
) -> StrategyOutput:
    """Generate samples augmented with CG error traces for each system."""
    n = A.shape[0]
    archive = ArchiveData(solutions, rhs_vectors) if solutions is not None else None
    sols = _load_or_generate_solutions(count, n, rng, target_rhs_scale, archive)
    rhs_samples = _load_or_compute_rhs(A, sols, archive)

    residual_blocks: list[np.ndarray] = []
    solution_current_blocks: list[np.ndarray] = []
    error_blocks: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    iteration_indices: list[np.ndarray] = []

    for sample_idx, (rhs_vec, true_sol) in enumerate(zip(rhs_samples, sols)):
        _, info_result = flexible_cg(
            A,
            rhs_vec,
            x0=np.zeros(n, dtype=np.float64),
            max_iter=cg_iters,
            preconditioner=None,
            stopping_criterion="fixed_iterations",
            tol=1e-12,
        )
        info = (
            SolverResult(**info_result)
            if isinstance(info_result, dict)
            else info_result
        )

        assert info.event_log is not None
        residual_list = info.event_log.get_history("residual")
        solution_list = info.event_log.get_history("solution")

        residual_seq = np.array(residual_list)
        solution_seq = np.array(solution_list)
        num_pairs = residual_seq.shape[0]

        error_seq = np.array([true_sol - x_k for x_k in solution_seq], dtype=np.float64)

        residual_blocks.append(residual_seq)
        solution_current_blocks.append(solution_seq)
        error_blocks.append(error_seq)
        sidx, iidx = _build_trace_indices(num_pairs, sample_idx)
        sample_indices.append(sidx)
        iteration_indices.append(iidx)

    error_traces = ErrorTraceSamples(
        residuals=np.vstack(residual_blocks),
        solutions_current=np.vstack(solution_current_blocks),
        errors=np.vstack(error_blocks),
        true_solutions=sols,
        sample_indices=np.concatenate(sample_indices),
        iteration_indices=np.concatenate(iteration_indices),
    )

    return StrategyOutput(
        rhs=rhs_samples,
        solutions=sols,
        error_traces=error_traces,
    )


# =============================================================================
# EIGENVECTOR STRATEGY HELPERS
# =============================================================================


def _compute_eigendecomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigendecomposition of symmetric matrix using scipy."""
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
    """Select subset of eigenvectors according to eigenvalue range."""
    n = eigenvectors.shape[0]
    if count > n:
        raise ValueError(
            f"Requested {count} samples but matrix has only {n} eigenvectors. "
            f"Maximum samples: {n}"
        )
    if count <= 0:
        raise ValueError(f"Sample count must be positive, got {count}")
    if which == "smallest":
        indices = np.arange(count)
    elif which == "largest":
        indices = np.arange(n - count, n)
    elif which == "random":
        indices = rng.choice(n, size=count, replace=False)
    else:
        raise ValueError(
            f"Invalid which: '{which}'. Must be 'smallest', 'largest', or 'random'"
        )
    return eigenvectors[:, indices], eigenvalues[indices], indices


def _generate_eigenvector_combinations(
    eigenvectors: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate random L2-normalized linear combinations of eigenvectors."""
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
    """Verify solution accuracy by computing relative residuals."""
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


def eigenvector_forward_strategy(
    A: np.ndarray,
    b: np.ndarray,
    count: int,
    rng: np.random.Generator,
    match_rhs_norm: bool = False,
    target_rhs_scale: float = 1.0,
    which: str = "smallest",
    num_eigenvectors: int | None = None,
    include_eigenvectors: bool = False,
) -> StrategyOutput:
    """Generate samples using eigenvectors or their linear combinations as solutions."""
    n = A.shape[0]
    if num_eigenvectors is None or num_eigenvectors == -1:
        num_eigenvectors = n
    assert num_eigenvectors is not None  # Type narrowing for pyright
    if num_eigenvectors <= 0 or num_eigenvectors > n:
        raise ValueError(f"num_eigenvectors must be positive and ≤ {n}")
    if include_eigenvectors and count < num_eigenvectors:
        raise ValueError(
            f"When include_eigenvectors=True, sample count ({count}) must be >= "
            f"num_eigenvectors ({num_eigenvectors})."
        )

    eigenvalues, eigenvectors = _compute_eigendecomposition(A)
    selected_eigenvectors, selected_eigenvalues, _ = _select_eigenvectors(
        eigenvectors, eigenvalues, num_eigenvectors, which, rng
    )

    if include_eigenvectors:
        num_combinations = count - num_eigenvectors
        if num_combinations > 0:
            X_combinations = _generate_eigenvector_combinations(
                selected_eigenvectors, num_combinations, rng
            )
            X = np.vstack([selected_eigenvectors.T, X_combinations])
        else:
            X = selected_eigenvectors.T
    else:
        X = _generate_eigenvector_combinations(selected_eigenvectors, count, rng)

    R = np.array([A @ x for x in X], dtype=np.float64)
    _verify_solution_accuracy(A, R, X, tolerance=1e-14)
    return StrategyOutput(rhs=R, solutions=X)


def eigenvector_inverse_strategy(
    A: np.ndarray,
    b: np.ndarray,
    count: int,
    rng: np.random.Generator,
    match_rhs_norm: bool = False,
    target_rhs_scale: float = 1.0,
    which: str = "smallest",
    solution_tolerance: float = EIGENVECTOR_SOLUTION_TOLERANCE,
    num_eigenvectors: int | None = None,
    include_eigenvectors: bool = False,
) -> StrategyOutput:
    """Generate samples using eigenvectors or their linear combinations as RHS, solving for x."""
    n = A.shape[0]
    if num_eigenvectors is None or num_eigenvectors == -1:
        num_eigenvectors = n
    assert num_eigenvectors is not None  # Type narrowing for pyright
    if num_eigenvectors <= 0 or num_eigenvectors > n:
        raise ValueError(f"num_eigenvectors must be positive and ≤ {n}")
    if include_eigenvectors and count < num_eigenvectors:
        raise ValueError(
            f"When include_eigenvectors=True, sample count ({count}) must be >= "
            f"num_eigenvectors ({num_eigenvectors})."
        )

    eigenvalues, eigenvectors = _compute_eigendecomposition(A)
    selected_eigenvectors, _, _ = _select_eigenvectors(
        eigenvectors, eigenvalues, num_eigenvectors, which, rng
    )

    if include_eigenvectors:
        num_combinations = count - num_eigenvectors
        if num_combinations > 0:
            R_combinations = _generate_eigenvector_combinations(
                selected_eigenvectors, num_combinations, rng
            )
            R = np.vstack([selected_eigenvectors.T, R_combinations])
        else:
            R = selected_eigenvectors.T
    else:
        R = _generate_eigenvector_combinations(selected_eigenvectors, count, rng)

    X = np.zeros((count, n), dtype=np.float64)
    for i in range(count):
        X[i] = solve(A, R[i], assume_a="pos")
    return StrategyOutput(rhs=R, solutions=X)


# =============================================================================
# TRACE MANIPULATION
# =============================================================================


def _offset_residual_traces(
    traces: ResidualTraceSamples,
    offset: int,
) -> ResidualTraceSamples:
    return ResidualTraceSamples(
        residuals=traces.residuals,
        solutions=traces.solutions,
        sample_indices=traces.sample_indices + offset,
        iteration_indices=traces.iteration_indices,
        search_directions=traces.search_directions,
        search_direction_products=traces.search_direction_products,
    )


def _merge_residual_traces(
    blocks: list[ResidualTraceSamples],
) -> ResidualTraceSamples:
    residuals = np.vstack([block.residuals for block in blocks])
    solutions = np.vstack([block.solutions for block in blocks])
    sample_indices = np.concatenate([block.sample_indices for block in blocks])
    iteration_indices = np.concatenate([block.iteration_indices for block in blocks])
    search_directions = None
    search_direction_products = None

    has_search_directions = [block.search_directions is not None for block in blocks]
    has_search_direction_products = [
        block.search_direction_products is not None for block in blocks
    ]

    if any(has_search_directions) and not all(has_search_directions):
        raise ValueError(
            "Residual trace blocks must uniformly include search_directions."
        )
    if any(has_search_direction_products) and not all(has_search_direction_products):
        raise ValueError(
            "Residual trace blocks must uniformly include search_direction_products."
        )

    if all(has_search_directions):
        # All blocks have search_directions (validated above), filter None for type safety
        search_directions = np.vstack(
            [
                block.search_directions
                for block in blocks
                if block.search_directions is not None
            ]
        )
    if all(has_search_direction_products):
        # All blocks have search_direction_products (validated above), filter None for type safety
        search_direction_products = np.vstack(
            [
                block.search_direction_products
                for block in blocks
                if block.search_direction_products is not None
            ]
        )
    return ResidualTraceSamples(
        residuals=residuals,
        solutions=solutions,
        sample_indices=sample_indices,
        iteration_indices=iteration_indices,
        search_directions=search_directions,
        search_direction_products=search_direction_products,
    )


def _offset_error_traces(
    traces: ErrorTraceSamples,
    offset: int,
) -> ErrorTraceSamples:
    return ErrorTraceSamples(
        residuals=traces.residuals,
        solutions_current=traces.solutions_current,
        errors=traces.errors,
        true_solutions=traces.true_solutions,
        sample_indices=traces.sample_indices + offset,
        iteration_indices=traces.iteration_indices,
    )


def _merge_error_traces(
    blocks: list[ErrorTraceSamples],
) -> ErrorTraceSamples:
    residuals = np.vstack([block.residuals for block in blocks])
    solutions_current = np.vstack([block.solutions_current for block in blocks])
    errors = np.vstack([block.errors for block in blocks])
    true_solutions = np.vstack([block.true_solutions for block in blocks])
    sample_indices = np.concatenate([block.sample_indices for block in blocks])
    iteration_indices = np.concatenate([block.iteration_indices for block in blocks])
    return ErrorTraceSamples(
        residuals=residuals,
        solutions_current=solutions_current,
        errors=errors,
        true_solutions=true_solutions,
        sample_indices=sample_indices,
        iteration_indices=iteration_indices,
    )


# =============================================================================
# ORCHESTRATION
# =============================================================================


def _execute_strategy(
    strategy_name: str,
    count: int,
    A: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    match_rhs_norm: bool,
    target_rhs_scale: float,
    krylov_iters: int,
    residual_iters: int,
    counts_represent_final_pairs: bool,
    strategy_options: dict[str, Any],
    archive_solutions: np.ndarray | None,
    archive_rhs: np.ndarray | None,
) -> StrategyOutput:
    raise ValueError("Legacy strategy execution path removed; use generation runner.")


def _shuffle_samples(
    X: np.ndarray,
    Y: np.ndarray,
    residual_traces: ResidualTraceSamples | None,
    error_traces: ErrorTraceSamples | None,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    ResidualTraceSamples | None,
    ErrorTraceSamples | None,
]:
    indices = rng.permutation(len(X))
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    inverse = np.empty_like(indices)
    inverse[indices] = np.arange(len(indices))

    if residual_traces is not None:
        residual_traces = ResidualTraceSamples(
            residuals=residual_traces.residuals,
            solutions=residual_traces.solutions,
            sample_indices=inverse[residual_traces.sample_indices],
            iteration_indices=residual_traces.iteration_indices,
            search_directions=residual_traces.search_directions,
            search_direction_products=residual_traces.search_direction_products,
        )

    return X_shuffled, Y_shuffled, residual_traces, error_traces


def generate_mixture(
    A: np.ndarray,
    b: np.ndarray | None = None,
    counts: Mapping[str, int] | None = None,
    *,
    mix: Mapping[str, float] | None = None,
    total: int | None = None,
    counts_represent_final_pairs: bool = False,
    krylov_iters: int = 15,
    residual_iters: int = 8,
    seed: int = 42,
    shuffle: bool = True,
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    archive_solutions: np.ndarray | None = None,
    archive_rhs: np.ndarray | None = None,
) -> tuple[
    np.ndarray, np.ndarray, ResidualTraceSamples | None, ErrorTraceSamples | None
]:
    """Generate mixed training data from multiple strategies.

    Args:
        A: System matrix
        b: Mother RHS vector (optional - only required for strategies that need it)
    """
    # Ensure strategy modules are registered
    from . import strategies  # noqa: F401
    from .runner import run_generation

    rng = rng_from_seed(seed)
    strategy_counts = _resolve_strategy_counts(counts, mix, total)
    overrides: dict[str, dict[str, Any]] = {
        name: dict(options) for name, options in (strategy_overrides or {}).items()
    }

    all_features = []
    all_targets = []
    residual_blocks: list[ResidualTraceSamples] = []
    error_blocks: list[ErrorTraceSamples] = []
    sample_offset = 0

    for strategy_name, count in strategy_counts.items():
        if count == 0:
            continue

        cfg = overrides.get(strategy_name, {}).copy()
        cfg.setdefault("samples", count)
        cfg.setdefault("seed", seed)
        cfg.setdefault("krylov_iters", krylov_iters)
        cfg.setdefault("residual_iters", residual_iters)
        cfg.setdefault("archive_solutions", archive_solutions)
        cfg.setdefault("archive_rhs", archive_rhs)

        adjusted_count = count
        if counts_represent_final_pairs and strategy_name in {
            "cg_residual",
            "residual",
            "cg_residual_error",
            "residual_error",
        }:
            adjusted_count = max(1, (count + residual_iters - 1) // residual_iters)
            cfg["samples"] = adjusted_count

        # Only pass b if strategy requires it
        from .runner import _registry

        strategy = _registry.get(strategy_name)
        rhs_to_pass = b if strategy.requires_rhs() else None
        generated = run_generation(strategy_name, A, rhs_to_pass, cfg=cfg)

        if generated.rhs is not None:
            all_features.append(generated.rhs)
        if generated.solutions is not None:
            all_targets.append(generated.solutions)

        if generated.residual_traces is not None:
            residual_blocks.append(
                _offset_residual_traces(generated.residual_traces, sample_offset)
            )
        if generated.error_traces is not None:
            error_blocks.append(generated.error_traces)

        if generated.rhs is not None:
            sample_offset += generated.rhs.shape[0]

    X, Y = _merge_strategy_outputs(all_features, all_targets)
    residual_traces = (
        _merge_residual_traces(residual_blocks) if residual_blocks else None
    )
    error_traces = _merge_error_traces(error_blocks) if error_blocks else None

    if shuffle:
        X, Y, residual_traces, error_traces = _shuffle_samples(
            X, Y, residual_traces, error_traces, rng
        )

    return X, Y, residual_traces, error_traces
