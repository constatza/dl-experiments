"""Orchestration functions for mixed-strategy data generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from loguru import logger

from .data_types import NormalizeType
from ..normalization import ErrorTraceSamples, ResidualTraceSamples
from .helpers import rng_from_seed, _resolve_strategy_counts, _merge_strategy_outputs
from .trace_utils import (
    _offset_residual_traces,
    _merge_residual_traces,
    _merge_error_traces,
)
from .interfaces import ArchiveData
from ..constants import DEFAULT_RESIDUAL_TRACE_ITERS


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
    """Shuffle samples while maintaining trace consistency.

    Args:
        X: Feature array, shape (N, n)
        Y: Target array, shape (N, n)
        residual_traces: Optional residual trace samples
        error_traces: Optional error trace samples
        rng: Random number generator

    Returns:
        Tuple of (shuffled_X, shuffled_Y, shuffled_residual_traces, error_traces)
    """
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
    counts: Mapping[str, int] | None = None,
    *,
    mix: Mapping[str, float] | None = None,
    total: int | None = None,
    counts_represent_final_pairs: bool = False,
    seed: int = 42,
    shuffle: bool = True,
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    archive_solutions: np.ndarray | None = None,
    archive_rhs: np.ndarray | None = None,
    single_rhs: np.ndarray | None = None,
) -> tuple[
    np.ndarray, np.ndarray, ResidualTraceSamples | None, ErrorTraceSamples | None
]:
    """Generate mixed training data from multiple strategies.

    Args:
        A: System matrix, shape (n, n)
        counts: Optional explicit strategy counts
        mix: Optional strategy proportions (used with total)
        total: Total samples (required if mix provided)
        counts_represent_final_pairs: If True, counts refer to final iteration pairs
            (used for trace strategies to compute number of solver runs needed)
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle final samples
        strategy_overrides: Per-strategy configuration overrides. Use this to configure
            strategy-specific parameters like krylov_iters and cg_iters.
        archive_solutions: Pre-computed solutions for archive-based generation
        archive_rhs: Pre-computed RHS vectors for archive-based generation
        single_rhs: Optional single RHS vector, shape (n,). If provided to single-RHS strategies
            (trace strategies), all samples will solve the same system A @ x = single_rhs

    Returns:
        Tuple of (features, targets, residual_traces, error_traces):
            - features: RHS vectors, shape (N, n)
            - targets: Solution vectors, shape (N, n)
            - residual_traces: Optional residual trace samples from CG
            - error_traces: Optional error trace samples from CG

    Raises:
        ValueError: If counts/mix arguments invalid or strategies unknown

    Examples:
        >>> # Generate 100 samples with equal mix of normal and krylov
        >>> X, Y, res_traces, err_traces = generate_mixture(
        ...     A,
        ...     mix={"normal": 1.0, "krylov": 1.0},
        ...     total=100,
        ...     seed=42,
        ...     strategy_overrides={
        ...         "krylov": {"krylov_iters": 20},
        ...     }
        ... )
        >>> X.shape
        (100, n)

        >>> # Generate explicit counts with strategy-specific configuration
        >>> X, Y, _, _ = generate_mixture(
        ...     A,
        ...     counts={"normal": 50, "krylov": 30, "cg_residual": 20},
        ...     seed=42,
        ...     strategy_overrides={
        ...         "cg_residual": {"cg_iters": 10},
        ...     }
        ... )

        >>> # Generate with single RHS for trace strategies
        >>> rhs = np.random.randn(n)
        >>> X, Y, res_traces, _ = generate_mixture(
        ...     A,
        ...     counts={"cg_residual": 20},
        ...     single_rhs=rhs,  # All 20 samples solve A @ x = rhs
        ...     seed=42,
        ... )
    """
    # Ensure strategy modules are registered
    from . import strategies  # noqa: F401
    from .runner import run_generation
    from pydantic import ValidationError

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
        
        archive_data: ArchiveData | None = None
        if archive_solutions is not None:
            archive_data = ArchiveData(solutions=archive_solutions, rhs_vectors=archive_rhs)

        # For counts_represent_final_pairs mode, we need to read cg_iters from config
        adjusted_count = count
        if counts_represent_final_pairs and strategy_name in {
            "cg_residual",
            "residual",
            "cg_residual_error",
            "residual_error",
            "search_directions",
        }:
            # Get cg_iters and every_n from strategy override or use defaults
            strategy_cg_iters = cfg.get("cg_iters", DEFAULT_RESIDUAL_TRACE_ITERS)
            every_n = cfg.get("every_n", 1)
            # Pairs per base system = ceil((cg_iters + 1) / every_n)
            # The +1 accounts for the initial residual recorded before any CG step
            pairs_per_system = (strategy_cg_iters + every_n) // every_n
            adjusted_count = max(
                1, (count + pairs_per_system - 1) // pairs_per_system
            )
            cfg["samples"] = adjusted_count

        # Run generation (all strategies now use unified interface)
        # Single-RHS strategies (trace strategies) will receive single_rhs if provided
        # Pydantic (extra="forbid") will raise ValidationError on unknown keys — fail fast.
        try:
            generated = run_generation(
                strategy_name,
                A,
                cfg=cfg,
                archive=archive_data,
                single_rhs=single_rhs,
            )
        except ValidationError as e:
            raise ValueError(
                f"Invalid configuration for strategy '{strategy_name}': {e}"
            ) from e

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


def build_dataset(
    matrix_path: str,
    dataset_dir: str,
    *,
    counts: dict[str, int] | None = None,
    mix: dict[str, float] | None = None,
    total: int | None = None,
    rhs_path: str | None = None,
    normalize: NormalizeType = "matrix",
    matrix_norm_type: str = "spectral",
    shuffle: bool = True,
    seed: int = 42,
    strategy_overrides: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Unified dataset builder for all data sources (archives + synthetic).

    This function replaces collect_case(), generate_case(), and solution_archive_case()
    by treating archive collection as generation strategies that can be mixed freely
    with synthetic strategies.

    Args:
        matrix_path: Path to system matrix file
        dataset_dir: Output directory for dataset
        counts: Explicit strategy counts (e.g., {"random": 100, "rhs_archive": 50})
        mix: Strategy proportions (used with total)
        total: Total samples (required if mix provided)
        rhs_path: Path to single RHS file (optional, for single-RHS strategies like trace strategies)
        normalize: Normalization type ("none", "matrix", "rhs", "spectral", "diagonal")
        matrix_norm_type: Type of matrix norm to compute and save (default: "spectral").
            Options: "spectral", "frobenius", "nuclear", "one", "inf"
        shuffle: Whether to shuffle final samples
        seed: Random seed for reproducibility
        strategy_overrides: Per-strategy configuration overrides
            For archive strategies:
            - rhs_archive: {" rhs_glob": "/path/*.txt", "solve_systems": True, ...}
            - solution_archive: {"solutions_glob": "/path/*.txt", ...}

    Returns:
        Path to created dataset directory

    Examples:
        >>> # Pure synthetic generation
        >>> build_dataset(
        ...     "matrix.txt",
        ...     "output/",
        ...     counts={"random": 100, "krylov": 50},
        ...     rhs_path="rhs.txt",
        ... )

        >>> # Pure archive collection
        >>> build_dataset(
        ...     "matrix.txt",
        ...     "output/",
        ...     counts={"rhs_archive": 200},
        ...     strategy_overrides={
        ...         "rhs_archive": {"rhs_glob": "/data/rhs_*.txt"}
        ...     },
        ... )

        >>> # Mixed archives + synthetic (THE KEY BENEFIT!)
        >>> build_dataset(
        ...     "matrix.txt",
        ...     "output/",
        ...     counts={
        ...         "rhs_archive": 100,
        ...         "solution_archive": 50,
        ...         "random": 200,
        ...         "krylov": 150,
        ...     },
        ...     rhs_path="rhs.txt",
        ...     strategy_overrides={
        ...         "rhs_archive": {"rhs_glob": "/data/rhs_*.txt"},
        ...         "solution_archive": {"solutions_glob": "/data/sol_*.txt"},
        ...     },
        ... )

    Raises:
        ValueError: If arguments invalid or strategies unknown
        FileNotFoundError: If matrix or RHS files not found
    """
    from pathlib import Path
    from ..io.base import load_matrix
    from .types import RawSamples
    from .helpers import _normalize_matrix_for_generation

    logger.info("Building dataset...")
    logger.info(f"  Matrix: {matrix_path}")
    logger.info(f"  Output: {dataset_dir}")

    # Step 1: Load matrix
    matrix = load_matrix(Path(matrix_path))
    dimension = matrix.shape[0]
    logger.info(f"  Dimension: {dimension}")

    # Step 2: Load single RHS if provided (for single-RHS strategies like trace strategies)
    single_rhs: np.ndarray | None
    if rhs_path:
        single_rhs = np.loadtxt(rhs_path, dtype=np.float64)
        if single_rhs.ndim > 1:
            single_rhs = single_rhs.reshape(-1)
    else:
        # No single RHS - trace strategies will generate multiple RHS vectors
        single_rhs = None

    # Step 3: Normalize matrix BEFORE strategy execution
    logger.info(f"  Normalization: {normalize}")
    matrix_norm, scale = _normalize_matrix_for_generation(
        matrix, normalize, spectral_radius_bound=None
    )

    # Step 4: Normalize single RHS (if provided and scale exists)
    single_rhs_norm: np.ndarray | None
    if single_rhs is not None and scale is not None:
        single_rhs_norm = scale.scale_rhs(single_rhs)
    else:
        single_rhs_norm = single_rhs

    # Step 5: Call generate_mixture() with ALL strategies (archives + synthetic)
    # Note: krylov_iters and cg_iters are now passed via strategy_overrides
    logger.info("Generating/loading samples...")
    X, Y, residual_traces, error_traces = generate_mixture(
        matrix_norm,
        counts=counts,
        mix=mix,
        total=total,
        seed=seed,
        shuffle=shuffle,
        strategy_overrides=strategy_overrides,
        single_rhs=single_rhs_norm,
    )

    logger.info(f"  Generated {X.shape[0]} samples")

    # Step 6: Use error trace pairs if available, otherwise use base pairs
    # Error trace pairs (r_k, e_k) include the original (A@x, x) pairs at iteration 0
    # since r_0 = b when x_0 = 0
    if error_traces is not None:
        logger.info("Using error trace pairs (r_k, e_k) for training")
        X_final = error_traces.residuals
        Y_final = error_traces.errors
    else:
        logger.info("Using base pairs (A@x, x) for training")
        X_final = X
        Y_final = Y

    # Step 7: Package into RawSamples (only essential data - no strategy internals)
    raw_samples = RawSamples(
        matrix=matrix_norm,
        rhs=X_final,
        solutions=Y_final,
    )

    # Step 8: Persist dataset
    from pathlib import Path
    from ..math_utils import calculate_matrix_norm
    from ..constants import NORMALIZED_DATASET_FILENAME

    dataset_dir_path = Path(dataset_dir)
    dataset_dir_path.mkdir(parents=True, exist_ok=True)

    # Persist normalized samples WITH scale metadata for denormalization
    logger.info(f"Saving to: {dataset_dir}")
    normalized_file = dataset_dir_path / NORMALIZED_DATASET_FILENAME

    # Compute matrix norm of normalized matrix using specified norm type
    # StrEnum handles validation internally - users just pass strings
    matrix_norm_value = calculate_matrix_norm(raw_samples.matrix, norm_type=matrix_norm_type)
    logger.info(f"  Normalized matrix {matrix_norm_type} norm: {matrix_norm_value:.6f}")

    # Build save dict with normalized data
    # Note: Structure is identical for both trace and base pairs
    save_dict = {
        "matrix": raw_samples.matrix,
        "rhs": raw_samples.rhs,          # Either residuals (r_k) or RHS (A@x)
        "solutions": raw_samples.solutions,  # Either errors (e_k) or solutions (x)
        "normalize_type": normalize,  # Save normalization type for reconstruction
        "matrix_norm": matrix_norm_value,  # Matrix norm of normalized matrix
        "matrix_norm_type": matrix_norm_type,  # Which norm was computed
    }

    # Add scale parameters using IScale.to_dict() (for denormalization)
    if scale is not None:
        scale_params = scale.to_dict()
        save_dict.update(scale_params)  # Merge scale parameters into save dict
        logger.debug(f"  Saved scale metadata: {list(scale_params.keys())}")

    np.savez(normalized_file, **save_dict)

    logger.info(f"Dataset built successfully: {dataset_dir}")
    return dataset_dir


__all__ = [
    "_shuffle_samples",
    "generate_mixture",
    "build_dataset",
]
