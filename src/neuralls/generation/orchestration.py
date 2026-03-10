"""Orchestration functions for mixed-strategy data generation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
from loguru import logger
from scipy.sparse import csc_matrix

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
from .source_streams import bind_sources, open_matrix_stream, open_vector_stream


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
        counts_represent_final_pairs: Deprecated compatibility flag. Trace strategies
            now interpret counts as final output rows directly.
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

        cfg["samples"] = count

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

    if all_features and all_targets:
        X, Y = _merge_strategy_outputs(all_features, all_targets)
    else:
        X = np.empty((0, A.shape[0]), dtype=np.float64)
        Y = np.empty((0, A.shape[0]), dtype=np.float64)
    residual_traces = (
        _merge_residual_traces(residual_blocks) if residual_blocks else None
    )
    error_traces = _merge_error_traces(error_blocks) if error_blocks else None

    if shuffle and X.shape[0] > 0:
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
    solutions_path: str | None = None,
    sample_id_regex: str | None = None,
    normalize: NormalizeType = "matrix",
    matrix_norm_type: str = "spectral",
    shuffle: bool = True,
    seed: int = 42,
    strategy_overrides: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Build dataset from streamed matrix sources without dense N-matrix materialization."""
    from pathlib import Path
    from ..math_utils import calculate_matrix_norm
    from ..io.dataset_storage import (
        SparsePackAccumulator,
        save_dataset_from_sparse,
    )
    from .helpers import _normalize_matrix_for_generation

    logger.info("Building dataset...")
    logger.info(f"  Matrix: {matrix_path}")
    logger.info(f"  Output: {dataset_dir}")
    if solutions_path is not None:
        logger.info(f"  Solutions source: {solutions_path}")
    if rhs_path is not None:
        logger.info(f"  RHS source: {rhs_path}")

    matrix_stream = open_matrix_stream(
        matrix_path_expr=matrix_path,
        sample_id_regex=sample_id_regex,
    )
    rhs_stream = (
        open_vector_stream(rhs_path, sample_id_regex=sample_id_regex)
        if rhs_path is not None
        else None
    )
    bindings = bind_sources(
        matrix_ids=matrix_stream.sample_ids,
        rhs_ids=rhs_stream.sample_ids if rhs_stream is not None else None,
        solution_ids=None,
    )
    logger.info(
        f"  Matrix samples: {len(matrix_stream.sample_ids)} | System bindings: {len(bindings)}"
    )

    sparse_acc = SparsePackAccumulator()
    single_matrix_mode = len(matrix_stream.sample_ids) == 1
    single_matrix_written = False
    rhs_blocks: list[np.ndarray] = []
    solution_blocks: list[np.ndarray] = []
    matrix_norm_values: list[float] = []
    matrix_value_scale_values: list[float] = []
    scale_metadata_values: list[dict[str, Any] | None] = []
    matrix_cache: dict[
        int, tuple[np.ndarray, Any, Any, float, float, dict[str, Any] | None]
    ] = {}

    logger.info(f"  Normalization: {normalize}")
    for binding in bindings:
        cached = matrix_cache.get(binding.matrix_sample_id)
        if cached is None:
            dense_sample = matrix_stream.load_dense_sample(binding.matrix_sample_id)
            dense_matrix = np.asarray(dense_sample.matrix, dtype=np.float64)
            if dense_matrix.shape[0] != dense_matrix.shape[1]:
                raise ValueError(
                    f"Matrix sample {binding.matrix_sample_id} must be square, got {dense_matrix.shape}"
                )
            matrix_norm, scale, matrix_value_scale = _normalize_matrix_for_generation(
                dense_matrix,
                normalize,
                spectral_radius_bound=None,
            )
            matrix_for_generation = csc_matrix(matrix_norm)
            matrix_norm_value = calculate_matrix_norm(
                matrix_norm,
                norm_type=matrix_norm_type,
            )
            scale_params = scale.to_dict() if scale is not None else None
            cached = (
                matrix_norm,
                matrix_for_generation,
                scale,
                matrix_norm_value,
                matrix_value_scale,
                scale_params,
            )
            matrix_cache[binding.matrix_sample_id] = cached
        (
            matrix_norm,
            matrix_for_generation,
            scale,
            matrix_norm_value,
            matrix_value_scale,
            scale_params,
        ) = cached

        single_rhs: np.ndarray | None = None
        if rhs_stream is not None and binding.rhs_sample_id is not None:
            rhs_sample = rhs_stream.load_sample(binding.rhs_sample_id)
            single_rhs = np.asarray(rhs_sample.vector, dtype=np.float64)
            if single_rhs.shape[0] != matrix_norm.shape[0]:
                raise ValueError(
                    f"RHS sample {binding.rhs_sample_id} length {single_rhs.shape[0]} "
                    f"doesn't match matrix size {matrix_norm.shape[0]}"
                )
            if scale is not None:
                single_rhs = scale.scale_rhs(single_rhs)

        logger.info(
            f"Generating/loading samples for binding sample_id={binding.sample_id} "
            f"(matrix_id={binding.matrix_sample_id})..."
        )
        X, Y, residual_traces, error_traces = generate_mixture(
            matrix_for_generation,
            counts=counts,
            mix=mix,
            total=total,
            seed=seed,
            shuffle=shuffle,
            strategy_overrides=strategy_overrides,
            single_rhs=single_rhs,
        )

        if error_traces is not None:
            X_final = error_traces.residuals
            Y_final = error_traces.errors
        elif residual_traces is not None:
            X_final = residual_traces.residuals
            Y_final = residual_traces.solutions
        else:
            X_final = X
            Y_final = Y

        if X_final.shape != Y_final.shape:
            raise ValueError(
                f"Generated RHS/solution shape mismatch: {X_final.shape} vs {Y_final.shape}"
            )
        if X_final.shape[0] == 0:
            continue

        rhs_blocks.append(np.asarray(X_final, dtype=np.float64))
        solution_blocks.append(np.asarray(Y_final, dtype=np.float64))
        if single_matrix_mode:
            if not single_matrix_written:
                # Persist single source matrix once; dlkit reader broadcasts n_samples=1 at runtime.
                sparse_acc.append_dense_matrix(matrix_norm, repeats=1)
                single_matrix_written = True
        else:
            sparse_acc.append_dense_matrix(matrix_norm, repeats=int(X_final.shape[0]))
        matrix_norm_values.append(float(matrix_norm_value))
        matrix_value_scale_values.append(float(matrix_value_scale))
        scale_metadata_values.append(scale_params)

    if not rhs_blocks or not solution_blocks:
        raise ValueError("No samples were generated for dataset persistence.")

    rhs_all = np.vstack(rhs_blocks)
    solutions_all = np.vstack(solution_blocks)
    indices, values, nnz_ptr, matrix_size = sparse_acc.build_arrays()

    dataset_dir_path = Path(dataset_dir)
    dataset_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to: {dataset_dir}")

    matrix_norm_value = float(matrix_norm_values[0])
    if not all(np.isclose(v, matrix_norm_value, rtol=1e-10, atol=1e-12) for v in matrix_norm_values):
        logger.warning(
            "Multiple normalized matrix norms detected across bindings; "
            "persisting first value in manifest metadata."
        )
    matrix_value_scale = float(matrix_value_scale_values[0])
    if not all(
        np.isclose(v, matrix_value_scale, rtol=1e-10, atol=1e-12)
        for v in matrix_value_scale_values
    ):
        logger.warning(
            "Multiple matrix value scales detected across bindings; "
            "persisting sparse pack with value_scale=1.0."
        )
        matrix_value_scale = 1.0

    unique_scale_payloads = {
        json.dumps(payload, sort_keys=True) if payload is not None else "null"
        for payload in scale_metadata_values
    }
    scale_metadata: dict[str, Any] | None = None
    if len(unique_scale_payloads) == 1:
        scale_metadata = scale_metadata_values[0]
    else:
        logger.warning(
            "Multiple scale metadata payloads detected across bindings; "
            "persisting empty scale metadata in manifest."
        )
        scale_metadata = {}

    save_dataset_from_sparse(
        dataset_dir=dataset_dir_path,
        rhs=rhs_all,
        solutions=solutions_all,
        indices=indices,
        values=values,
        nnz_ptr=nnz_ptr,
        size=matrix_size,
        normalization_type=str(normalize),
        matrix_norm=matrix_norm_value,
        matrix_norm_type=matrix_norm_type,
        matrix_value_scale=matrix_value_scale,
        scale_metadata=scale_metadata,
    )

    logger.info(
        f"Dataset built successfully: {dataset_dir} "
        f"(samples={rhs_all.shape[0]}, matrix_samples={len(bindings)})"
    )
    return dataset_dir


__all__ = [
    "_shuffle_samples",
    "generate_mixture",
    "build_dataset",
]
