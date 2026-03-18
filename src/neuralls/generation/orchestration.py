"""Orchestration functions for mixed-strategy data generation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from scipy.sparse import csc_matrix

from .data_types import NormalizeType, ScaleMetadata
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


@dataclass(frozen=True)
class _CachedMatrix:
    """Immutable cached matrix data for generation.

    Stores all derived matrices and scaling information computed once
    per unique matrix_sample_id, avoiding redundant computation.

    Attributes:
        matrix_norm: Normalized system matrix (as dense array)
        matrix_for_generation: CSC sparse version of normalized matrix
        scale: IScale object or None (scaling strategy applied)
        matrix_norm_value: Computed matrix norm value
        matrix_value_scale: Scaling factor applied
        scale_params: Dictionary of scale parameters or None
    """

    matrix_norm: np.ndarray
    matrix_for_generation: Any  # csc_matrix; avoid top-level scipy import
    scale: Any  # IScale | None
    matrix_norm_value: float
    matrix_value_scale: float
    scale_params: dict[str, Any] | None


def _open_streams(
    matrix_path: str,
    rhs_path: str | None,
    sample_id_regex: str | None,
) -> tuple[Any, Any, list[Any]]:
    """Open matrix and optional RHS streams and bind them.

    Args:
        matrix_path: Path expression for matrix stream
        rhs_path: Optional path expression for RHS stream
        sample_id_regex: Optional regex for sample ID extraction

    Returns:
        Tuple of (matrix_stream, rhs_stream, bindings)
    """
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
    return matrix_stream, rhs_stream, bindings


@dataclass(frozen=True)
class _BindingResult:
    """Result from processing one binding.

    Stores generated and accumulated data from a single matrix-RHS binding.

    Attributes:
        rhs_block: Generated RHS block for this binding
        solution_block: Generated solution block for this binding
        matrix_norm_value: Computed norm of normalized matrix
        matrix_value_scale: Scaling factor applied to matrix values
        scale_params: Dictionary of scale parameters or None
    """

    rhs_block: np.ndarray
    solution_block: np.ndarray
    matrix_norm_value: float
    matrix_value_scale: float
    scale_params: ScaleMetadata | None


def _process_binding(
    binding: Any,
    matrix_stream: Any,
    rhs_stream: Any,
    matrix_cache: dict[int, _CachedMatrix],
    normalize: NormalizeType,
    matrix_norm_type: str,
    counts: dict[str, int] | None,
    mix: dict[str, float] | None,
    total: int | None,
    seed: int,
    shuffle: bool,
    strategy_overrides: dict[str, dict[str, Any]] | None,
) -> _BindingResult:
    """Process one matrix-RHS binding and generate samples.

    Args:
        binding: Single binding to process
        matrix_stream: Opened matrix stream
        rhs_stream: Optional opened RHS stream
        matrix_cache: Cache of computed matrices
        normalize: Normalization type
        matrix_norm_type: Norm type for matrix calculation
        counts: Sample counts per strategy
        mix: Mixing proportions
        total: Total samples
        seed: Random seed
        shuffle: Whether to shuffle samples
        strategy_overrides: Strategy-specific overrides

    Returns:
        BindingResult with generated data blocks
    """
    from ..math_utils import calculate_matrix_norm

    # Get or compute normalized matrix
    cached = matrix_cache.get(binding.matrix_sample_id)
    if cached is None:
        from .helpers import _normalize_matrix_for_generation

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
        cached = _CachedMatrix(
            matrix_norm=matrix_norm,
            matrix_for_generation=matrix_for_generation,
            scale=scale,
            matrix_norm_value=matrix_norm_value,
            matrix_value_scale=matrix_value_scale,
            scale_params=scale_params,
        )
        matrix_cache[binding.matrix_sample_id] = cached

    # Load optional RHS
    single_rhs: np.ndarray | None = None
    if rhs_stream is not None and binding.rhs_sample_id is not None:
        rhs_sample = rhs_stream.load_sample(binding.rhs_sample_id)
        single_rhs = np.asarray(rhs_sample.vector, dtype=np.float64)
        if single_rhs.shape[0] != cached.matrix_norm.shape[0]:
            raise ValueError(
                f"RHS sample {binding.rhs_sample_id} length {single_rhs.shape[0]} "
                f"doesn't match matrix size {cached.matrix_norm.shape[0]}"
            )
        if cached.scale is not None:
            single_rhs = cached.scale.scale_rhs(single_rhs)

    # Generate mixture
    logger.info(
        f"Generating/loading samples for binding sample_id={binding.sample_id} "
        f"(matrix_id={binding.matrix_sample_id})..."
    )
    X, Y, residual_traces, error_traces = generate_mixture(
        cached.matrix_for_generation,
        counts=counts,
        mix=mix,
        total=total,
        seed=seed,
        shuffle=shuffle,
        strategy_overrides=strategy_overrides,
        single_rhs=single_rhs,
    )

    # Select final blocks based on available traces
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

    return _BindingResult(
        rhs_block=np.asarray(X_final, dtype=np.float64),
        solution_block=np.asarray(Y_final, dtype=np.float64),
        matrix_norm_value=float(cached.matrix_norm_value),
        matrix_value_scale=float(cached.matrix_value_scale),
        scale_params=cached.scale_params,  # type: ignore[arg-type]  # ScaleMetadata compatible
    )


def _resolve_final_scale(
    norm_values: list[float],
    scale_values: list[float],
    metadata_values: list[ScaleMetadata | None],
) -> tuple[float, float, ScaleMetadata | None]:
    """Resolve final scaling parameters from all bindings.

    Checks consistency across bindings and logs warnings for inconsistencies.

    Args:
        norm_values: Matrix norm values from each binding
        scale_values: Matrix value scale factors from each binding
        metadata_values: Scale metadata from each binding

    Returns:
        Tuple of (final_matrix_norm, final_matrix_scale, final_scale_metadata)
    """
    if not norm_values or not scale_values:
        raise ValueError("No norm or scale values to resolve")

    # Resolve matrix norm
    matrix_norm_value = float(norm_values[0])
    if not all(np.isclose(v, matrix_norm_value, rtol=1e-10, atol=1e-12) for v in norm_values):
        logger.warning(
            "Multiple normalized matrix norms detected across bindings; "
            "persisting first value in manifest metadata."
        )

    # Resolve matrix value scale
    matrix_value_scale = float(scale_values[0])
    if not all(
        np.isclose(v, matrix_value_scale, rtol=1e-10, atol=1e-12)
        for v in scale_values
    ):
        logger.warning(
            "Multiple matrix value scales detected across bindings; "
            "persisting sparse pack with value_scale=1.0."
        )
        matrix_value_scale = 1.0

    # Resolve scale metadata
    unique_scale_payloads = {
        json.dumps(payload, sort_keys=True) if payload is not None else "null"
        for payload in metadata_values
    }
    scale_metadata: ScaleMetadata | None = None
    if len(unique_scale_payloads) == 1:
        scale_metadata = metadata_values[0]
    else:
        logger.warning(
            "Multiple scale metadata payloads detected across bindings; "
            "persisting empty scale metadata in manifest."
        )
        scale_metadata = None

    return matrix_norm_value, matrix_value_scale, scale_metadata


def _persist_dataset(
    dataset_dir: str,
    rhs_all: np.ndarray,
    solutions_all: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    nnz_ptr: np.ndarray,
    matrix_size: tuple[int, int],
    normalize: NormalizeType,
    matrix_norm_type: str,
    matrix_norm: float,
    matrix_value_scale: float,
    scale_metadata: ScaleMetadata | None,
    num_bindings: int,
) -> None:
    """Persist dataset to disk.

    Side effect: Writes files to dataset_dir with manifest and arrays.

    Args:
        dataset_dir: Output directory path
        rhs_all: Stacked RHS vectors
        solutions_all: Stacked solution vectors
        indices: Sparse matrix indices
        values: Sparse matrix values
        nnz_ptr: Sparse matrix row pointers
        matrix_size: Tuple of (rows, cols) for system matrix
        normalize: Normalization type applied
        matrix_norm_type: Type of norm computed
        matrix_norm: Final matrix norm value
        matrix_value_scale: Final value scale factor
        scale_metadata: Metadata about scaling
        num_bindings: Number of matrix-RHS bindings processed
    """
    from pathlib import Path
    from ..io.dataset_storage import save_dataset_from_sparse

    dataset_dir_path = Path(dataset_dir)
    dataset_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to: {dataset_dir}")

    save_dataset_from_sparse(
        dataset_dir=dataset_dir_path,
        rhs=rhs_all,
        solutions=solutions_all,
        indices=indices,
        values=values,
        nnz_ptr=nnz_ptr,
        size=matrix_size,
        normalization_type=str(normalize),
        matrix_norm=matrix_norm,
        matrix_norm_type=matrix_norm_type,
        matrix_value_scale=matrix_value_scale,
        scale_metadata=scale_metadata,  # type: ignore[arg-type]  # ScaleMetadata compatible
    )

    logger.info(
        f"Dataset built successfully: {dataset_dir} "
        f"(samples={rhs_all.shape[0]}, matrix_samples={num_bindings})"
    )


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
    """Build dataset from streamed matrix sources without dense N-matrix materialization.

    Orchestrates dataset generation pipeline by coordinating stream opening,
    matrix normalization, sample generation, and persistent storage.

    Args:
        matrix_path: Path expression for matrix stream
        dataset_dir: Output directory for dataset
        counts: Sample counts per strategy
        mix: Mixing proportions
        total: Total sample count
        rhs_path: Optional path expression for RHS stream
        solutions_path: Optional path expression for solutions
        sample_id_regex: Optional regex for sample ID extraction
        normalize: Normalization type to apply
        matrix_norm_type: Type of norm to compute
        shuffle: Whether to shuffle samples
        seed: Random seed for reproducibility
        strategy_overrides: Strategy-specific configuration overrides

    Returns:
        Path to generated dataset directory
    """
    from ..io.dataset_storage import SparsePackAccumulator

    logger.info("Building dataset...")
    logger.info(f"  Matrix: {matrix_path}")
    logger.info(f"  Output: {dataset_dir}")
    if solutions_path is not None:
        logger.info(f"  Solutions source: {solutions_path}")
    if rhs_path is not None:
        logger.info(f"  RHS source: {rhs_path}")

    # Open streams and bind sources
    matrix_stream, rhs_stream, bindings = _open_streams(
        matrix_path, rhs_path, sample_id_regex
    )
    logger.info(
        f"  Matrix samples: {len(matrix_stream.sample_ids)} | System bindings: {len(bindings)}"
    )

    # Setup accumulators
    sparse_acc = SparsePackAccumulator()
    single_matrix_mode = len(matrix_stream.sample_ids) == 1
    single_matrix_written = False
    rhs_blocks: list[np.ndarray] = []
    solution_blocks: list[np.ndarray] = []
    matrix_norm_values: list[float] = []
    matrix_value_scale_values: list[float] = []
    scale_metadata_values: list[ScaleMetadata | None] = []
    matrix_cache: dict[int, _CachedMatrix] = {}

    # Process each binding
    logger.info(f"  Normalization: {normalize}")
    for binding in bindings:
        result = _process_binding(
            binding,
            matrix_stream,
            rhs_stream,
            matrix_cache,
            normalize,
            matrix_norm_type,
            counts,
            mix,
            total,
            seed,
            shuffle,
            strategy_overrides,
        )

        # Skip empty results
        if result.rhs_block.shape[0] == 0:
            continue

        # Accumulate blocks and metadata
        rhs_blocks.append(result.rhs_block)
        solution_blocks.append(result.solution_block)

        # Cache matrix in appropriate format and accumulate
        cached = matrix_cache.get(binding.matrix_sample_id)
        if cached is not None:
            if single_matrix_mode:
                if not single_matrix_written:
                    sparse_acc.append_dense_matrix(cached.matrix_norm, repeats=1)
                    single_matrix_written = True
            else:
                sparse_acc.append_dense_matrix(cached.matrix_norm, repeats=int(result.rhs_block.shape[0]))

        matrix_norm_values.append(result.matrix_norm_value)
        matrix_value_scale_values.append(result.matrix_value_scale)
        scale_metadata_values.append(result.scale_params)

    if not rhs_blocks or not solution_blocks:
        raise ValueError("No samples were generated for dataset persistence.")

    # Build sparse arrays
    rhs_all = np.vstack(rhs_blocks)
    solutions_all = np.vstack(solution_blocks)
    indices, values, nnz_ptr, matrix_size = sparse_acc.build_arrays()

    # Resolve final scaling parameters
    matrix_norm_value, matrix_value_scale, scale_metadata = _resolve_final_scale(
        matrix_norm_values, matrix_value_scale_values, scale_metadata_values
    )

    # Persist dataset
    _persist_dataset(
        dataset_dir,
        rhs_all,
        solutions_all,
        indices,
        values,
        nnz_ptr,
        matrix_size,
        normalize,
        matrix_norm_type,
        matrix_norm_value,
        matrix_value_scale,
        scale_metadata,
        len(bindings),
    )

    return dataset_dir


__all__ = [
    "_shuffle_samples",
    "generate_mixture",
    "build_dataset",
]
