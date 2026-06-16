"""Orchestration functions for mixed-strategy data generation."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from .data_types import NormalizeType
from neuralls.shared.types import ScaleMetadata
from neuralls.domain.normalization import ErrorTraceSamples, IScale, ResidualTraceSamples
from .helpers import rng_from_seed, _resolve_strategy_counts, _merge_strategy_outputs
from .helpers import serialize_scale_metadata
from .trace_utils import (
    _offset_error_traces,
    _offset_residual_traces,
    _merge_residual_traces,
    _merge_error_traces,
)
from .interfaces import ArchiveData, TracingSolverCallable
from .payloads import GeneratedDatasetPayload
from .ports import ZarrAccumulatorPort
from .runner import strategy_supports_matrix_replacement
from .source_streams import (
    EnumerateBy,
    MatrixSampleStream,
    SystemBinding,
    VectorSampleStream,
    bind_sources,
    open_matrix_stream,
    open_vector_stream,
)


@dataclass(frozen=True)
class _StrategyProperties:
    """Compile-time properties for a known generation strategy."""

    uses_finite_source: bool
    supports_replacement: bool


_STRATEGY_PROPERTIES: dict[str, _StrategyProperties] = {
    "solution_archive": _StrategyProperties(uses_finite_source=True, supports_replacement=False),
    "rhs_archive": _StrategyProperties(uses_finite_source=True, supports_replacement=False),
    "scaled_solutions": _StrategyProperties(uses_finite_source=True, supports_replacement=False),
    "validated_archive": _StrategyProperties(uses_finite_source=True, supports_replacement=False),
    "residual_traces": _StrategyProperties(uses_finite_source=True, supports_replacement=True),
    "residuals": _StrategyProperties(uses_finite_source=True, supports_replacement=True),
    "gaussian_residuals": _StrategyProperties(uses_finite_source=True, supports_replacement=True),
    "search_directions": _StrategyProperties(uses_finite_source=True, supports_replacement=True),
}


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

    if error_traces is not None:
        error_traces = ErrorTraceSamples(
            residuals=error_traces.residuals,
            solutions_current=error_traces.solutions_current,
            errors=error_traces.errors,
            true_solutions=error_traces.true_solutions[indices],
            sample_indices=inverse[error_traces.sample_indices],
            iteration_indices=error_traces.iteration_indices,
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
    solver_overrides: dict[str, TracingSolverCallable] | None = None,
    archive_solutions: np.ndarray | None = None,
    archive_rhs: np.ndarray | None = None,
    single_rhs: np.ndarray | None = None,
    single_solution: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, ResidualTraceSamples | None, ErrorTraceSamples | None]:
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
        single_solution: Optional single pre-computed solution vector, shape (n,). If provided
            and no archive_solutions are present, it is used as a single-row archive for
            archive-based strategies.

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
        ...     },
        ... )
        >>> X.shape
        (100, n)

        >>> # Generate explicit counts with strategy-specific configuration
        >>> X, Y, _, _ = generate_mixture(
        ...     A,
        ...     counts={"normal": 50, "krylov": 30, "residual_traces": 20},
        ...     seed=42,
        ...     strategy_overrides={
        ...         "residual_traces": {"cg_iters": 10},
        ...     },
        ... )

        >>> # Generate with single RHS for trace strategies
        >>> rhs = np.random.randn(n)
        >>> X, Y, res_traces, _ = generate_mixture(
        ...     A,
        ...     counts={"residual_traces": 20},
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

        effective_archive_solutions = archive_solutions
        if single_solution is not None and archive_solutions is None:
            effective_archive_solutions = single_solution.reshape(1, -1)

        archive_data: ArchiveData | None = None
        if effective_archive_solutions is not None:
            archive_data = ArchiveData(
                solutions=effective_archive_solutions, rhs_vectors=archive_rhs
            )

        cfg["samples"] = count

        # Run generation (all strategies now use unified interface)
        # Single-RHS strategies (trace strategies) will receive single_rhs if provided
        # Pydantic (extra="forbid") will raise ValidationError on unknown keys — fail fast.
        try:
            generated = run_generation(
                strategy_name,
                A,
                cfg=cfg,
                solver=solver_overrides.get(strategy_name) if solver_overrides else None,
                archive=archive_data,
                single_rhs=single_rhs,
            )
        except ValidationError as e:
            raise ValueError(f"Invalid configuration for strategy '{strategy_name}': {e}") from e

        if generated.rhs is not None:
            all_features.append(generated.rhs)
        if generated.solutions is not None:
            all_targets.append(generated.solutions)

        if generated.residual_traces is not None:
            residual_blocks.append(
                _offset_residual_traces(generated.residual_traces, sample_offset)
            )
        if generated.error_traces is not None:
            error_blocks.append(_offset_error_traces(generated.error_traces, sample_offset))

        if generated.rhs is not None:
            sample_offset += generated.rhs.shape[0]

    if all_features and all_targets:
        X, Y = _merge_strategy_outputs(all_features, all_targets)
    else:
        X = np.empty((0, A.shape[0]), dtype=np.float64)
        Y = np.empty((0, A.shape[0]), dtype=np.float64)
    residual_traces = _merge_residual_traces(residual_blocks) if residual_blocks else None
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
        scale: IScale object or None (scaling strategy applied)
        matrix_norm_value: Computed matrix norm value
        matrix_value_scale: Scaling factor applied
        scale_params: Dictionary of scale parameters or None
    """

    matrix_norm: np.ndarray
    scale: IScale | None
    matrix_norm_value: float
    matrix_value_scale: float
    scale_params: ScaleMetadata | None


def _open_streams(
    matrix_path: str,
    rhs_path: str | None,
    sample_id_regex: str | None,
    enumerate_by: EnumerateBy | None,
    parameters_paths: tuple[str, ...] = (),
    solution_path: str | None = None,
) -> tuple[
    MatrixSampleStream,
    VectorSampleStream | None,
    VectorSampleStream | None,
    list[VectorSampleStream],
    list[SystemBinding],
]:
    """Open matrix, optional RHS, optional solution, and optional parameter streams and bind them.

    Args:
        matrix_path: Path expression for matrix stream
        rhs_path: Optional path expression for RHS stream
        sample_id_regex: Optional regex for sample ID extraction
        enumerate_by: Optional sequential ID assignment strategy for glob sources
        parameters_paths: Tuple of path expressions for parameter vector streams
        solution_path: Optional path expression for pre-computed solution stream

    Returns:
        Tuple of (matrix_stream, rhs_stream, solution_stream, param_streams, bindings)
    """
    matrix_stream = open_matrix_stream(
        matrix_path_expr=matrix_path,
        sample_id_regex=sample_id_regex,
        enumerate_by=enumerate_by,
    )
    rhs_stream = (
        open_vector_stream(
            rhs_path,
            sample_id_regex=sample_id_regex,
            enumerate_by=enumerate_by,
        )
        if rhs_path is not None
        else None
    )
    solution_stream = (
        open_vector_stream(
            solution_path,
            sample_id_regex=sample_id_regex,
            enumerate_by=enumerate_by,
        )
        if solution_path is not None
        else None
    )
    param_streams = [
        open_vector_stream(p, sample_id_regex=sample_id_regex, enumerate_by=enumerate_by)
        for p in parameters_paths
    ]
    bindings = bind_sources(
        matrix_ids=matrix_stream.sample_ids,
        rhs_ids=rhs_stream.sample_ids if rhs_stream is not None else None,
        solution_ids=solution_stream.sample_ids if solution_stream is not None else None,
        parameters_ids_list=tuple(s.sample_ids for s in param_streams),
    )
    return matrix_stream, rhs_stream, solution_stream, param_streams, bindings


def _strategy_uses_finite_source(
    strategy_name: str,
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None,
    has_rhs_source: bool,
) -> bool:
    """Return whether a strategy is configured to use a finite external source."""
    props = _STRATEGY_PROPERTIES.get(strategy_name)
    if props is not None:
        return props.uses_finite_source
    # Fallback heuristic for unknown/future strategies
    overrides = strategy_overrides.get(strategy_name, {}) if strategy_overrides is not None else {}
    return has_rhs_source or isinstance(overrides.get("solutions_glob"), str)


def _validate_replacement_support(
    strategy_counts: Mapping[str, int],
    *,
    replacement: bool,
    num_matrix_samples: int,
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None,
    has_rhs_source: bool,
) -> None:
    """Fail fast when replacement is requested for unsupported strategy allocations."""
    if not replacement or num_matrix_samples <= 1:
        return

    for strategy_name, count in strategy_counts.items():
        if count == 0:
            continue
        props = _STRATEGY_PROPERTIES.get(strategy_name)
        supports_replacement = (
            props.supports_replacement
            if props is not None
            else strategy_supports_matrix_replacement(strategy_name)
        )
        if not supports_replacement:
            raise ValueError(
                f"Strategy '{strategy_name}' does not support matrix replacement allocation."
            )
        if _strategy_uses_finite_source(strategy_name, strategy_overrides, has_rhs_source):
            raise ValueError(
                f"Strategy '{strategy_name}' does not support matrix replacement when configured "
                "with a finite external source."
            )


def _append_binding_count(
    counts_by_binding: list[dict[str, int]],
    binding_idx: int,
    strategy_name: str,
    count: int,
) -> None:
    """Accumulate one per-binding strategy count."""
    if count == -1:
        counts_by_binding[binding_idx][strategy_name] = -1
        return
    if count <= 0:
        return
    counts_by_binding[binding_idx][strategy_name] = (
        counts_by_binding[binding_idx].get(strategy_name, 0) + count
    )


def _allocate_strategy_counts_across_bindings(
    *,
    count: int,
    bindings: Sequence[SystemBinding],
    replacement: bool,
    rng: np.random.Generator,
) -> list[int]:
    """Distribute one strategy's global row budget across bindings."""
    num_bindings = len(bindings)
    if num_bindings < 1:
        raise ValueError("At least one binding is required for allocation.")
    if count == -1:
        return [-1] * num_bindings
    if count <= 0:
        return [0] * num_bindings

    if not replacement:
        base = count // num_bindings
        remainder = count % num_bindings
        return [base + (1 if idx < remainder else 0) for idx in range(num_bindings)]

    allocations = [0] * num_bindings
    sampled_indices = rng.integers(0, num_bindings, size=count)
    for binding_idx in sampled_indices.tolist():
        allocations[binding_idx] += 1
    return allocations


def _resolve_binding_strategy_counts(
    *,
    bindings: Sequence[SystemBinding],
    counts: Mapping[str, int] | None,
    mix: Mapping[str, float] | None,
    total: int | None,
    replacement: bool,
    seed: int,
    strategy_overrides: Mapping[str, Mapping[str, Any]] | None,
    has_rhs_source: bool,
    num_matrix_samples: int,
) -> list[dict[str, int]]:
    """Resolve global strategy counts into per-binding count maps."""
    strategy_counts = _resolve_strategy_counts(counts, mix, total)
    _validate_replacement_support(
        strategy_counts,
        replacement=replacement,
        num_matrix_samples=num_matrix_samples,
        strategy_overrides=strategy_overrides,
        has_rhs_source=has_rhs_source,
    )

    if num_matrix_samples <= 1:
        return [dict(strategy_counts) for _ in bindings]

    rng = rng_from_seed(seed)
    counts_by_binding = [{} for _ in bindings]
    for strategy_name, count in strategy_counts.items():
        allocations = _allocate_strategy_counts_across_bindings(
            count=count,
            bindings=bindings,
            replacement=replacement,
            rng=rng,
        )
        for binding_idx, allocated_count in enumerate(allocations):
            _append_binding_count(counts_by_binding, binding_idx, strategy_name, allocated_count)
    return counts_by_binding


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
    binding: SystemBinding,
    rhs_stream: VectorSampleStream | None,
    get_matrix: Callable[[int], _CachedMatrix],
    counts: dict[str, int] | None,
    mix: dict[str, float] | None,
    total: int | None,
    seed: int,
    shuffle: bool,
    strategy_overrides: dict[str, dict[str, Any]] | None,
    solver_overrides: dict[str, TracingSolverCallable] | None = None,
    solution_stream: VectorSampleStream | None = None,
) -> _BindingResult:
    """Process one matrix-RHS binding and generate samples.

    Args:
        binding: Single binding to process
        rhs_stream: Optional opened RHS stream
        get_matrix: Callable ``(sample_id: int) -> _CachedMatrix``; caches internally
        counts: Sample counts per strategy
        mix: Mixing proportions
        total: Total samples
        seed: Random seed
        shuffle: Whether to shuffle samples
        strategy_overrides: Strategy-specific overrides
        solver_overrides: Optional per-strategy solver overrides
        solution_stream: Optional opened solution stream for per-binding pre-computed solutions

    Returns:
        BindingResult with generated data blocks
    """
    cached = get_matrix(binding.matrix_sample_id)

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

    # Load optional per-binding solution
    single_solution: np.ndarray | None = None
    if solution_stream is not None and binding.solution_sample_id is not None:
        sol_sample = solution_stream.load_sample(binding.solution_sample_id)
        single_solution = np.asarray(sol_sample.vector, dtype=np.float64)

    # Generate mixture
    logger.info(
        f"Generating/loading samples for binding sample_id={binding.sample_id} "
        f"(matrix_id={binding.matrix_sample_id})..."
    )
    X, Y, residual_traces, error_traces = generate_mixture(
        cached.matrix_norm,
        counts=counts,
        mix=mix,
        total=total,
        seed=seed,
        shuffle=shuffle,
        strategy_overrides=strategy_overrides,
        solver_overrides=solver_overrides,
        single_rhs=single_rhs,
        single_solution=single_solution,
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
        scale_params=cached.scale_params,
    )


def _resolve_final_scale(
    norm_values: list[float],
    scale_values: list[float],
    metadata_values: list[ScaleMetadata | None],
) -> tuple[float, float, ScaleMetadata | None]:
    """Resolve manifest-level normalization metadata from all bindings.

    Matrix samples are normalized per binding before they are written. The
    manifest, however, exposes only one dataset-level normalization block.
    When bindings disagree on scale details, the manifest omits ambiguous scale
    metadata instead of inventing a shared reversible scale.

    Args:
        norm_values: Matrix norm values from each binding.
        scale_values: Matrix value scale factors from each binding.
        metadata_values: Scale metadata from each binding.

    Returns:
        Tuple of ``(final_matrix_norm, final_matrix_scale, final_scale_metadata)``.
        The returned ``final_scale_metadata`` is ``None`` when a multi-binding
        dataset does not share one exact scale payload.
    """
    if not norm_values or not scale_values:
        raise ValueError("No norm or scale values to resolve")

    # Resolve matrix norm
    matrix_norm_value = float(norm_values[0])
    if not all(np.isclose(v, matrix_norm_value, rtol=1e-10, atol=1e-12) for v in norm_values):
        logger.warning(
            "Bindings produced different normalized matrix norms; "
            "the dataset manifest keeps a representative matrix_norm only."
        )

    # Resolve matrix value scale
    matrix_value_scale = float(scale_values[0])
    if not all(np.isclose(v, matrix_value_scale, rtol=1e-10, atol=1e-12) for v in scale_values):
        logger.warning(
            "Bindings produced different matrix value scales; "
            "the dataset manifest omits a shared reversible matrix scale."
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
            "Bindings produced different scale metadata payloads; "
            "the dataset manifest intentionally stores no shared scale metadata."
        )
        scale_metadata = None

    return matrix_norm_value, matrix_value_scale, scale_metadata


def _build_dataset_payload(
    rhs_all: np.ndarray,
    solutions_all: np.ndarray,
    matrix_zarr_path: Path,
    matrix_size: tuple[int, int],
    normalize: NormalizeType,
    matrix_norm_type: str,
    matrix_norm: float,
    matrix_value_scale: float,
    scale_metadata: ScaleMetadata | None,
    num_bindings: int,
    parameters_arrays: tuple[np.ndarray, ...] = (),
) -> GeneratedDatasetPayload:
    """Build the final immutable dataset payload for the composition layer."""
    return GeneratedDatasetPayload(
        rhs=rhs_all,
        solutions=solutions_all,
        matrix_zarr_path=matrix_zarr_path,
        matrix_size=matrix_size,
        normalization_type=str(normalize),
        matrix_norm=matrix_norm,
        matrix_norm_type=matrix_norm_type,
        matrix_value_scale=matrix_value_scale,
        scale_metadata=scale_metadata,
        num_bindings=num_bindings,
        parameters_arrays=parameters_arrays,
    )


def _prepare_generation_context(
    *,
    matrix_path: str,
    rhs_path: str | None,
    solution_path: str | None,
    parameters_paths: tuple[str, ...],
    sample_id_regex: str | None,
    enumerate_by: EnumerateBy | None,
    counts: dict[str, int] | None,
    mix: dict[str, float] | None,
    total: int | None,
    replacement: bool,
    seed: int,
    strategy_overrides: dict[str, dict[str, Any]] | None,
) -> tuple[
    MatrixSampleStream,
    VectorSampleStream | None,
    VectorSampleStream | None,
    list[VectorSampleStream],
    list[SystemBinding],
    list[dict[str, int]],
]:
    """Open all source streams, bind them, and resolve per-binding strategy counts.

    Args:
        matrix_path: Path expression for matrix stream
        rhs_path: Optional path expression for RHS stream
        solution_path: Optional path expression for per-binding solution stream
        parameters_paths: Tuple of path expressions for parameter streams
        sample_id_regex: Optional regex for sample ID extraction
        enumerate_by: Optional sequential ID assignment strategy
        counts: Sample counts per strategy
        mix: Mixing proportions
        total: Total sample count
        replacement: Whether to allow matrix replacement allocation
        seed: Random seed for reproducibility
        strategy_overrides: Per-strategy configuration overrides

    Returns:
        Tuple of (matrix_stream, rhs_stream, solution_stream, param_streams, bindings, binding_counts)
    """
    matrix_stream, rhs_stream, solution_stream, param_streams, bindings = _open_streams(
        matrix_path,
        rhs_path,
        sample_id_regex,
        enumerate_by,
        parameters_paths,
        solution_path,
    )
    binding_counts = _resolve_binding_strategy_counts(
        bindings=bindings,
        counts=counts,
        mix=mix,
        total=total,
        replacement=replacement,
        seed=seed,
        strategy_overrides=strategy_overrides,
        has_rhs_source=rhs_stream is not None,
        num_matrix_samples=len(matrix_stream.sample_ids),
    )
    return matrix_stream, rhs_stream, solution_stream, param_streams, bindings, binding_counts


def _accumulate_bindings(
    *,
    bindings: list[SystemBinding],
    binding_counts: list[dict[str, int]],
    rhs_stream: VectorSampleStream | None,
    solution_stream: VectorSampleStream | None,
    param_streams: list[VectorSampleStream],
    matrix_stream: MatrixSampleStream,
    get_matrix: Callable[[int], _CachedMatrix],
    accumulator: ZarrAccumulatorPort,
    seed: int,
    shuffle: bool,
    strategy_overrides: dict[str, dict[str, Any]] | None,
    solver_overrides: dict[str, TracingSolverCallable] | None,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[list[np.ndarray]],
    list[float],
    list[float],
    list[ScaleMetadata | None],
    int,
]:
    """Process all bindings and accumulate rhs/solution blocks, param blocks, and scale metadata.

    Args:
        bindings: Ordered list of system bindings to process
        binding_counts: Per-binding strategy count maps (parallel to bindings)
        rhs_stream: Optional opened RHS stream
        solution_stream: Optional opened solution stream
        param_streams: Opened parameter vector streams
        matrix_stream: Opened matrix stream (used to detect single-matrix mode)
        get_matrix: Callable that loads and caches a normalized matrix by sample ID
        accumulator: Zarr accumulator for writing matrix samples
        seed: Random seed
        shuffle: Whether to shuffle samples within each binding
        strategy_overrides: Per-strategy configuration overrides
        solver_overrides: Optional per-strategy solver overrides

    Returns:
        Tuple of (rhs_blocks, solution_blocks, param_blocks,
                  matrix_norm_values, matrix_value_scale_values, scale_metadata_values,
                  emitted_binding_count)
    """
    single_matrix_mode = len(matrix_stream.sample_ids) == 1
    single_matrix_written = False
    rhs_blocks: list[np.ndarray] = []
    solution_blocks: list[np.ndarray] = []
    param_blocks: list[list[np.ndarray]] = [[] for _ in param_streams]
    matrix_norm_values: list[float] = []
    matrix_value_scale_values: list[float] = []
    scale_metadata_values: list[ScaleMetadata | None] = []
    emitted_binding_count = 0

    for binding, binding_strategy_counts in zip(bindings, binding_counts, strict=True):
        if not binding_strategy_counts:
            continue
        result = _process_binding(
            binding,
            rhs_stream,
            get_matrix,
            binding_strategy_counts,
            None,
            None,
            seed,
            shuffle,
            strategy_overrides,
            solver_overrides=solver_overrides,
            solution_stream=solution_stream,
        )

        n_samples = result.rhs_block.shape[0]
        if n_samples == 0:
            continue

        rhs_blocks.append(result.rhs_block)
        solution_blocks.append(result.solution_block)
        emitted_binding_count += 1

        for k, (stream, sample_id) in enumerate(zip(param_streams, binding.parameters_sample_ids)):
            if sample_id is not None:
                vec = stream.load_sample(sample_id).vector
                param_blocks[k].append(np.tile(vec, (n_samples, 1)))

        cached = get_matrix(binding.matrix_sample_id)
        if single_matrix_mode:
            if not single_matrix_written:
                accumulator.append_dense_matrix(cached.matrix_norm, repeats=1)
                single_matrix_written = True
        else:
            accumulator.append_dense_matrix(cached.matrix_norm, repeats=int(n_samples))

        matrix_norm_values.append(result.matrix_norm_value)
        matrix_value_scale_values.append(result.matrix_value_scale)
        scale_metadata_values.append(result.scale_params)

    return (
        rhs_blocks,
        solution_blocks,
        param_blocks,
        matrix_norm_values,
        matrix_value_scale_values,
        scale_metadata_values,
        emitted_binding_count,
    )


def _finalize_payload(
    *,
    rhs_blocks: list[np.ndarray],
    solution_blocks: list[np.ndarray],
    param_blocks: list[list[np.ndarray]],
    matrix_norm_values: list[float],
    matrix_value_scale_values: list[float],
    scale_metadata_values: list[ScaleMetadata | None],
    accumulator: ZarrAccumulatorPort,
    normalize: NormalizeType,
    matrix_norm_type: str,
    emitted_binding_count: int,
) -> GeneratedDatasetPayload:
    """Stack arrays, resolve scale, finalize accumulator, and build the payload.

    Args:
        rhs_blocks: RHS array blocks from each binding
        solution_blocks: Solution array blocks from each binding
        param_blocks: Parameter array blocks per stream per binding
        matrix_norm_values: Matrix norm values from each binding
        matrix_value_scale_values: Matrix value scale factors from each binding
        scale_metadata_values: Scale metadata from each binding
        accumulator: Zarr accumulator to finalize
        normalize: Normalization type applied
        matrix_norm_type: Type of norm used
        emitted_binding_count: Number of bindings that emitted samples

    Returns:
        Immutable GeneratedDatasetPayload ready for persistence.

    Raises:
        ValueError: If no samples were generated or accumulator is empty.
    """
    if not rhs_blocks or not solution_blocks:
        raise ValueError("No samples were generated for dataset persistence.")

    parameters_arrays: tuple[np.ndarray, ...] = tuple(
        np.vstack(blocks) for blocks in param_blocks if blocks
    )
    rhs_all = np.vstack(rhs_blocks)
    solutions_all = np.vstack(solution_blocks)
    matrix_zarr_path = accumulator.finalize()
    matrix_size = accumulator.matrix_size

    if matrix_size is None:
        raise ValueError("Accumulator is empty — no matrix samples were written.")

    matrix_norm_value, matrix_value_scale, scale_metadata = _resolve_final_scale(
        matrix_norm_values, matrix_value_scale_values, scale_metadata_values
    )
    return _build_dataset_payload(
        rhs_all,
        solutions_all,
        matrix_zarr_path,
        matrix_size,
        normalize,
        matrix_norm_type,
        matrix_norm_value,
        matrix_value_scale,
        scale_metadata,
        emitted_binding_count,
        parameters_arrays,
    )


def build_dataset_payload(
    matrix_path: str,
    *,
    counts: dict[str, int] | None = None,
    mix: dict[str, float] | None = None,
    total: int | None = None,
    rhs_path: str | None = None,
    solution_path: str | None = None,
    parameters_paths: tuple[str, ...] = (),
    sample_id_regex: str | None = None,
    enumerate_by: EnumerateBy | None = None,
    replacement: bool = False,
    normalize: NormalizeType = "matrix",
    matrix_norm_type: str = "spectral",
    shuffle: bool = True,
    seed: int = 42,
    strategy_overrides: dict[str, dict[str, Any]] | None = None,
    solver_overrides: dict[str, TracingSolverCallable] | None = None,
    accumulator: ZarrAccumulatorPort,
) -> GeneratedDatasetPayload:
    """Build an in-memory dataset payload from streamed matrix sources.

    Orchestrates dataset generation pipeline by coordinating stream opening,
    matrix normalization, sample generation, and sparse payload assembly.

    Args:
        matrix_path: Path expression for matrix stream
        counts: Sample counts per strategy
        mix: Mixing proportions
        total: Total sample count
        rhs_path: Optional path expression for RHS stream
        solution_path: Optional path expression for pre-computed solution stream; when
            provided, each binding loads the corresponding solution vector and passes it
            into generate_mixture as an archive seed.
        parameters_paths: Tuple of path expressions for parameter vector streams
        sample_id_regex: Optional regex for sample ID extraction
        enumerate_by: Optional sequential ID assignment strategy for glob sources
        replacement: Whether multi-matrix allocation may reuse matrix bindings for supported
            random strategies.
        normalize: Normalization type to apply
        matrix_norm_type: Type of norm to compute
        shuffle: Whether to shuffle samples
        seed: Random seed for reproducibility
        strategy_overrides: Strategy-specific configuration overrides
        solver_overrides: Optional per-strategy solver overrides
        accumulator: Zarr accumulator for writing matrix samples

    Returns:
        Immutable generated dataset payload ready for persistence.
    """
    logger.info("Building dataset...")
    logger.info(f"  Matrix: {matrix_path}")
    if rhs_path is not None:
        logger.info(f"  RHS source: {rhs_path}")
    if solution_path is not None:
        logger.info(f"  Solution stream: {solution_path}")
    for i, pp in enumerate(parameters_paths):
        logger.info(f"  Parameters stream [{i}]: {pp}")

    matrix_stream, rhs_stream, solution_stream, param_streams, bindings, binding_counts = (
        _prepare_generation_context(
            matrix_path=matrix_path,
            rhs_path=rhs_path,
            solution_path=solution_path,
            parameters_paths=parameters_paths,
            sample_id_regex=sample_id_regex,
            enumerate_by=enumerate_by,
            counts=counts,
            mix=mix,
            total=total,
            replacement=replacement,
            seed=seed,
            strategy_overrides=strategy_overrides,
        )
    )
    logger.info(
        f"  Matrix samples: {len(matrix_stream.sample_ids)} | System bindings: {len(bindings)}"
    )
    logger.info(f"  Normalization: {normalize}")

    @cache
    def _get_matrix(sample_id: int) -> _CachedMatrix:
        from neuralls.domain.linalg import calculate_matrix_norm
        from .helpers import _normalize_matrix_for_generation

        dense_sample = matrix_stream.load_dense_sample(sample_id)
        dense_matrix = np.asarray(dense_sample.matrix, dtype=np.float64)
        if dense_matrix.shape[0] != dense_matrix.shape[1]:
            raise ValueError(f"Matrix sample {sample_id} must be square, got {dense_matrix.shape}")
        matrix_norm, scale, matrix_value_scale = _normalize_matrix_for_generation(
            dense_matrix,
            normalize,
            spectral_radius_bound=None,
        )
        matrix_norm_value = calculate_matrix_norm(matrix_norm, norm_type=matrix_norm_type)
        scale_params = serialize_scale_metadata(scale)
        return _CachedMatrix(
            matrix_norm=matrix_norm,
            scale=scale,
            matrix_norm_value=matrix_norm_value,
            matrix_value_scale=matrix_value_scale,
            scale_params=scale_params,
        )

    rhs_blocks, solution_blocks, param_blocks, norm_vals, scale_vals, meta_vals, n_bindings = (
        _accumulate_bindings(
            bindings=bindings,
            binding_counts=binding_counts,
            rhs_stream=rhs_stream,
            solution_stream=solution_stream,
            param_streams=param_streams,
            matrix_stream=matrix_stream,
            get_matrix=_get_matrix,
            accumulator=accumulator,
            seed=seed,
            shuffle=shuffle,
            strategy_overrides=strategy_overrides,
            solver_overrides=solver_overrides,
        )
    )
    payload = _finalize_payload(
        rhs_blocks=rhs_blocks,
        solution_blocks=solution_blocks,
        param_blocks=param_blocks,
        matrix_norm_values=norm_vals,
        matrix_value_scale_values=scale_vals,
        scale_metadata_values=meta_vals,
        accumulator=accumulator,
        normalize=normalize,
        matrix_norm_type=matrix_norm_type,
        emitted_binding_count=n_bindings,
    )
    logger.info(
        "Dataset payload built successfully: "
        f"samples={payload.rhs.shape[0]}, matrix_samples={n_bindings}, "
        f"parameter_streams={len(payload.parameters_arrays)}"
    )
    return payload


__all__ = [
    "_shuffle_samples",
    "generate_mixture",
    "build_dataset_payload",
]
