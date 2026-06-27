"""Resolve safe dataset-backed or generated RHS inputs for comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from neuralls.composition.comparison.models import ResolvedComparisonInput
from neuralls.composition.comparison.rhs_generation import (
    generate_gaussian_rhs,
    generate_sparse_rhs,
)
from neuralls.platform.config.models.comparison import (
    GaussianRhsGenerationModel,
    SparseRhsGenerationModel,
)
from neuralls.platform.storage.comparison import load_system_arrays
from neuralls.platform.storage.dataset_readers import (
    list_available_matrix_indices,
    load_dense_training_arrays,
    load_optional_rhs_kind_codes,
)
from neuralls.shared.enum_codecs import decode_rhs_kind_array
from neuralls.shared.types import ComparisonRhsGenerationKind, RhsKind


@dataclass(frozen=True)
class _AllowedRhsResolution:
    """Comparison-local RHS candidate set and semantic enforcement status."""

    allowed_indices: tuple[int, ...]
    rhs_kinds: tuple[RhsKind, ...] | None
    metadata_enforced: bool


def _normalize_seed(seed: object) -> int | None:
    """Coerce optional seed-like values coming from config or test doubles."""
    return seed if isinstance(seed, int) else None


def _normalize_rhs_generation_kind(
    kind: object,
) -> ComparisonRhsGenerationKind | None:
    """Coerce optional generator kind values coming from config or test doubles."""
    if isinstance(kind, ComparisonRhsGenerationKind):
        return kind
    if isinstance(kind, str):
        try:
            return ComparisonRhsGenerationKind(kind)
        except ValueError:
            return None
    return None


def _resolve_allowed_rhs_rows(
    rhs_path: Path,
    *,
    require_non_residual_rhs: bool,
) -> _AllowedRhsResolution:
    """Resolve allowed dataset RHS rows with optional semantic metadata enforcement."""
    codes = load_optional_rhs_kind_codes(rhs_path)
    if codes is None:
        rhs, _solutions = load_dense_training_arrays(rhs_path)
        sample_count = int(rhs.shape[0]) if rhs.ndim > 1 else 1
        if require_non_residual_rhs:
            logger.warning(
                "Comparison dataset '{}' does not expose rhs_kind metadata; "
                "non-residual RHS enforcement cannot be verified.",
                rhs_path,
            )
        return _AllowedRhsResolution(
            allowed_indices=tuple(range(sample_count)),
            rhs_kinds=None,
            metadata_enforced=False,
        )

    rhs_kinds = decode_rhs_kind_array(codes)
    if not require_non_residual_rhs:
        return _AllowedRhsResolution(
            allowed_indices=tuple(range(len(rhs_kinds))),
            rhs_kinds=rhs_kinds,
            metadata_enforced=True,
        )
    return _AllowedRhsResolution(
        allowed_indices=tuple(
            index for index, kind in enumerate(rhs_kinds) if kind is RhsKind.NON_RESIDUAL
        ),
        rhs_kinds=rhs_kinds,
        metadata_enforced=True,
    )


def resolve_comparison_input(
    *,
    matrix_path: Path,
    matrix_dataset_id: str,
    matrix_index: int | None,
    rhs_path: Path | None,
    rhs_dataset_id: str | None,
    rhs_index: int | None,
    require_non_residual_rhs: bool,
    seed: int | None,
    rhs_generation_kind: ComparisonRhsGenerationKind | None,
    rhs_generation_params: dict[str, object] | None,
) -> ResolvedComparisonInput:
    """Resolve the final comparison system and provenance before solver execution."""
    resolved_seed = _normalize_seed(seed)
    rng = np.random.default_rng(resolved_seed)
    resolved_generation_kind = _normalize_rhs_generation_kind(rhs_generation_kind)
    if not matrix_path.is_dir():
        resolved_matrix_index = matrix_index if matrix_index is not None else 0
        if resolved_generation_kind is not None:
            matrix, _dummy_rhs = load_system_arrays(
                matrix_path=matrix_path,
                rhs_path=matrix_path,
                rhs_index=0,
                matrix_index=resolved_matrix_index,
            )
            match resolved_generation_kind:
                case ComparisonRhsGenerationKind.GAUSSIAN:
                    cfg = GaussianRhsGenerationModel.model_validate(
                        {"kind": resolved_generation_kind, **(rhs_generation_params or {})}
                    )
                    rhs = generate_gaussian_rhs(matrix.shape[0], cfg, resolved_seed)
                case ComparisonRhsGenerationKind.SPARSE:
                    cfg = SparseRhsGenerationModel.model_validate(
                        {"kind": resolved_generation_kind, **(rhs_generation_params or {})}
                    )
                    rhs = generate_sparse_rhs(matrix.shape[0], cfg)
                case _:
                    raise ValueError(
                        f"Unsupported comparison RHS generator: {resolved_generation_kind}."
                    )
            return ResolvedComparisonInput(
                matrix=matrix,
                rhs=rhs,
                matrix_dataset_id=matrix_dataset_id,
                matrix_index=resolved_matrix_index,
                rhs_source_type="generated",
                generator_kind=resolved_generation_kind,
                generator_params=rhs_generation_params,
            )
        if rhs_path is None:
            raise ValueError("Dataset-backed comparison RHS requires rhs_path and rhs_dataset_id.")
        resolved_rhs_index = rhs_index if rhs_index is not None else 0
        matrix, rhs = load_system_arrays(
            matrix_path=matrix_path,
            rhs_path=rhs_path,
            rhs_index=resolved_rhs_index,
            matrix_index=resolved_matrix_index,
        )
        return ResolvedComparisonInput(
            matrix=matrix,
            rhs=rhs,
            matrix_dataset_id=matrix_dataset_id,
            matrix_index=resolved_matrix_index,
            rhs_source_type="dataset",
            rhs_dataset_id=rhs_dataset_id,
            rhs_index=resolved_rhs_index,
        )

    matrix_indices = list_available_matrix_indices(matrix_path)
    if not matrix_indices:
        raise ValueError(f"No matrix samples are available in comparison dataset '{matrix_path}'.")
    resolved_matrix_index = (
        matrix_index if matrix_index is not None else int(rng.choice(np.asarray(matrix_indices)))
    )
    if resolved_matrix_index not in matrix_indices:
        raise ValueError(
            f"matrix_index={resolved_matrix_index} is not valid for comparison dataset '{matrix_path}'."
        )

    if resolved_generation_kind is not None:
        matrix, _dummy_rhs = load_system_arrays(
            matrix_path=matrix_path,
            rhs_path=matrix_path,
            rhs_index=0,
            matrix_index=resolved_matrix_index,
        )
        match resolved_generation_kind:
            case ComparisonRhsGenerationKind.GAUSSIAN:
                cfg = GaussianRhsGenerationModel.model_validate(
                    {"kind": resolved_generation_kind, **(rhs_generation_params or {})}
                )
                rhs = generate_gaussian_rhs(matrix.shape[0], cfg, resolved_seed)
            case ComparisonRhsGenerationKind.SPARSE:
                cfg = SparseRhsGenerationModel.model_validate(
                    {"kind": resolved_generation_kind, **(rhs_generation_params or {})}
                )
                rhs = generate_sparse_rhs(matrix.shape[0], cfg)
            case _:
                raise ValueError(
                    f"Unsupported comparison RHS generator: {resolved_generation_kind}."
                )
        return ResolvedComparisonInput(
            matrix=matrix,
            rhs=rhs,
            matrix_dataset_id=matrix_dataset_id,
            matrix_index=resolved_matrix_index,
            rhs_source_type="generated",
            generator_kind=resolved_generation_kind,
            generator_params=rhs_generation_params,
        )

    if rhs_path is None or rhs_dataset_id is None:
        raise ValueError("Dataset-backed comparison RHS requires rhs_path and rhs_dataset_id.")
    allowed_rhs = _resolve_allowed_rhs_rows(
        rhs_path, require_non_residual_rhs=require_non_residual_rhs
    )
    safe_rhs_indices = allowed_rhs.allowed_indices
    if not safe_rhs_indices:
        raise ValueError(f"No safe RHS rows are available in comparison dataset '{rhs_path}'.")
    resolved_rhs_index = (
        rhs_index if rhs_index is not None else int(rng.choice(np.asarray(safe_rhs_indices)))
    )
    if resolved_rhs_index not in safe_rhs_indices:
        raise ValueError(
            f"rhs_index={resolved_rhs_index} is not allowed by the active non-residual RHS policy."
        )
    matrix, rhs = load_system_arrays(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        rhs_index=resolved_rhs_index,
        matrix_index=resolved_matrix_index,
    )
    return ResolvedComparisonInput(
        matrix=matrix,
        rhs=rhs,
        matrix_dataset_id=matrix_dataset_id,
        matrix_index=resolved_matrix_index,
        rhs_source_type="dataset",
        rhs_dataset_id=rhs_dataset_id,
        rhs_index=resolved_rhs_index,
        rhs_kind=(
            allowed_rhs.rhs_kinds[resolved_rhs_index] if allowed_rhs.rhs_kinds is not None else None
        ),
    )
