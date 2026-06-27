"""Semantic classification helpers for persisted generation outputs."""

from __future__ import annotations

from neuralls.shared.types import GenerationStrategyKind, RhsKind, TargetKind


def classify_strategy_rhs_kind(kind: GenerationStrategyKind) -> RhsKind:
    """Return the persisted RHS semantic kind for one generation strategy."""
    if kind in {
        GenerationStrategyKind.RANDOM,
        GenerationStrategyKind.NORMAL,
        GenerationStrategyKind.KRYLOV,
        GenerationStrategyKind.RHS_ARCHIVE,
        GenerationStrategyKind.SOLUTION_ARCHIVE,
        GenerationStrategyKind.VALIDATED_ARCHIVE,
        GenerationStrategyKind.SCALED_SOLUTIONS,
        GenerationStrategyKind.SPARSE_RHS,
        GenerationStrategyKind.EIGENVECTOR_FORWARD,
        GenerationStrategyKind.EIGENVECTOR_INVERSE,
        GenerationStrategyKind.GAUSSIAN_FORWARD,
        GenerationStrategyKind.GAUSSIAN_INVERSE,
        GenerationStrategyKind.UNIFORM_FORWARD,
        GenerationStrategyKind.UNIFORM_INVERSE,
        GenerationStrategyKind.CONSTANT_FORWARD,
        GenerationStrategyKind.CONSTANT_INVERSE,
        GenerationStrategyKind.NEUTRAL_ONES,
    }:
        return RhsKind.NON_RESIDUAL
    if kind in {
        GenerationStrategyKind.RESIDUAL_TRACES,
        GenerationStrategyKind.RESIDUALS,
        GenerationStrategyKind.GAUSSIAN_RESIDUALS,
    }:
        return RhsKind.RESIDUAL
    if kind is GenerationStrategyKind.SEARCH_DIRECTIONS:
        return RhsKind.SEARCH_DIRECTION_PRODUCT
    return RhsKind.UNKNOWN


def classify_strategy_target_kind(kind: GenerationStrategyKind) -> TargetKind:
    """Return the persisted target semantic kind for one generation strategy."""
    if kind in {
        GenerationStrategyKind.RANDOM,
        GenerationStrategyKind.NORMAL,
        GenerationStrategyKind.KRYLOV,
        GenerationStrategyKind.RHS_ARCHIVE,
        GenerationStrategyKind.SOLUTION_ARCHIVE,
        GenerationStrategyKind.VALIDATED_ARCHIVE,
        GenerationStrategyKind.SCALED_SOLUTIONS,
        GenerationStrategyKind.SPARSE_RHS,
        GenerationStrategyKind.EIGENVECTOR_FORWARD,
        GenerationStrategyKind.EIGENVECTOR_INVERSE,
        GenerationStrategyKind.GAUSSIAN_FORWARD,
        GenerationStrategyKind.GAUSSIAN_INVERSE,
        GenerationStrategyKind.UNIFORM_FORWARD,
        GenerationStrategyKind.UNIFORM_INVERSE,
        GenerationStrategyKind.CONSTANT_FORWARD,
        GenerationStrategyKind.CONSTANT_INVERSE,
        GenerationStrategyKind.NEUTRAL_ONES,
    }:
        return TargetKind.SOLUTION
    if kind is GenerationStrategyKind.RESIDUAL_TRACES:
        return TargetKind.ITERATE
    if kind in {
        GenerationStrategyKind.RESIDUALS,
        GenerationStrategyKind.GAUSSIAN_RESIDUALS,
    }:
        return TargetKind.ERROR
    if kind is GenerationStrategyKind.SEARCH_DIRECTIONS:
        return TargetKind.SEARCH_DIRECTION
    return TargetKind.UNKNOWN
