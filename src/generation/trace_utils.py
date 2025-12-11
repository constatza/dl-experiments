"""Trace data manipulation utilities for residual and error traces."""

from __future__ import annotations

import numpy as np

from ..normalization import ErrorTraceSamples, ResidualTraceSamples


def _offset_residual_traces(
    traces: ResidualTraceSamples,
    offset: int,
) -> ResidualTraceSamples:
    """Offset sample indices in residual trace samples.

    Args:
        traces: Residual trace samples
        offset: Offset to add to sample indices

    Returns:
        New ResidualTraceSamples with offset sample indices
    """
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
    """Merge multiple residual trace blocks into single ResidualTraceSamples.

    Args:
        blocks: List of residual trace blocks to merge

    Returns:
        Merged ResidualTraceSamples

    Raises:
        ValueError: If blocks inconsistently include optional fields
    """
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
        search_directions = np.vstack([
            block.search_directions for block in blocks
            if block.search_directions is not None
        ])
    if all(has_search_direction_products):
        # All blocks have search_direction_products (validated above), filter None for type safety
        search_direction_products = np.vstack([
            block.search_direction_products for block in blocks
            if block.search_direction_products is not None
        ])
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
    """Offset sample indices in error trace samples.

    Args:
        traces: Error trace samples
        offset: Offset to add to sample indices

    Returns:
        New ErrorTraceSamples with offset sample indices
    """
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
    """Merge multiple error trace blocks into single ErrorTraceSamples.

    Args:
        blocks: List of error trace blocks to merge

    Returns:
        Merged ErrorTraceSamples
    """
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


__all__ = [
    "_offset_residual_traces",
    "_merge_residual_traces",
    "_offset_error_traces",
    "_merge_error_traces",
]
