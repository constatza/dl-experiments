"""Preconditioner factory for TOML workflow.

This module provides a simple factory function for creating preconditioners
from configuration objects. Used exclusively by the TOML workflow (compare.py).

For direct usage, just instantiate preconditioners directly:
    >>> precond = JacobiPreconditioner(matrix)

For TOML workflow:
    >>> config = load_config("solver.toml")
    >>> precond = create_preconditioner(matrix, config.preconditioner)

Design:
    - Simple factory function with isinstance checks and explicit mapping
    - No builder classes needed - just call preconditioner constructors
    - Supports dependency injection for neural preconditioners (testing)
    - Better type safety via isinstance checks (understood by mypy)
    - Explicit enum-to-class mapping for clarity and extensibility
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .implementations import (
    Identity,
    JacobiPreconditioner,
    ILUPreconditioner,
    IC0Preconditioner,
    ICholeskyPreconditioner,
    NeuralPreconditioner,
)
from .base import Preconditioner
from ...configuration.preconditioner import PreconditionerType

if TYPE_CHECKING:
    from neuralls.configuration.preconditioner import (
        PreconditionerConfig,
        IC0PreconditionerConfig,
        NeuralPreconditionerConfig,
    )
    from .ports import PredictorAdapter


def create_preconditioner(
    matrix: NDArray,
    config: PreconditionerConfig,
    adapter: PredictorAdapter | None = None,
) -> Preconditioner:
    """Create preconditioner from configuration.

    Uses isinstance checks for type narrowing and explicit mapping
    for better type safety and IDE support.

    Args:
        matrix: System matrix A
        config: Preconditioner configuration from TOML
        adapter: Optional adapter for neural preconditioner (DI for testing)

    Returns:
        Preconditioner instance

    Example:
        >>> # Load from TOML
        >>> config = load_comparison_config("comparison.toml")
        >>> precond = create_preconditioner(A, config.preconditioner)
        >>>
        >>> # Use it
        >>> z = precond.apply(residual)

    Raises:
        ValueError: If preconditioner type is not supported
    """
    # Special cases with additional config parameters
    from neuralls.configuration.preconditioner import (
        IC0PreconditionerConfig,
        NeuralPreconditionerConfig,
    )

    # Check if type is NEURAL but config is not NeuralPreconditionerConfig
    if config.type == PreconditionerType.NEURAL:
        if not isinstance(config, NeuralPreconditionerConfig):
            raise TypeError(
                f"Neural type requires NeuralPreconditionerConfig, got {type(config)}"
            )
        return NeuralPreconditioner(
            checkpoint_path=config.checkpoint_path,
            config_path=config.config_path,
            data_config_path=config.data_config_path,
            adapter=adapter,
        )

    # IC(0) with threshold parameter
    if config.type == PreconditionerType.IC0:
        if not isinstance(config, IC0PreconditionerConfig):
            raise TypeError(
                f"IC(0) type requires IC0PreconditionerConfig, got {type(config)}"
            )
        return IC0Preconditioner(matrix, threshold=config.threshold)

    # Standard cases: Use explicit mapping
    PRECONDITIONER_CLASSES: dict[PreconditionerType, type[Preconditioner]] = {
        PreconditionerType.IDENTITY: Identity,
        PreconditionerType.NONE: Identity,
        PreconditionerType.JACOBI: JacobiPreconditioner,
        PreconditionerType.ILU: ILUPreconditioner,
        PreconditionerType.ICHOLESKY: ICholeskyPreconditioner,
    }

    cls = PRECONDITIONER_CLASSES.get(config.type)
    if cls is None:
        raise ValueError(
            f"Unsupported preconditioner type: {config.type}. "
            f"Supported: {list(PRECONDITIONER_CLASSES.keys())}"
        )

    # Identity doesn't need matrix argument - explicit return for type safety
    if config.type in (PreconditionerType.IDENTITY, PreconditionerType.NONE):
        return Identity()
    else:
        return cls(matrix)  # type: ignore[call-arg]  # Matrix-based preconditioners


def create_scheduled_preconditioner(
    primary: Preconditioner,
    fallback: Preconditioner | None = None,
    limit_iters: int | None = None,
) -> Preconditioner:
    """Create a scheduled preconditioner if iteration limit is specified.

    Args:
        primary: Main preconditioner to apply
        fallback: Preconditioner to use after limit (defaults to Identity)
        limit_iters: Switch to fallback after N iterations (None = unlimited)

    Returns:
        ScheduledPreconditioner if limit_iters is specified, otherwise primary unchanged

    Example:
        >>> # Limit neural preconditioner to first 10 iterations
        >>> scheduled = create_scheduled_preconditioner(
        ...     primary=neural_precond,
        ...     fallback=jacobi_precond,
        ...     limit_iters=10,
        ... )
    """
    if limit_iters is None:
        return primary

    from .implementations.scheduled import ScheduledPreconditioner

    return ScheduledPreconditioner(
        primary=primary,
        fallback=fallback or Identity(),
        limit_iters=limit_iters,
    )
