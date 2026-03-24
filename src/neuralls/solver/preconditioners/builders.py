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

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
        ConcretePreconditionerConfig,
        IC0PreconditionerConfig,
        NeuralPreconditionerConfig,
    )
    from .ports import PredictorAdapter


@dataclass(frozen=True)
class PreconditionerScheduleConfig:
    """Scheduling parameters for preconditioner switching.

    Extracted from BasePreconditionerConfig for internal use.
    Separates scheduling concerns from preconditioner configuration.

    Attributes:
        limit_iters: Number of iterations to apply primary preconditioner.
                     -1 means unlimited (use primary for entire solve).
        fallback: Preconditioner type to switch to after limit is reached.
    """

    limit_iters: int = -1
    fallback: PreconditionerType = PreconditionerType.IDENTITY


def create_preconditioner(
    matrix: NDArray,
    config: ConcretePreconditionerConfig,
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
        ckpt = config.resolved_checkpoint_path or config.checkpoint_path
        if ckpt is None:
            raise ValueError("NeuralPreconditionerConfig requires checkpoint_path or resolved_checkpoint_path")
        return NeuralPreconditioner(
            checkpoint_path=ckpt,
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

    # Standard cases with explicit dispatch for type safety
    if config.type in (PreconditionerType.IDENTITY, PreconditionerType.NONE):
        return Identity()
    if config.type == PreconditionerType.JACOBI:
        return JacobiPreconditioner(matrix)
    if config.type == PreconditionerType.ILU:
        return ILUPreconditioner(matrix)
    if config.type == PreconditionerType.ICHOLESKY:
        return ICholeskyPreconditioner(matrix)

    raise ValueError(f"Unsupported preconditioner type: {config.type}")


def _extract_schedule(cfg: ConcretePreconditionerConfig) -> PreconditionerScheduleConfig:
    """Extract scheduling parameters from preconditioner config.

    Pure function to extract scheduling concerns from mixed config.

    Args:
        cfg: Preconditioner configuration from TOML

    Returns:
        Extracted schedule configuration
    """
    return PreconditionerScheduleConfig(
        limit_iters=cfg.limit_iters,
        fallback=cfg.fallback,
    )


def create_scheduled_preconditioner(
    primary: Preconditioner,
    schedule: PreconditionerScheduleConfig,
) -> Preconditioner:
    """Create a scheduled preconditioner based on schedule config.

    Args:
        primary: Main preconditioner to apply
        schedule: Schedule configuration with iteration limit and fallback type

    Returns:
        ScheduledPreconditioner if limit_iters > 0, otherwise primary unchanged

    Example:
        >>> # Limit neural preconditioner to first 10 iterations
        >>> schedule = PreconditionerScheduleConfig(limit_iters=10)
        >>> scheduled = create_scheduled_preconditioner(neural_precond, schedule)
    """
    if schedule.limit_iters < 0:
        return primary

    from .implementations.scheduled import ScheduledPreconditioner

    # Create fallback preconditioner based on type
    if schedule.fallback == PreconditionerType.IDENTITY:
        fallback_precond = Identity()
    else:
        raise ValueError(f"Unsupported fallback type: {schedule.fallback}")

    return ScheduledPreconditioner(
        primary=primary,
        fallback=fallback_precond,
        limit_iters=schedule.limit_iters,
    )
