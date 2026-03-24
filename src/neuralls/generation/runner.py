"""Registry and dispatcher for generation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from .interfaces import (
    MatrixGenerationStrategy,
    SingleRhsGenerationStrategy,
    GeneratedSamples,
    ArchiveData,
)


@dataclass(frozen=True)
class MatrixStrategyRegistration:
    """Registered strategy entry for matrix-only generators."""

    strategy: MatrixGenerationStrategy
    supports_single_rhs: Literal[False] = False


@dataclass(frozen=True)
class SingleRhsStrategyRegistration:
    """Registered strategy entry for single-RHS-capable generators."""

    strategy: SingleRhsGenerationStrategy
    supports_single_rhs: Literal[True] = True


StrategyRegistration = MatrixStrategyRegistration | SingleRhsStrategyRegistration


class StrategyRegistry:
    """In-memory registry for generation strategies."""

    def __init__(self) -> None:
        self._strategies: dict[str, StrategyRegistration] = {}

    def register_matrix(self, strategy: MatrixGenerationStrategy) -> None:
        self._strategies[strategy.name] = MatrixStrategyRegistration(strategy)

    def register_single_rhs(self, strategy: SingleRhsGenerationStrategy) -> None:
        self._strategies[strategy.name] = SingleRhsStrategyRegistration(strategy)

    def get(self, name: str) -> StrategyRegistration:
        if name not in self._strategies:
            raise KeyError(f"Unknown generation strategy '{name}'")
        return self._strategies[name]


_registry = StrategyRegistry()


def register_strategy[StrategyClass](
    strategy_cls: type[StrategyClass],
) -> type[StrategyClass]:
    """Register a matrix-only generation strategy."""
    _registry.register_matrix(cast(MatrixGenerationStrategy, strategy_cls()))
    return strategy_cls


def register_single_rhs_strategy[StrategyClass](
    strategy_cls: type[StrategyClass],
) -> type[StrategyClass]:
    """Register a generation strategy that supports shared RHS dispatch."""
    _registry.register_single_rhs(cast(SingleRhsGenerationStrategy, strategy_cls()))
    return strategy_cls


def run_generation(
    strategy_name: str,
    matrix: np.ndarray,
    *,
    cfg: dict[str, Any],
    archive: ArchiveData | None = None,
    single_rhs: np.ndarray | None = None,
) -> GeneratedSamples:
    """Execute a specific generation strategy.

    Dispatches to the appropriate strategy type, passing single_rhs only to strategies that support it.

    Args:
        strategy_name: Name of the strategy to run (must be registered)
        matrix: System matrix, shape (n, n)
        cfg: Strategy configuration dictionary (validated by strategy)
        archive: Optional pre-loaded archive data to pass to the strategy
        single_rhs: Optional single RHS vector, shape (n,). If provided to single-RHS strategies
            (trace strategies), all samples will solve the same system A @ x = single_rhs

    Returns:
        GeneratedSamples containing matrix, rhs, solutions, and optional traces

    Raises:
        KeyError: If strategy name is unknown
    """
    registration = _registry.get(strategy_name)
    if isinstance(registration, SingleRhsStrategyRegistration):
        return registration.strategy.generate(
            matrix,
            cfg=cfg,
            single_rhs=single_rhs,
            archive=archive,
        )
    return registration.strategy.generate(matrix, cfg=cfg, archive=archive)
