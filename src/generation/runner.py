"""Registry and dispatcher for generation strategies."""

from __future__ import annotations

from typing import Any

import numpy as np

from .interfaces import IDataGenerationStrategy, GeneratedSamples, ArchiveData


class StrategyRegistry:
    """In-memory registry for generation strategies."""

    def __init__(self) -> None:
        self._strategies: dict[str, IDataGenerationStrategy] = {}

    def register(self, strategy: IDataGenerationStrategy) -> None:
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> IDataGenerationStrategy:
        if name not in self._strategies:
            raise KeyError(f"Unknown generation strategy '{name}'")
        return self._strategies[name]


_registry = StrategyRegistry()


def register_strategy(
    strategy_cls: type[IDataGenerationStrategy],
) -> type[IDataGenerationStrategy]:
    _registry.register(strategy_cls())  # type: ignore[arg-type]
    return strategy_cls


def run_generation(
    strategy_name: str,
    matrix: np.ndarray,
    rhs: np.ndarray | None,
    *,
    cfg: dict[str, Any],
    archive: ArchiveData | None = None,
) -> GeneratedSamples:
    """Execute a specific generation strategy.

    Args:
        strategy_name: Name of the strategy to run (must be registered)
        matrix: System matrix, shape (n, n)
        rhs: Mother RHS vector, shape (n,) (optional depending on strategy)
        cfg: Strategy configuration dictionary (validated by strategy)
        archive: Optional pre-loaded archive data to pass to the strategy

    Returns:
        GeneratedSamples containing matrix, rhs, solutions, and optional traces

    Raises:
        ValueError: If strategy requires RHS but none provided
        KeyError: If strategy name is unknown
    """
    strategy = _registry.get(strategy_name)
    if strategy.requires_rhs() and rhs is None:
        raise ValueError(
            f"Strategy '{strategy_name}' requires RHS but none was provided."
        )
    return strategy.generate(matrix, rhs, cfg=cfg, archive=archive)
