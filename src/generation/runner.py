"""Registry and dispatcher for generation strategies."""

from __future__ import annotations

from typing import Any

import numpy as np

from .interfaces import IDataGenerationStrategy, GeneratedSamples


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
) -> GeneratedSamples:
    strategy = _registry.get(strategy_name)
    if strategy.requires_rhs() and rhs is None:
        raise ValueError(
            f"Strategy '{strategy_name}' requires RHS but none was provided."
        )
    return strategy.generate(matrix, rhs, cfg=cfg)
