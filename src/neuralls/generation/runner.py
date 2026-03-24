"""Registry and dispatcher for generation strategies."""

from __future__ import annotations

from typing import Any

import numpy as np

from .interfaces import (
    GenerationStrategy,
    GeneratedSamples,
    ArchiveData,
)


class StrategyRegistry:
    """In-memory registry for generation strategies."""

    def __init__(self) -> None:
        self._strategies: dict[str, GenerationStrategy] = {}

    def register(self, strategy: GenerationStrategy) -> None:
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> GenerationStrategy:
        if name not in self._strategies:
            raise KeyError(f"Unknown generation strategy '{name}'")
        return self._strategies[name]


_registry = StrategyRegistry()


def register_strategy(
    strategy_cls: type[GenerationStrategy],
) -> type[GenerationStrategy]:
    _registry.register(strategy_cls())  # type: ignore[arg-type]
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
    strategy = _registry.get(strategy_name)

    # Trace strategies support optional single_rhs parameter
    SINGLE_RHS_STRATEGIES = {
        "residual_traces",
        "residuals",
        "gaussian_residuals",
        "search_directions",
    }

    # Dispatch based on strategy name
    if strategy_name in SINGLE_RHS_STRATEGIES:
        # Single-RHS strategy: pass single_rhs parameter (trace strategies)
        return strategy.generate(matrix, cfg=cfg, single_rhs=single_rhs, archive=archive)  # type: ignore[call-arg]
    else:
        # Matrix-only strategy: don't pass single_rhs (other strategies don't support it)
        return strategy.generate(matrix, cfg=cfg, archive=archive)  # type: ignore[call-arg]
