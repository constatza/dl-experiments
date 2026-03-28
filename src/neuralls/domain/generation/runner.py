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
    TracingSolverCallable,
)
from neuralls.domain.solver import pcg
from neuralls.domain.solver.monitoring.trace_mode import TraceMode


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


def _default_solver() -> TracingSolverCallable:
    """Return a default PCG solver with full iteration tracing (captures residuals, solutions, directions)."""

    def _solve(
        A: np.ndarray,
        b: np.ndarray,
        x0: np.ndarray,
        *,
        maxiter: int,
        rtol: float,
        atol: float,
    ) -> tuple[np.ndarray, Any]:
        return pcg(A, b, x0, maxiter=maxiter, rtol=rtol, atol=atol, trace_mode=TraceMode.FULL)

    return _solve


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
    solver: TracingSolverCallable | None = None,
    archive: ArchiveData | None = None,
    single_rhs: np.ndarray | None = None,
) -> GeneratedSamples:
    """Execute a specific generation strategy.

    Dispatches to the appropriate strategy type, passing single_rhs only to strategies that support it.

    Args:
        strategy_name: Name of the strategy to run (must be registered)
        matrix: System matrix, shape (n, n)
        cfg: Strategy configuration dictionary (validated by strategy)
        solver: Tracing solver callable. Defaults to SciPy PCG with full tracing when not provided.
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
        effective_solver: TracingSolverCallable = (
            solver if solver is not None else _default_solver()
        )
        return registration.strategy.generate(
            matrix,
            cfg=cfg,
            solver=effective_solver,
            single_rhs=single_rhs,
            archive=archive,
        )
    return registration.strategy.generate(matrix, cfg=cfg, archive=archive)
