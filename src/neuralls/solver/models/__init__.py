"""Typed, immutable data models for iterative solvers.

This package contains all state, history, and result dataclasses:
- state: Solver state hierarchy (SolverState, KrylovState, CGState)
- history: Direction and residual history tracking
- result: Solver execution results

Design Principles:
    - Immutable (frozen=True) for thread safety
    - Fully typed (no dynamic attributes)
    - Single Responsibility (each class has one purpose)
"""

from .history import DirectionHistory, ResidualHistory
from .result import CGComparisonResult, IterationContext, SolverResult
from .state import CGState, SolverState

__all__ = [
    # State hierarchy
    "SolverState",
    "SolverState",
    "CGState",
    # History
    "DirectionHistory",
    "ResidualHistory",
    # Results
    "SolverResult",
    "CGComparisonResult",
    "IterationContext",
]
