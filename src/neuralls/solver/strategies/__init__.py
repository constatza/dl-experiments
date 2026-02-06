"""Strategy pattern implementations for solver components.

This package contains all strategy implementations:
- orthogonalization: OrthogonalizationStrategy hierarchy
- convergence: ConvergenceCriterion implementations
- norms: Norm type and factory functions for convergence criteria
- direction: DirectionStrategy implementations (will be moved here)

Design Principles:
    - Strategy Pattern: Interchangeable algorithms
    - Open/Closed: Easy to add new strategies without modifying existing code
    - Dependency Inversion: Solvers depend on abstractions, not concrete strategies
"""

from .convergence import CombinedToleranceCriterion, IConvergenceCriterion
from .norms import Norm, energy_norm, euclidean_norm
from .orthogonalization import (
    ModifiedGramSchmidt,
    OrthogonalizationReport,
    OrthogonalizationStrategy,
    PeriodicRestartOrthogonalization,
    TruncatedGramSchmidt,
)

__all__ = [
    # Norms
    "Norm",
    "euclidean_norm",
    "energy_norm",
    # Orthogonalization
    "OrthogonalizationStrategy",
    "OrthogonalizationReport",
    "PeriodicRestartOrthogonalization",
    "TruncatedGramSchmidt",
    "ModifiedGramSchmidt",
    # Convergence
    "IConvergenceCriterion",
    "CombinedToleranceCriterion",
]
