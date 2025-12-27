"""Strategy pattern implementations for solver components.

This package contains all strategy implementations:
- orthogonalization: OrthogonalizationStrategy hierarchy
- reorthogonalization: ReorthogonalizationStrategy hierarchy  
- convergence: ConvergenceCriterion implementations
- direction: DirectionStrategy implementations (will be moved here)

Design Principles:
    - Strategy Pattern: Interchangeable algorithms
    - Open/Closed: Easy to add new strategies without modifying existing code
    - Dependency Inversion: Solvers depend on abstractions, not concrete strategies
"""

from .convergence import CombinedToleranceCriterion, IConvergenceCriterion
from .orthogonalization import (
    FullOrthogonalization,
    ModifiedGramSchmidt,
    OrthogonalizationReport,
    OrthogonalizationStrategy,
    TruncatedGramSchmidt,
)
from .reorthogonalization import (
    FullReorthogonalization,
    PartialReorthogonalization,
    ReorthogonalizationStrategy,
    SelectiveReorthogonalization,
)

__all__ = [
    # Orthogonalization
    "OrthogonalizationStrategy",
    "OrthogonalizationReport",
    "TruncatedGramSchmidt",
    "ModifiedGramSchmidt",
    "FullOrthogonalization",
    # Reorthogonalization
    "ReorthogonalizationStrategy",
    "FullReorthogonalization",
    "PartialReorthogonalization",
    "SelectiveReorthogonalization",
    # Convergence
    "IConvergenceCriterion",
    "CombinedToleranceCriterion",
]
