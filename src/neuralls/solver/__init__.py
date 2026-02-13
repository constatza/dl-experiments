"""Flexible CG solver module with modular architecture.

This package provides a complete implementation of Conjugate Gradient algorithms
with support for flexible preconditioning, non-SPD preconditioners, and strategy-based
design patterns.

Architecture Overview:

1. **Core Abstractions** (base.py):
   - IterativeSolverBase: Template method base class

2. **Solver Implementations**:
   - ConjugateGradientSolver (conjugate_gradient.py): Unified CG implementation
   - PCGSolver, FCGSolver: Convenience wrappers
   - SciPyCGSolver (scipy_wrapper.py): Wrapper for scipy.sparse.linalg.cg

3. **Strategies** (strategies/):
   - DirectionStrategy: Search direction computation
     * TwoTermRecurrenceStrategy: PCG two-term recurrence
     * OrthogonalizationDirectionStrategy: FCG orthogonalization
     * CompositeDirectionStrategy: Combined strategies
   - OrthogonalizationStrategy: Direction orthogonalization
   - IConvergenceCriterion: Convergence testing

4. **State Management** (models/):
   - SolverState, KrylovState, CGState: Immutable state hierarchy
   - DirectionHistory, ResidualHistory: History tracking
   - SolverResult: Result container

5. **Factory Functions** (factories.py):
   - flexible_cg(): FCG for neural/non-SPD preconditioners
   - preconditioned_cg(): PCG for classical preconditioners

6. **Monitoring** (monitoring/):
   - IterationHistory: Continuous iteration diagnostics
   - EventLog: Discrete solver event logging
   - ResidualHistoryTracker: Scipy callback integration

Mathematical Background:

Conjugate Gradient methods solve SPD linear systems Ax = b iteratively.
This package supports:
- Time-varying preconditioners M_k (flexible preconditioning)
- Non-symmetric preconditioners (neural networks)
- Explicit orthogonalization for numerical stability
- Strategy-based algorithm customization

Design Principles:
- SOLID: Single responsibility, open/closed, dependency inversion
- Strategy Pattern: Inject orthogonalization, convergence criteria
- Template Method: Base classes define algorithm structure
- Immutable State: Thread-safe, functional updates

References:
    - Notay, Y. (2000). Flexible Conjugate Gradients. SIAM J. Sci. Comput.
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems.
    - Hestenes & Stiefel (1952). Methods of Conjugate Gradients.
"""

from __future__ import annotations

# Factory functions (primary entry points)
from .factories import flexible_cg, pcg, scipy_cg

# Solver classes
from .conjugate_gradient import ConjugateGradientSolver, PCGSolver, FCGSolver
from .scipy_wrapper import SciPyCGSolver

# Core bases
from .base import IterativeSolverBase

# State models
from .models.state import CGState, SolverState
from .models.history import DirectionHistory, ResidualHistory
from .models.result import CGComparisonResult, IterationContext, SolverResult

# Strategies
from .strategies.convergence import CombinedToleranceCriterion, IConvergenceCriterion
from .strategies.direction import (
    DirectionStrategy,
    TwoTermRecurrenceStrategy,
    OrthogonalizationDirectionStrategy,
    CompositeDirectionStrategy,
)
from .strategies.orthogonalization import (
    OrthogonalizationStrategy,
    OrthogonalizationReport,
    PeriodicRestartOrthogonalization,
    TruncatedGramSchmidt,
    ModifiedGramSchmidt,
)

# Preconditioners
from .preconditioners import (
    CallablePreconditioner,
    Identity,
    Preconditioner,
)

# Monitoring (with type-safe events)
from .monitoring.events import EventType
from .monitoring.trace_mode import TraceMode
from .monitoring.iteration_history import IterationHistory
from .monitoring.event_log import EventLog
from .monitoring.residual_history_tracker import ResidualHistoryTracker
from .monitoring.callbacks import SciPyCallbackAdapter, InitialStateComputer

# Protocols for type-safe state access
from .models.protocols import HasVectors, HasDirectionHistory

# Utilities
from .utils.numerics import stable_dot_product, compute_curvature, check_breakdown

__all__ = [
    # Factory functions (recommended entry points)
    "flexible_cg",
    "pcg",
    "scipy_cg",
    # Solver classes
    "ConjugateGradientSolver",
    "PCGSolver",
    "FCGSolver",
    "SciPyCGSolver",
    # Core bases
    "IterativeSolverBase",
    # State models
    "SolverState",
    "SolverState",
    "CGState",
    "DirectionHistory",
    "ResidualHistory",
    "SolverResult",
    "CGComparisonResult",
    "IterationContext",
    # Strategies
    "IConvergenceCriterion",
    "CombinedToleranceCriterion",
    "DirectionStrategy",
    "TwoTermRecurrenceStrategy",
    "OrthogonalizationDirectionStrategy",
    "CompositeDirectionStrategy",
    "OrthogonalizationStrategy",
    "OrthogonalizationReport",
    "PeriodicRestartOrthogonalization",
    "TruncatedGramSchmidt",
    "ModifiedGramSchmidt",
    # Preconditioners
    "Preconditioner",
    "Identity",
    "CallablePreconditioner",
    # Monitoring (type-safe events)
    "EventType",
    "TraceMode",
    "IterationHistory",
    "EventLog",
    "ResidualHistoryTracker",
    "SciPyCallbackAdapter",
    "InitialStateComputer",
    # Protocols
    "HasVectors",
    "HasDirectionHistory",
    # Utilities
    "stable_dot_product",
    "compute_curvature",
    "check_breakdown",
]

__version__ = "2.0.0"
__author__ = "Graph-CG Contributors"
__description__ = "Modular Flexible CG solver with strategy pattern design"
