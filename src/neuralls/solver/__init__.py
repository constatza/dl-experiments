"""Flexible CG solver module with modular architecture.

This package provides a complete implementation of Conjugate Gradient algorithms
with support for flexible preconditioning, non-SPD preconditioners, and strategy-based
design patterns.

Architecture Overview:

1. **Core Abstractions** (core/):
   - ISolver, IIterativeSolver: Solver interfaces
   - IterativeSolverBase: Template method base class
   - KrylovSolverBase: Krylov subspace method base

2. **Solver Implementations** (solvers/):
   - FlexibleCGSolver: FCG with truncated orthogonalization
   - PreconditionedCGSolver: PCG with two-term recurrence
   - ConjugateGradientBase: CG-specific base class

3. **Strategies** (strategies/):
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
   - TraceRecorder: Iteration diagnostics and event logging
   - HistoryTracker: Residual and solution tracking

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
from .factories import flexible_cg, preconditioned_cg, scipy_cg

# Solver classes
from .solvers.fcg_solver import FlexibleCGSolver
from .solvers.pcg_solver import PreconditionedCGSolver
from .solvers.scipy_cg_solver import SciPyCGSolver
from .solvers.cg_base import ConjugateGradientBase

# Core interfaces and bases
from .core.interfaces import IIterativeSolver, ISolver
from .core.base import IterativeSolverBase
from .core.krylov_base import KrylovSolverBase

# State models
from .models.state import CGState, KrylovState, SolverState
from .models.history import DirectionHistory, ResidualHistory
from .models.result import CGComparisonResult, IterationContext, SolverResult

# Strategies
from .strategies.convergence import CombinedToleranceCriterion, IConvergenceCriterion
from .strategies.orthogonalization import (
    OrthogonalizationStrategy,
    OrthogonalizationReport,
    PeriodicRestartOrthogonalization,
    TruncatedGramSchmidt,
    ModifiedGramSchmidt,
    FullOrthogonalization,
)

# Preconditioners
from .preconditioners import (
    FunctionPreconditioner,
    IdentityPreconditioner,
    Preconditioner,
)

# Monitoring (with type-safe events)
from .monitoring.events import EventType
from .monitoring.trace_mode import TraceMode
from .monitoring.trace_recorder import TraceRecorder
from .monitoring.history_tracker import HistoryTracker
from .monitoring.callbacks import SciPyCallbackAdapter, InitialStateComputer

# Protocols for type-safe state access
from .models.protocols import HasVectors, HasDirectionHistory

# Comparison tools
from .comparison import (
    format_results_summary,
    run_cg_comparison,
    summarize_best_combinations,
)

# Utilities
from .utils.numerics import stable_dot_product, compute_curvature, check_breakdown

__all__ = [
    # Factory functions (recommended entry points)
    "flexible_cg",
    "preconditioned_cg",
    "scipy_cg",
    # Solver classes
    "FlexibleCGSolver",
    "PreconditionedCGSolver",
    "SciPyCGSolver",
    "ConjugateGradientBase",
    # Core interfaces and bases
    "ISolver",
    "IIterativeSolver",
    "IterativeSolverBase",
    "KrylovSolverBase",
    # State models
    "SolverState",
    "KrylovState",
    "CGState",
    "DirectionHistory",
    "ResidualHistory",
    "SolverResult",
    "CGComparisonResult",
    "IterationContext",
    # Strategies
    "IConvergenceCriterion",
    "CombinedToleranceCriterion",
    "OrthogonalizationStrategy",
    "OrthogonalizationReport",
    "PeriodicRestartOrthogonalization",
    "TruncatedGramSchmidt",
    "ModifiedGramSchmidt",
    "FullOrthogonalization",
    # Preconditioners
    "Preconditioner",
    "IdentityPreconditioner",
    "FunctionPreconditioner",
    # Monitoring (type-safe events)
    "EventType",
    "TraceMode",
    "TraceRecorder",
    "HistoryTracker",
    "SciPyCallbackAdapter",
    "InitialStateComputer",
    # Protocols
    "HasVectors",
    "HasDirectionHistory",
    # Comparison
    "run_cg_comparison",
    "format_results_summary",
    "summarize_best_combinations",
    # Utilities
    "stable_dot_product",
    "compute_curvature",
    "check_breakdown",
]

__version__ = "2.0.0"
__author__ = "Graph-CG Contributors"
__description__ = "Modular Flexible CG solver with strategy pattern design"
