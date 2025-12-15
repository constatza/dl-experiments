"""Flexible PCG solver module with non-SPD preconditioner support.

This package provides a complete implementation of the Flexible Preconditioned
Conjugate Gradient (Flexible PCG) algorithm, designed to handle time-varying
and non-symmetric preconditioners while maintaining numerical robustness.

Key Components:

1. **IterationState & DirectionHistory** (state.py):
   Immutable dataclasses tracking algorithm progress, convergence status,
   restart events, numerical diagnostics, and direction history buffers.

2. **Direction Strategies** (direction_strategies.py):
   - DirectionStrategy: Abstract base class for direction computation
   - FlexibleCGDirection: Classical two-term recurrence with residual-only beta
   - TruncatedOrthogonalDirection: (Future) Explicit Gram-Schmidt orthogonalization

3. **Helper Functions** (helpers.py):
   - initialize_state: Initialize state from initial residual
   - convergence_check: Check convergence criterion
   - curvature: Evaluate A-conjugacy with non-SPD restart
   - step_length: Compute step length with breakdown detection
   - beta_update: Compute residual ratio with restart bounds
   - direction_update: Update search direction
   - residual_management: Periodic recomputation and divergence detection

4. **Main Algorithm** (fcg_solver.py):
   FlexibleConjugateGradientSolver implementing the FCG algorithm
   with full error handling and diagnostic reporting.

5. **Factory Functions** (factories.py):
   - flexible_cg: FCG with Gram-Schmidt orthogonalization (neural preconditioners)
   - preconditioned_cg: PCG with two-term recurrence (classical preconditioners or baseline when preconditioner=None)

6. **Result Structure** (info.py):
   SolverResult dataclass containing convergence metrics and optional event log.

Mathematical Background:

Flexible PCG extends standard CG to handle:
- Time-varying preconditioners M_k (k-dependent)
- Non-symmetric preconditioners (M_k^T != M_k)
- Non-SPD preconditioners (negative curvature possible)

The algorithm uses a residual-only beta formula:
    β_k = ||r_{k+1}||^2 / ||r_k||^2

instead of the standard CG formula, which would require M_k to be
symmetric and positive definite.

Numerical Robustness Features:
- Restart on loss of conjugacy (large beta)
- Restart on negative/small curvature
- Divergence detection and recovery via residual recomputation
- Periodic true residual recomputation to correct rounding errors
- Explicit breakdown detection (NaN/Inf) with safe termination

References:
    Flexible CG for multiple shifts: Golub & van der Vorst (2000)
    Flexible GMRES: Saad (1993)
    CG Theory: Hestenes & Stiefel (1952), Nocedal & Wright (2006)
    Non-symmetric preconditioners: Saad (2003)
"""

from __future__ import annotations

from .comparison import (
    format_results_summary,
    run_cg_comparison,
    summarize_best_combinations,
)
from .direction_strategies import (
    DirectionStrategy,
    FlexibleCGDirection,
    TruncatedOrthogonalDirection,
)
from .convergence import CombinedToleranceCriterion, IConvergenceCriterion
from .event_logging import VectorLogger
from .factories import flexible_cg, preconditioned_cg
from .fcg_solver import FlexibleConjugateGradientSolver
from .helpers import (
    beta_update,
    convergence_check,
    curvature,
    direction_update,
    initialize_state,
    residual_management,
    step_length,
)
from .info import CGComparisonResult, IterationContext, SolverResult
from .interfaces import IIterativeSolver, ISolver
from .preconditioners import (
    CallablePreconditioner,
    IdentityPreconditioner,
    Preconditioner,
)
from .reorthogonalization import (
    FullReorthogonalization,
    PartialReorthogonalization,
    ReorthogonalizationStrategy,
    SelectiveReorthogonalization,
    create_reorthogonalization_strategy,
)
from .state import DirectionHistory, IterationState

__all__ = [
    # Solver interfaces
    "ISolver",
    "IIterativeSolver",
    # Core solver classes
    "FlexibleConjugateGradientSolver",
    # Factory functions
    "flexible_cg",
    "preconditioned_cg",
    "run_cg_comparison",
    "format_results_summary",
    "summarize_best_combinations",
    # State management
    "IterationState",
    "DirectionHistory",
    "IterationContext",
    # Result structures
    "SolverResult",
    "CGComparisonResult",
    # Event sourcing
    "VectorLogger",  # NEW: Event logging for iteration history
    # Convergence rules
    "IConvergenceCriterion",
    "CombinedToleranceCriterion",
    # Direction strategies
    "DirectionStrategy",
    "FlexibleCGDirection",
    "TruncatedOrthogonalDirection",
    # Preconditioners
    "Preconditioner",
    "IdentityPreconditioner",
    "CallablePreconditioner",
    # Reorthogonalization strategies
    "ReorthogonalizationStrategy",
    "FullReorthogonalization",
    "PartialReorthogonalization",
    "SelectiveReorthogonalization",
    "create_reorthogonalization_strategy",
    # Helper functions
    "initialize_state",
    "convergence_check",
    "curvature",
    "step_length",
    "beta_update",
    "direction_update",
    "residual_management",
]

__version__ = "1.0.0"
__author__ = "Graph-CG Contributors"
__description__ = "Flexible PCG solver with non-SPD preconditioner support"
