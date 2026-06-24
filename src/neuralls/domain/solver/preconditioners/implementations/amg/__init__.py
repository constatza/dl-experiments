"""AMG (Algebraic Multigrid) preconditioner package.

Public API:
    Presets (recommended entry points):
        - VCycleAMG: SA-AMG with V-cycle; standard default.
        - WCycleAMG: SA-AMG with W-cycle; more robust, higher cost per cycle.

    Core (for custom wiring):
        - AMGPreconditioner: Top-level preconditioner; implements Preconditioner + BindableInputs.

    Protocols (OCP extension points):
        - TransferOperator: P/R interface (sparse, neural, …).
        - MultigridSmoother: Smoothing interface (Jacobi, Gauss-Seidel, …).
        - CoarseningStrategy: Coarsening interface (aggregation, RS, neural, …).
        - MultigridCycle: Cycle interface (V, W, F, …).

    Smoother hierarchy:
        - SmootherBase: ABC for fixed-step error dampers.
        - JacobiSmoother: Weighted Jacobi smoother (SmootherBase).

    Cycles:
        - VCycle: Recursive V-cycle (γ = 1).
        - WCycle: W-cycle with two coarse-grid corrections (γ = 2).

    Coarsening:
        - SparseAggregationCoarsening: Smoothed aggregation (SA-AMG).
        - NeuralCoarseningStrategy: Neural coarsening strategy (stub).

    Transfer operators:
        - SparseTransferOperator: P/R backed by a scipy sparse matrix.
        - NeuralTransferOperator: P/R backed by neural predictors (stub).

    Data:
        - MultigridHierarchy, MultigridLevel: Frozen dataclasses for the grid hierarchy.
"""

from .amg import AMGPreconditioner
from .coarsening import NeuralCoarseningStrategy, SparseAggregationCoarsening
from .cycle import VCycle, WCycle
from .hierarchy import MultigridHierarchy, MultigridLevel
from .protocols import CoarseningStrategy, MultigridCycle, MultigridSmoother, TransferOperator
from .smoothers import JacobiSmoother, SmootherBase
from .transfer import NeuralTransferOperator, SparseTransferOperator
from .variants import VCycleAMG, WCycleAMG

__all__ = [
    # Presets
    "VCycleAMG",
    "WCycleAMG",
    # Core
    "AMGPreconditioner",
    # Protocols
    "TransferOperator",
    "MultigridSmoother",
    "CoarseningStrategy",
    "MultigridCycle",
    # Smoother hierarchy
    "SmootherBase",
    "JacobiSmoother",
    # Cycles
    "VCycle",
    "WCycle",
    # Coarsening
    "SparseAggregationCoarsening",
    "NeuralCoarseningStrategy",
    # Transfer operators
    "SparseTransferOperator",
    "NeuralTransferOperator",
    # Data
    "MultigridHierarchy",
    "MultigridLevel",
]
