"""AMG (Algebraic Multigrid) preconditioner package.

Public API:
    Core:
        - AMGPreconditioner: Top-level preconditioner; implements Preconditioner + BindableInputs.

    Protocols (extension points):
        - TransferOperator: P/R interface (OCP hook for sparse, neural, …).
        - MultigridSmoother: Smoothing interface (OCP hook for Jacobi, GS, …).
        - CoarseningStrategy: Coarsening interface (OCP hook for aggregation, RS, …).
        - MultigridCycle: Cycle interface (OCP hook for V, W, F cycles).

    Data:
        - MultigridHierarchy, MultigridLevel: Frozen dataclasses for the grid hierarchy.

    Implementations:
        - JacobiSmoother: Weighted Jacobi smoother.
        - SparseTransferOperator: P/R backed by a scipy sparse matrix.
        - NeuralTransferOperator: P/R backed by neural predictors (stub).
        - SparseAggregationCoarsening: Smoothed aggregation coarsening.
        - NeuralCoarseningStrategy: Neural coarsening strategy (stub).
        - VCycle: Recursive V-cycle.
"""

from .amg import AMGPreconditioner
from .coarsening import NeuralCoarseningStrategy, SparseAggregationCoarsening
from .cycle import VCycle
from .hierarchy import MultigridHierarchy, MultigridLevel
from .protocols import CoarseningStrategy, MultigridCycle, MultigridSmoother, TransferOperator
from .smoothers import JacobiSmoother
from .transfer import NeuralTransferOperator, SparseTransferOperator

__all__ = [
    # Core
    "AMGPreconditioner",
    # Protocols
    "TransferOperator",
    "MultigridSmoother",
    "CoarseningStrategy",
    "MultigridCycle",
    # Data
    "MultigridHierarchy",
    "MultigridLevel",
    # Implementations
    "JacobiSmoother",
    "SparseTransferOperator",
    "NeuralTransferOperator",
    "SparseAggregationCoarsening",
    "NeuralCoarseningStrategy",
    "VCycle",
]
