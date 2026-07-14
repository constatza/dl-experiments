"""POD-2G (Proper Orthogonal Decomposition, two-grid) preconditioner package.

Reuses the generic multigrid engine from ``implementations.amg`` (AMGPreconditioner,
VCycle/WCycle, JacobiSmoother) with a POD-basis coarsening strategy in place of
algebraic aggregation, following Nikolopoulos et al. (2022), §3.3-3.4.

Public API:
    Preset (recommended entry point):
        - POD2GPreconditioner: POD-2G bundled with VCycle + JacobiSmoother.

    Core (for custom wiring, e.g. via AMGPreconditioner directly):
        - PODCoarseningStrategy: Builds a POD-reduced coarse level from a
          snapshot ensemble; satisfies the ``CoarseningStrategy`` protocol.
        - DenseTransferOperator: P/R backed by a dense POD basis; satisfies
          the ``TransferOperator`` protocol.
        - compute_pod_basis: Pure function computing the truncated POD basis
          (snapshot method) from a snapshot ensemble.

References:
    - Nikolopoulos, S., Kalogeris, I., Stavroulakis, G., & Papadopoulos, V.
      (2022). AI-enhanced iterative solvers for accelerating the solution of
      large-scale parametrized systems. arXiv:2207.02543.
"""

from .basis import compute_pod_basis
from .coarsening import PODCoarseningStrategy
from .transfer import DenseTransferOperator
from .variants import POD2GPreconditioner

__all__ = [
    # Preset
    "POD2GPreconditioner",
    # Core
    "PODCoarseningStrategy",
    "DenseTransferOperator",
    "compute_pod_basis",
]
