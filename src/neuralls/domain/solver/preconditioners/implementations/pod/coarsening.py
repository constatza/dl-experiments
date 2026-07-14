"""POD-based coarsening strategy (POD-2G, Nikolopoulos et al. 2022)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .basis import compute_pod_basis
from .transfer import DenseTransferOperator

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PODCoarseningStrategy:
    """Coarsening via a POD basis fit to a snapshot ensemble (POD-2G).

    Builds a single reduced level using a POD basis Φ_r fit to an ensemble
    of high-fidelity solution snapshots, instead of algebraic aggregation.
    The Galerkin coarse matrix K_r = Φ_rᵀKΦ_r (eq. 28) has shape
    (rank × rank), with rank ≪ n_dofs.

    The snapshot ensemble is a plain array — where it comes from (FEM
    archive files today, a neural snapshot generator later) is entirely a
    composition-layer concern; this class does not care.

    Args:
        snapshots: Solution snapshot ensemble, shape (n_samples, n_dofs).
        rank: Fixed mode count (int) or minimum cumulative captured energy
            (float in (0, 1]) — see ``compute_pod_basis``.
    """

    def __init__(self, snapshots: NDArray, rank: int | float) -> None:
        self._basis = compute_pod_basis(snapshots, rank)

    def build_transfer(self, A: NDArray) -> tuple[NDArray, DenseTransferOperator]:
        """Build the POD-reduced coarse level from the fine-grid matrix A.

        Args:
            A: Fine-grid matrix (n_dofs × n_dofs).

        Returns:
            ``(A_coarse, transfer)`` — A_coarse is the Galerkin coarse matrix
            (rank × rank); transfer applies Φ_r/Φ_rᵀ.
        """
        A_coarse = self._basis.T @ A @ self._basis
        return A_coarse, DenseTransferOperator(self._basis)
