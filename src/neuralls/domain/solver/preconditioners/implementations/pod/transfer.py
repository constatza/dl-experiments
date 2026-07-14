"""Transfer operator backed by a dense POD basis."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DenseTransferOperator:
    """Transfer operator backed by a dense POD basis Φ_r.

    Prolongation: Φ_r @ coarse (paper eq. 27, u ≈ Φ_r u_r).
    Restriction:  Φ_rᵀ @ fine  (Galerkin restriction, R = Φ_rᵀ, eq. 28).

    Unlike ``SparseTransferOperator``, Φ_r is dense — POD modes have no
    sparsity structure to exploit.

    Args:
        basis: POD basis Φ_r (n_fine × rank), typically from ``compute_pod_basis``.
    """

    def __init__(self, basis: NDArray) -> None:
        self._basis = basis

    def prolongate(self, coarse: NDArray) -> NDArray:
        """Map POD-reduced coefficients to the fine grid."""
        return self._basis @ coarse

    def restrict(self, fine: NDArray) -> NDArray:
        """Project a fine-grid vector onto the POD basis (R = Φ_rᵀ)."""
        return self._basis.T @ fine
