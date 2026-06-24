"""Transfer operators for multigrid prolongation and restriction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from neuralls.domain.solver.preconditioners.ports import ExtraInputPredictorPort


class SparseTransferOperator:
    """Transfer operator backed by a scipy sparse prolongation matrix P.

    Prolongation: P @ coarse_vec (inject coarse correction to fine grid).
    Restriction:  P.T @ fine_vec  (Galerkin restriction, R = P^T).

    Args:
        P: Prolongation matrix (n_fine × n_coarse), scipy sparse.
    """

    def __init__(self, P: csr_matrix) -> None:
        self._P = P
        self._R = P.transpose()

    def prolongate(self, coarse: NDArray) -> NDArray:
        """Interpolate coarse-grid vector to fine grid."""
        return np.asarray(self._P @ coarse, dtype=np.float64).ravel()

    def restrict(self, fine: NDArray) -> NDArray:
        """Restrict fine-grid vector to coarse grid (R = P^T)."""
        return np.asarray(self._R @ fine, dtype=np.float64).ravel()


class NeuralTransferOperator:
    """Transfer operator implemented by two neural predictors.

    Prolongation and restriction are computed by separate neural networks.
    Extra inputs (positions, stiffness matrix) are pre-bound at construction
    time by ``NeuralCoarseningStrategy``; nothing outside that class sets them.

    Args:
        prolongator: Neural predictor for coarse→fine mapping.
        restrictor: Neural predictor for fine→coarse mapping.
        **bound_inputs: Pre-bound extra arrays forwarded on every apply call.
    """

    def __init__(
        self,
        prolongator: ExtraInputPredictorPort,
        restrictor: ExtraInputPredictorPort,
        **bound_inputs: NDArray,
    ) -> None:
        self._prolongator = prolongator
        self._restrictor = restrictor
        self._bound = bound_inputs

    def prolongate(self, coarse: NDArray) -> NDArray:
        """Apply neural prolongator to coarse-grid vector."""
        return np.asarray(self._prolongator.apply(coarse, **self._bound), dtype=np.float64).ravel()

    def restrict(self, fine: NDArray) -> NDArray:
        """Apply neural restrictor to fine-grid vector."""
        return np.asarray(self._restrictor.apply(fine, **self._bound), dtype=np.float64).ravel()
