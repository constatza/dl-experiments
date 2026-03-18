"""Jacobi (diagonal scaling) preconditioner."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..base import LinearPreconditioner, PreconditionerContext


class JacobiPreconditioner(LinearPreconditioner[NDArray]):
    """Jacobi (diagonal scaling) preconditioner: z = D^{-1}r.

    Diagonal preconditioning using inverse of matrix diagonal.
    Fast to apply, effective for diagonally dominant matrices.

    Mathematical Properties:
        - M = diag(A) (diagonal of system matrix)
        - z = D^{-1}r where D = diag(A)
        - O(n) storage, O(n) application cost
        - SPD if A is SPD

    Example:
        >>> # Simple! User passes matrix
        >>> precond = JacobiPreconditioner(matrix)
        >>> z = precond.apply(residual)  # z = D^{-1}r
    """

    def _compute_operator(self, matrix: NDArray) -> NDArray:
        """Extract and invert diagonal from matrix.

        Args:
            matrix: System matrix A

        Returns:
            Diagonal inverse for fast elementwise multiplication
        """
        diag = np.diag(matrix)
        # Protect against near-zero diagonals
        diag_safe = np.where(np.abs(diag) < 1e-14, 1.0, diag)
        return 1.0 / diag_safe  # Store diagonal inverse

    def apply(self, residual: NDArray, context: PreconditionerContext | None = None) -> NDArray:
        """Apply diagonal scaling.

        Args:
            residual: Residual vector r
            context: Ignored (Jacobi doesn't need context)

        Returns:
            Preconditioned residual z = D^{-1}r
        """
        return self._operator * residual
