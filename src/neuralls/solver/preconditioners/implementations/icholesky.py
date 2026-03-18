"""Incomplete Cholesky preconditioner using provided factor."""

from __future__ import annotations

from numpy.typing import NDArray
from scipy.linalg import cho_solve

from ..base import LinearPreconditioner, PreconditionerContext


class ICholeskyPreconditioner(LinearPreconditioner[NDArray]):
    """Incomplete Cholesky preconditioner using provided factor L.

    Uses a provided lower triangular matrix L (approximate Cholesky factor)
    to precondition using M = L L^T.

    Mathematical Properties:
        - M = L L^T
        - z = M^{-1}r = (L L^T)^{-1}r
        - Solved via scipy.linalg.cho_solve((L, True), r)

    Note:
        Unlike other preconditioners, this class expects the INPUT matrix
        to be the factor L, not the system matrix A.

    Example:
        >>> # User passes factor L
        >>> precond = ICholeskyPreconditioner(L)
        >>> z = precond.apply(residual)  # z = (LL^T)^{-1}r
    """

    def _compute_operator(self, matrix: NDArray) -> NDArray:
        """Store the lower triangular factor L.

        Args:
            matrix: Lower triangular matrix L

        Returns:
            The matrix L itself.
        """
        return matrix

    def apply(self, residual: NDArray, context: PreconditionerContext | None = None) -> NDArray:
        """Solve (L L^T) z = r using cho_solve.

        Args:
            residual: Residual vector r
            context: Ignored (ICholesky doesn't need context)

        Returns:
            Preconditioned residual z = (LL^T)^{-1}r
        """
        # cho_solve expects (c, lower) where c is the factor
        # We pass lower=True since we have L
        return cho_solve((self._operator, True), residual)
