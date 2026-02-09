"""Incomplete LU (ILU) preconditioner."""

from __future__ import annotations

from numpy.typing import NDArray

from ..base import LinearPreconditioner, PreconditionerContext


class ILUPreconditioner(LinearPreconditioner):
    """Incomplete LU preconditioner: z = (LU)^{-1}r.

    Incomplete LU factorization preconditioner.
    More effective than Jacobi, but more expensive to apply.

    Mathematical Properties:
        - M ≈ A via incomplete LU factorization
        - z = (LU)^{-1}r via forward/backward substitution
        - O(nnz) storage, O(nnz) application cost
        - Not guaranteed SPD even if A is SPD

    Example:
        >>> # Simple! User passes matrix
        >>> precond = ILUPreconditioner(matrix)
        >>> z = precond.apply(residual)  # z = (LU)^{-1}r
    """

    def _compute_operator(self, matrix: NDArray, *args, **kwargs):
        """Compute sparse ILU factorization.

        Args:
            matrix: System matrix A

        Returns:
            scipy.sparse.linalg.SuperLU object for fast solves
        """
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import spilu

        # Convert to CSC for efficient factorization
        A_csc = csc_matrix(matrix)
        return spilu(A_csc, *args, **kwargs)  # Returns SuperLU object

    def apply(self, residual: NDArray, context: PreconditionerContext | None = None) -> NDArray:
        """Solve (LU) z = r via triangular solves.

        Args:
            residual: Residual vector r
            context: Ignored (ILU doesn't need context)

        Returns:
            Preconditioned residual z = (LU)^{-1}r
        """
        return self._operator.solve(residual)
