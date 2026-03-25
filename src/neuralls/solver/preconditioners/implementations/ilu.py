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

    def __init__(
        self,
        matrix: NDArray,
        drop_tol: float | None = None,
        fill_factor: int | None = None,
    ) -> None:
        """Initialize ILU preconditioner.

        Args:
            matrix: System matrix A
            drop_tol: Drop tolerance for sparse ILU (scipy spilu parameter)
            fill_factor: Fill factor for ILU (scipy spilu parameter)
        """
        self._drop_tol = drop_tol
        self._fill_factor = fill_factor
        super().__init__(matrix)

    def _compute_operator(self, matrix: NDArray):
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

        if self._drop_tol is None and self._fill_factor is None:
            return spilu(A_csc)
        if self._drop_tol is None:
            return spilu(A_csc, fill_factor=self._fill_factor)
        if self._fill_factor is None:
            return spilu(A_csc, drop_tol=self._drop_tol)
        return spilu(A_csc, drop_tol=self._drop_tol, fill_factor=self._fill_factor)

    def apply(self, residual: NDArray, context: PreconditionerContext | None = None) -> NDArray:
        """Solve (LU) z = r via triangular solves.

        Args:
            residual: Residual vector r
            context: Ignored (ILU doesn't need context)

        Returns:
            Preconditioned residual z = (LU)^{-1}r
        """
        return self._operator.solve(residual)
