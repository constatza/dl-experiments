"""LinearOperator preconditioner wrapper for scipy compatibility."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from .base import Preconditioner, PreconditionerContext


def _ensure_vector(
    vector: np.ndarray | None, template: np.ndarray, *, name: str
) -> np.ndarray:
    """Validate and reshape preconditioner output to match template shape.

    Ensures the preconditioner returns a valid ndarray with shape matching the
    input residual. Handles None returns and automatic reshaping for flattened
    or incorrectly shaped outputs.

    Args:
        vector: Preconditioner output (may be None or misshapen).
        template: Reference vector defining expected shape.
        name: Preconditioner name for error messages.

    Returns:
        Validated ndarray with shape matching template.

    Raises:
        ValueError: If vector is None or cannot be reshaped to match template.
    """
    if vector is None:
        raise ValueError(f"{name} returned None; expected ndarray")
    arr = np.asarray(vector, dtype=np.float64)
    if arr.shape != template.shape:
        arr = arr.reshape(template.shape)
    return arr


class LinearOperatorPreconditioner(Preconditioner):
    """Wrap a SciPy LinearOperator so it can be used by the solvers.

    Provides backward compatibility with scipy.sparse.linalg interfaces.

    Args:
        operator: SciPy LinearOperator to wrap

    Example:
        >>> from scipy.sparse.linalg import LinearOperator
        >>> M_op = LinearOperator((n, n), matvec=lambda r: r / diag)
        >>> precond = LinearOperatorPreconditioner(M_op)
        >>> z = precond.apply(residual)
    """

    def __init__(self, operator: LinearOperator) -> None:
        """Initialize from LinearOperator.

        Args:
            operator: SciPy LinearOperator
        """
        self.operator = operator

    def apply(self, residual: NDArray, context: PreconditionerContext | None = None) -> NDArray:
        """Apply linear operator to residual.

        Args:
            residual: Residual vector r
            context: Ignored (LinearOperator doesn't need context)

        Returns:
            Preconditioned residual z = M^{-1}r
        """
        return _ensure_vector(
            self.operator.matvec(np.asarray(residual, dtype=np.float64, copy=False)),
            residual,
            name="linear_operator_preconditioner",
        )
