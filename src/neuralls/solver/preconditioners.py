"""Preconditioner abstractions for iterative solvers.

This module defines the Preconditioner protocol and concrete implementations.
Preconditioners can be linear (matrix-based) or non-linear (neural networks,
adaptive strategies), providing maximum flexibility for solver acceleration.

Design Principles:
    - Protocol-based: Duck typing for maximum flexibility
    - General operators: Support both linear (M^{-1}) and non-linear transformations
    - No context coupling: Simpler interface, context via closures if needed

Mathematical Background:
    Preconditioners accelerate iterative solvers by transforming the linear system.
    At each iteration k, we compute:

        z_k = M^{-1}(r_k)  or  z_k = f(r_k)

    where:
    - r_k is the current residual
    - z_k is the preconditioned residual
    - M is an approximation to A (for linear preconditioners)
    - f is a learned/adaptive function (for non-linear preconditioners)

    Desirable properties:
    1. M ≈ A or f approximates A^{-1} (fast convergence)
    2. Application is computationally cheap
    3. For CG: M should be SPD (relaxed in Flexible CG)

    Flexible CG allows M_k to vary with iteration and be non-symmetric,
    enabling neural preconditioners and adaptive strategies.

References:
    - Notay, Y. (2000). Flexible Conjugate Gradients. SIAM J. Sci. Comput.
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. Ch. 9.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from collections.abc import Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Preconditioner(Protocol):
    """Protocol for preconditioner operators.

    A preconditioner transforms residual r_k into preconditioned direction z_k.
    Can be linear (matrix-based: z = M^{-1}r) or non-linear (learned: z = f(r)).

    This is a Protocol (structural typing), not an ABC, allowing maximum flexibility.
    Any object with an apply(residual) method satisfies this protocol.

    Examples of valid preconditioners:
        - Linear: Identity, Jacobi, ILU, incomplete Cholesky
        - Non-linear: Neural networks, learned approximations
        - Adaptive: Change behavior based on iteration (via closure)

    Mathematical interpretation:
        z = M^{-1}(r)  [linear preconditioner]
        z = f_θ(r)     [non-linear preconditioner, e.g., neural network]

    where the goal is to accelerate convergence of the iterative solver.
    """

    def apply(self, residual: NDArray) -> NDArray:
        """Apply preconditioner to residual vector.

        Computes z = M^{-1}(r) for linear preconditioners or z = f(r) for
        non-linear preconditioners. The output must match the input shape.

        Args:
            residual: Residual vector r ∈ R^n

        Returns:
            Preconditioned residual z ∈ R^n with shape matching input

        Examples:
            >>> # Identity preconditioner
            >>> z = precond.apply(r)  # z = r
            >>> assert z.shape == r.shape
            >>>
            >>> # Jacobi preconditioner
            >>> z = precond.apply(r)  # z = D^{-1}r
            >>>
            >>> # Neural preconditioner
            >>> z = precond.apply(r)  # z = network(r)
        """
        ...


class IdentityPreconditioner:
    """Identity preconditioner (no preconditioning).

    Returns the residual unchanged, equivalent to M = I. Used for standard
    unpreconditioned CG where the system matrix A is already well-conditioned
    or for baseline comparisons.

    Mathematical Properties:
        - M = I (identity matrix)
        - z = M^{-1}r = Ir = r
        - Convergence rate: O(√κ(A)) where κ(A) = λ_max(A) / λ_min(A)
        - No computational overhead

    Usage:
        >>> precond = IdentityPreconditioner()
        >>> r = np.array([1.0, 2.0, 3.0])
        >>> z = precond.apply(r)
        >>> assert np.array_equal(z, r)  # z = r for identity
    """

    def apply(self, residual: NDArray) -> NDArray:
        """Return residual unchanged (identity preconditioning).

        Args:
            residual: Residual vector r

        Returns:
            Copy of input residual vector (z = r)
        """
        return residual.copy()


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
    """Wrap a SciPy LinearOperator so it can be used by the solvers."""

    def __init__(self, operator: LinearOperator) -> None:
        self.operator = operator

    def apply(self, residual: NDArray) -> NDArray:
        """Apply linear operator to residual (no context needed)."""
        return _ensure_vector(
            self.operator.matvec(np.asarray(residual, dtype=np.float64, copy=False)),
            residual,
            name="linear_operator_preconditioner",
        )


class FunctionPreconditioner(Preconditioner):
    """Wrap a simple function (residual → result) as a Preconditioner.

    This is a simple adapter with no signature detection - the function
    must take a single NDArray argument (the residual) and return an NDArray.

    Args:
        func: Function taking residual and returning preconditioned result

    Example:
        >>> jacobi_fn = lambda r: r / diag_A
        >>> precond = FunctionPreconditioner(jacobi_fn)
        >>> z = precond.apply(residual)
    """

    def __init__(self, func: Callable[[NDArray], NDArray]) -> None:
        """Initialize function preconditioner.

        Args:
            func: Callable taking residual → preconditioned_residual
        """
        self._func = func

    def apply(self, residual: NDArray) -> NDArray:
        """Apply function to residual.

        Args:
            residual: Current residual vector r_k

        Returns:
            Preconditioned residual z_k = f(r_k)
        """
        result = self._func(residual)
        return _ensure_vector(result, residual, name="function_preconditioner")
