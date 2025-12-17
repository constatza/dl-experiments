"""Preconditioner abstractions for Flexible PCG.

This module defines the Preconditioner abstraction and concrete implementations
for use in the Flexible PCG solver. It follows the Open/Closed principle by
providing an abstract base class that can be extended with new preconditioner
types without modifying existing code.

Mathematical Background:
    In preconditioned conjugate gradient methods, a preconditioner M approximates
    the system matrix A to accelerate convergence. At each iteration k, we solve:

        M z_k = r_k

    where r_k is the current residual and z_k is the preconditioned residual.
    The preconditioner M should satisfy:

    1. M ≈ A (good approximation for fast convergence)
    2. M is easy to invert (computationally efficient)
    3. M is SPD for standard CG (relaxed in Flexible PCG)

    Flexible PCG allows M_k to vary with iteration and even be non-symmetric,
    enabling neural preconditioners and adaptive strategies.

Theory References:
    - Notay, Y. (2000). Flexible Conjugate Gradients. SIAM J. Sci. Comput.
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. Ch. 9.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .info import IterationContext

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Preconditioner(ABC):
    """Abstract base class for CG preconditioners.

    A preconditioner transforms the residual r_k into a preconditioned direction z_k
    that better approximates the error. Concrete implementations can leverage matrix
    factorizations, neural networks, or other strategies.

    Mathematical Role:
        The preconditioner implements the mapping M^{-1}: R^n → R^n where we want
        M ≈ A for good convergence but M^{-1} is cheap to compute. In practice,
        we don't form M explicitly but provide the action z = M^{-1} r.
    """

    @abstractmethod
    def apply(self, residual: NDArray, context: IterationContext) -> NDArray:
        """Apply preconditioner to residual vector.

        Computes z_k = M_k^{-1} r_k, transforming the residual into a preconditioned
        search direction component. The context provides iteration number, solution
        estimate, and matrix for adaptive preconditioning strategies.

        Args:
            residual: Current residual vector r_k ∈ R^n.
            context: Iteration context containing current state (iteration number,
                solution estimate x_k, matrix A, right-hand side b).

        Returns:
            Preconditioned residual z_k ∈ R^n with shape matching input residual.

        Raises:
            ValueError: If output shape doesn't match input residual shape.

        Mathematical Notes:
            - Identity: z_k = r_k (no preconditioning)
            - Jacobi: z_k = D^{-1} r_k where D = diag(A)
            - ILU: z_k = (LU)^{-1} r_k via forward/backward substitution
            - Neural: z_k = f_θ(r_k, x_k, k) via learned network f_θ
        """
        ...


class IdentityPreconditioner(Preconditioner):
    """Identity preconditioner (no preconditioning).

    Returns the residual unchanged, equivalent to M = I. Used for standard
    unpreconditioned CG where the system matrix A is already well-conditioned
    or for baseline comparisons.

    Mathematical Properties:
        - M = I (identity matrix)
        - z_k = M^{-1} r_k = I r_k = r_k
        - Convergence rate: O(√κ(A)) where κ(A) = λ_max(A) / λ_min(A)
        - No computational overhead beyond standard CG

    Usage:
        >>> precond = IdentityPreconditioner()
        >>> r = np.array([1.0, 2.0, 3.0])
        >>> ctx = IterationContext(iteration=5, residual=r, ...)
        >>> z = precond.apply(r, ctx)
        >>> assert np.array_equal(z, r)  # z = r for identity
    """

    def apply(self, residual: NDArray, context: IterationContext) -> NDArray:
        """Return residual unchanged (identity preconditioning).

        Args:
            residual: Current residual vector r_k.
            context: Iteration context (unused for identity preconditioner).

        Returns:
            Copy of input residual vector.
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

    def apply(self, residual: NDArray, context: IterationContext) -> NDArray:
        return _ensure_vector(
            self.operator.matvec(np.asarray(residual, dtype=np.float64, copy=False)),
            residual,
            name="linear_operator_preconditioner",
        )


class CallablePreconditioner(Preconditioner):
    """Wraps user-provided callable preconditioners with flexible signatures.

    Adapts arbitrary Python callables (functions, lambdas, neural networks) into
    the Preconditioner interface by inspecting their signature and providing
    appropriate arguments. Supports multiple signature patterns:

    Supported Signatures:
        1. `f(r)` - Takes residual only
        2. `f(r, ctx)` - Takes residual and full context
        3. `f(ctx)` - Takes full context only (if param named ctx/context/iteration_context)

    The wrapper uses Python's inspect module to detect the signature and adapts
    the calling convention automatically, enabling seamless integration of diverse
    preconditioner implementations.

    Mathematical Interpretation:
        Regardless of the Python signature, the callable must implement:
            z_k = M_k^{-1}(r_k, x_k, k)
        where the dependencies on x_k and k are optional (accessed via context).

    Args:
        preconditioner: Callable implementing the preconditioning operation.
            Must return ndarray matching input residual shape.

    Raises:
        ValueError: If preconditioner returns None or wrong shape.

    Usage:
        >>> # Simple function signature
        >>> def jacobi_precond(r):
        ...     return r / diag_A
        >>> precond = CallablePreconditioner(jacobi_precond)

        >>> # Context-aware signature
        >>> def adaptive_precond(r, ctx):
        ...     if ctx.iteration < 10:
        ...         return simple_precond(r)
        ...     return expensive_precond(r)
        >>> precond = CallablePreconditioner(adaptive_precond)

        >>> # Neural preconditioner using context only
        >>> def neural_precond(ctx):
        ...     features = extract_features(ctx)
        ...     return model.predict(features)
        >>> precond = CallablePreconditioner(neural_precond)
    """

    def __init__(self, preconditioner: Callable[..., np.ndarray]) -> None:
        """Initialize callable preconditioner wrapper.

        Args:
            preconditioner: Callable implementing M^{-1} operation.
        """
        self._preconditioner = preconditioner
        self._call_pattern = self._detect_signature(preconditioner)

    def _detect_signature(
        self, func: Callable[..., np.ndarray]
    ) -> Callable[[IterationContext], np.ndarray]:
        """Detect callable signature and create appropriate wrapper.

        Uses Python's inspect module to examine the function signature and
        determine how to invoke it from an IterationContext. This enables
        flexible preconditioner APIs without requiring a single rigid signature.

        Detection Logic:
            1. No parameters → assume f(r) signature
            2. One parameter:
                - Named ctx/context/iteration_context → f(ctx)
                - Otherwise → f(r)
            3. Two+ parameters → f(r, ctx)

        Args:
            func: User-provided preconditioner callable.

        Returns:
            Wrapper function taking IterationContext and returning ndarray.

        Notes:
            - Signature detection handles lambdas, functions, and bound methods
            - Fallback to f(r) signature if inspect fails (e.g., C extensions)
        """
        try:
            signature = inspect.signature(func)
            params = list(signature.parameters.values())
        except (TypeError, ValueError):
            # Signature inspection failed (e.g., builtin, C extension)
            # Default to residual-only signature
            params = None

        if not params:
            # No parameters or inspection failed → residual-only
            return lambda ctx: _ensure_vector(
                self._preconditioner(ctx.residual),
                ctx.residual,
                name="preconditioner",
            )

        first_name = params[0].name
        if len(params) == 1:
            # Single parameter → check if it expects context
            if first_name in {"ctx", "context", "iteration_context"}:
                return lambda ctx: _ensure_vector(
                    self._preconditioner(ctx),
                    ctx.residual,
                    name="preconditioner",
                )
            # Otherwise assume it wants residual
            return lambda ctx: _ensure_vector(
                self._preconditioner(ctx.residual),
                ctx.residual,
                name="preconditioner",
            )

        # Two+ parameters → pass both residual and context
        return lambda ctx: _ensure_vector(
            self._preconditioner(ctx.residual, ctx),
            ctx.residual,
            name="preconditioner",
        )

    def apply(self, residual: NDArray, context: IterationContext) -> NDArray:
        """Apply wrapped preconditioner using detected signature.

        Args:
            residual: Current residual vector r_k.
            context: Iteration context providing full solver state.

        Returns:
            Preconditioned residual z_k = M^{-1} r_k.

        Raises:
            ValueError: If preconditioner returns None or wrong shape.
        """
        return self._call_pattern(context)
