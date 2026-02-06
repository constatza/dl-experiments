"""Concrete preconditioner implementations.

This module provides all concrete preconditioner classes.
Base abstractions are in base.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from .base import (
    Preconditioner,
    LinearPreconditioner,
    NonLinearPreconditioner,
    ContextualPreconditioner,
    PreconditionerContext,
)

if TYPE_CHECKING:
    pass  # NDArray moved to top-level for use in generics


class Identity(Preconditioner):
    """Identity preconditioner: z = r (no preconditioning).

    Returns the residual unchanged, equivalent to M = I. Used for standard
    unpreconditioned CG where the system matrix A is already well-conditioned
    or for baseline comparisons.

    Mathematical Properties:
        - M = I (identity matrix)
        - z = M^{-1}r = Ir = r
        - Convergence rate: O(√κ(A)) where κ(A) = λ_max(A) / λ_min(A)
        - No computational overhead

    Example:
        >>> precond = Identity()
        >>> r = np.array([1.0, 2.0, 3.0])
        >>> z = precond.apply(r)
        >>> assert np.array_equal(z, r)  # z = r for identity
    """

    def __init__(self):
        """No matrix needed - identity works for any system."""
        pass

    def apply(self, residual: NDArray, context: PreconditionerContext | None = None) -> NDArray:
        """Return residual unchanged (identity preconditioning).

        Args:
            residual: Residual vector r
            context: Ignored (identity doesn't need context)

        Returns:
            Copy of input residual vector (z = r)
        """
        return residual.copy()


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

    def _compute_operator(self, matrix: NDArray, **kwargs) -> NDArray:
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

    def _compute_operator(self, matrix: NDArray, **kwargs) -> NDArray:
        """Store the lower triangular factor L.

        Args:
            matrix: Lower triangular matrix L
            **kwargs: Ignored

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
        from scipy.linalg import cho_solve

        # cho_solve expects (c, lower) where c is the factor
        # We pass lower=True since we have L
        return cho_solve((self._operator, True), residual)


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


class CallablePreconditioner(NonLinearPreconditioner):
    """Wrap arbitrary function as preconditioner.

    For custom/experimental preconditioners provided as functions.
    Replaces FunctionPreconditioner from previous design.

    Args:
        func: Function taking residual → preconditioned residual

    Example:
        >>> def my_precond(r):
        ...     return r * 0.5  # Simple damping
        >>> precond = CallablePreconditioner(my_precond)
        >>> z = precond.apply(r)
    """

    def __init__(self, func: Callable[[NDArray], NDArray]):
        """Initialize from function.

        Args:
            func: Function taking residual → preconditioned residual
        """
        self._func = func

    def apply(self, residual: NDArray, context: PreconditionerContext | None = None) -> NDArray:
        """Apply function to residual.

        Args:
            residual: Current residual vector r_k
            context: Ignored (callable preconditioners don't use context)

        Returns:
            Preconditioned residual z_k = f(r_k)
        """
        return self._func(residual)


class NeuralPreconditioner(NonLinearPreconditioner):
    """Neural network preconditioner with automatic cleanup.

    Loads trained model from checkpoint and applies it to residuals.
    GPU resources are automatically freed when the preconditioner is garbage collected.

    Args:
        checkpoint_path: Path to trained model checkpoint
        config_path: Optional model configuration
        data_config_path: Optional data configuration
        adapter: Optional predictor adapter (defaults to DLKitAdapter)

    Example:
        >>> # Simple! Direct instantiation
        >>> precond = NeuralPreconditioner(Path("model.ckpt"))
        >>> z = precond.apply(residual)
        >>> # GPU resources automatically freed when precond is garbage collected
    """

    def __init__(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
        adapter=None,  # Type: PredictorAdapter | None (avoid circular import)
    ):
        """Initialize neural preconditioner from checkpoint.

        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Optional model configuration
            data_config_path: Optional data configuration
            adapter: Optional predictor adapter (defaults to DLKitAdapter)
        """
        # Lazy import to avoid hard dependency on DLKit
        if adapter is None:
            from .adapters import DLKitAdapter

            adapter = DLKitAdapter()

        # Load predictor (GPU model)
        self._predictor = adapter.create_predictor(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            data_config_path=data_config_path,
        )

    def apply(self, residual: NDArray, context: PreconditionerContext | None = None) -> NDArray:
        """Apply neural network to residual.

        Args:
            residual: Current residual vector r_k
            context: Ignored (neural preconditioner doesn't use context)

        Returns:
            Preconditioned residual z_k = network(r_k)
        """
        return self._predictor.apply(residual)

    def cleanup(self) -> None:
        """Free GPU resources manually.

        Called automatically by __del__, but can be called explicitly
        if you want to free resources before garbage collection.
        """
        # Check if _predictor exists (might not if __init__ failed)
        if hasattr(self, "_predictor") and hasattr(self._predictor, "cleanup"):
            self._predictor.cleanup()

    def __del__(self):
        """Automatic cleanup on garbage collection.

        Ensures GPU resources are freed when the preconditioner
        is no longer referenced.
        """
        self.cleanup()


class ScheduledPreconditioner(ContextualPreconditioner):
    """Preconditioner that switches based on iteration schedule.

    Enables strategies like:
    - Use expensive preconditioner for first N iterations
    - Apply preconditioner every K iterations
    - Switch to fallback after iteration limit

    Args:
        primary: Main preconditioner to apply
        fallback: Fallback preconditioner (default: Identity)
        limit_iters: Switch to fallback after this iteration
        apply_every: Apply primary every N iterations (1 = every iteration)

    Example:
        >>> # Use ILU for first 10 iterations, then Jacobi
        >>> scheduled = ScheduledPreconditioner(
        ...     primary=ILUPreconditioner(A),
        ...     fallback=JacobiPreconditioner(A),
        ...     limit_iters=10
        ... )
        >>>
        >>> # Apply neural preconditioner every 5 iterations
        >>> scheduled = ScheduledPreconditioner(
        ...     primary=NeuralPreconditioner(ckpt),
        ...     fallback=Identity(),
        ...     apply_every=5
        ... )
    """

    def __init__(
        self,
        primary: Preconditioner,
        fallback: Preconditioner | None = None,
        limit_iters: int | None = None,
        apply_every: int = 1,
    ) -> None:
        """Initialize scheduled preconditioner.

        Args:
            primary: Main preconditioner to apply
            fallback: Fallback preconditioner (default: Identity)
            limit_iters: Switch to fallback after this iteration
            apply_every: Apply primary every N iterations (1 = every iteration)
        """
        self._primary = primary
        self._fallback = fallback or Identity()
        self._limit_iters = limit_iters
        self._apply_every = apply_every

    def apply(self, residual: NDArray, context: PreconditionerContext | None = None) -> NDArray:
        """Apply primary or fallback based on schedule.

        Args:
            residual: Current residual vector r_k
            context: Iteration state (required for scheduling)

        Returns:
            Preconditioned residual z_k

        Raises:
            ValueError: If context is None
        """
        if context is None:
            raise ValueError("ScheduledPreconditioner requires context for iteration tracking")

        iteration = context.iteration

        # Determine which preconditioner to use
        use_primary = True
        if self._limit_iters is not None and iteration >= self._limit_iters:
            use_primary = False
        if self._apply_every > 1 and iteration % self._apply_every != 0:
            use_primary = False

        precond = self._primary if use_primary else self._fallback

        # Apply chosen preconditioner (pass context through for nested contextual preconditioners)
        return precond.apply(residual, context)


# Backward compatibility adapters for scipy integration


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
