"""Preconditioner builders - one builder per type.

This module implements the Builder Pattern for preconditioner creation,
following the Open/Closed Principle by allowing extension without modification.

Each builder is responsible for creating a specific type of preconditioner:
- IdentityBuilder: No preconditioning (M = I)
- JacobiBuilder: Diagonal scaling (M = diag(A))
- ILUBuilder: Incomplete LU factorization
- NeuralBuilder: Neural network-based preconditioner (with dependency injection)

Design Principles:
    - Single Responsibility: Each builder creates only one preconditioner type
    - Open/Closed: New types added by creating new builders, not modifying existing code
    - Dependency Inversion: NeuralBuilder depends on PredictorFactory abstraction

Example:
    >>> from neuralls.preconditioner.builders import JacobiBuilder
    >>> from neuralls.configuration.preconditioner import StandardPreconditionerConfig
    >>>
    >>> builder = JacobiBuilder()
    >>> config = StandardPreconditionerConfig(name="jacobi", type="jacobi")
    >>> precond_fn = builder.build(A, config)
    >>> z = precond_fn(residual)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import spilu

if TYPE_CHECKING:
    from neuralls.configuration.preconditioner import (
        PreconditionerConfig,
    )

# Type alias for preconditioner functions
PreconditionerFn = Callable[[NDArray], NDArray]


class PreconditionerBuilder(Protocol):
    """Builder protocol for creating preconditioner functions.

    Builders implement the Builder Pattern: each builder knows how to construct
    a specific type of preconditioner from a matrix and configuration.

    The protocol defines two methods:
    - build(): Create preconditioner function
    - supported_types(): Declare which preconditioner types this builder handles

    This enables the Registry Pattern: builders register themselves for specific
    types, and the registry dispatches creation requests to the appropriate builder.
    """

    def build(self, matrix: NDArray, config: PreconditionerConfig) -> PreconditionerFn:
        """Create preconditioner function from matrix and configuration.

        Args:
            matrix: System matrix A (n × n)
            config: Type-specific preconditioner configuration

        Returns:
            Preconditioner function taking residual → preconditioned_residual

        Mathematical Interpretation:
            For linear preconditioners: z = M^{-1}(r)
            For non-linear preconditioners: z = f(r)
        """
        ...

    def supported_types(self) -> set[str]:
        """Return preconditioner type names this builder supports.

        Returns:
            Set of type strings (e.g., {"jacobi"}, {"identity", "none"})

        Notes:
            Builders can support multiple type names (e.g., "none" → "identity")
        """
        ...


class IdentityBuilder:
    """Builder for identity/none preconditioners.

    Creates the identity preconditioner: z = r (no transformation).
    Equivalent to M = I in mathematical notation.

    Supports type names: "identity", "none"

    Properties:
        - No computational cost
        - No memory overhead
        - Equivalent to unpreconditioned solver
        - Used as baseline for comparison

    Example:
        >>> builder = IdentityBuilder()
        >>> config = StandardPreconditionerConfig(name="identity", type="identity")
        >>> identity_fn = builder.build(A, config)
        >>> z = identity_fn(residual)  # z == residual
    """

    def build(
        self, matrix: NDArray, config: PreconditionerConfig
    ) -> "PreconditionerFn":
        """Create identity preconditioner function.

        Args:
            matrix: System matrix (unused, accepted for interface consistency)
            config: Configuration (unused, type determines behavior)

        Returns:
            Function returning copy of input residual

        Notes:
            Returns a copy to maintain functional purity and avoid side effects.
        """
        return lambda residual: residual.copy()

    def supported_types(self) -> set[str]:
        """Return supported type names.

        Returns:
            {"identity", "none"} - both map to same implementation
        """
        return {"identity", "none"}


class JacobiBuilder:
    """Builder for Jacobi (diagonal scaling) preconditioner.

    Creates Jacobi preconditioner: z_i = r_i / A_ii
    Equivalent to M = diag(A) in mathematical notation.

    Properties:
        - O(n) storage for diagonal
        - O(n) cost per application
        - Effective for diagonally dominant matrices
        - Simple and robust

    Mathematical Background:
        The Jacobi preconditioner approximates A^{-1} by inverting only the
        diagonal. For well-conditioned diagonal elements, this provides good
        acceleration at minimal cost.

    Example:
        >>> builder = JacobiBuilder()
        >>> config = StandardPreconditionerConfig(name="jacobi", type="jacobi")
        >>> jacobi_fn = builder.build(A, config)
        >>> z = jacobi_fn(residual)
    """

    def build(
        self, matrix: NDArray, config: PreconditionerConfig
    ) -> "PreconditionerFn":
        """Create Jacobi preconditioner function.

        Args:
            matrix: System matrix A (extracts diagonal)
            config: Configuration (unused, type determines behavior)

        Returns:
            Function implementing z = diag(A)^{-1} r

        Notes:
            Handles near-zero diagonal elements by substituting 1.0 to avoid
            division by zero. This makes the preconditioner robust but may
            reduce effectiveness for poorly conditioned diagonals.
        """
        diag = np.diag(matrix)
        eps = 1e-14
        # Protect against division by zero: replace tiny values with 1.0
        diag_safe = np.where(np.abs(diag) < eps, 1.0, diag)
        diag_inv = 1.0 / diag_safe

        return lambda residual: diag_inv * residual

    def supported_types(self) -> set[str]:
        """Return supported type names.

        Returns:
            {"jacobi"}
        """
        return {"jacobi"}


class ILUBuilder:
    """Builder for ILU (Incomplete LU) preconditioner.

    Creates ILU preconditioner using sparse ILU factorization from SciPy.
    Approximates A ≈ LU with incomplete factorization to reduce fill-in.

    Properties:
        - Sparse storage (depends on fill level)
        - O(nnz) cost per triangular solve
        - More effective than Jacobi for general sparse matrices
        - Requires sparse matrix conversion

    Mathematical Background:
        ILU approximates the full LU factorization by dropping small fill-in
        entries according to a threshold or pattern. The resulting factors
        L and U are sparser than full LU, making forward/backward substitution
        cheaper while still providing good approximation to A^{-1}.

    Example:
        >>> builder = ILUBuilder()
        >>> config = StandardPreconditionerConfig(name="ilu", type="ilu")
        >>> ilu_fn = builder.build(A, config)
        >>> z = ilu_fn(residual)
    """

    def build(
        self, matrix: NDArray, config: PreconditionerConfig
    ) -> "PreconditionerFn":
        """Create ILU preconditioner function.

        Args:
            matrix: System matrix A (converted to CSC format internally)
            config: Configuration (unused, type determines behavior)

        Returns:
            Function implementing z = (LU)^{-1} r via triangular solves

        Notes:
            Converts matrix to CSC format for efficient sparse factorization.
            Stores ILU factors in closure for repeated application.
        """
        from neuralls.math_utils import _to_csc

        A_csc = _to_csc(matrix)
        # Type ignore: scipy sparse typing is incomplete
        ilu = spilu(A_csc)  # type: ignore[arg-type]
        ilu_dtype = np.asarray(matrix, copy=False).dtype

        return lambda residual: ilu.solve(residual.astype(ilu_dtype, copy=False))

    def supported_types(self) -> set[str]:
        """Return supported type names.

        Returns:
            {"ilu"}
        """
        return {"ilu"}


class NeuralBuilder:
    """Builder for neural preconditioners with dependency injection.

    Creates neural network-based preconditioner using learned approximation
    of A^{-1}. Uses Ports & Adapters pattern for framework independence.

    Design Patterns:
        - Dependency Inversion: Depends on PredictorAdapter abstraction
        - Resource Management: Context managers ensure GPU cleanup
        - Adapter Pattern: Framework-specific code isolated in adapters

    Properties:
        - Non-linear preconditioner (not matrix-based)
        - Cost depends on network architecture
        - Requires trained checkpoint
        - Automatic resource cleanup (no GPU memory leaks)

    Example:
        >>> builder = NeuralBuilder()  # Uses default DLKit adapter
        >>> config = NeuralPreconditionerConfig(
        ...     name="neural",
        ...     type="neural",
        ...     checkpoint_path=Path("model.ckpt"),
        ... )
        >>> neural_fn = builder.build(A, config)
        >>> z = neural_fn(residual)
        >>> # Automatic cleanup when solver completes

    Testing Example:
        >>> class MockAdapter(PredictorAdapter):
        ...     def create_predictor(self, **kwargs):
        ...         class MockPredictor:
        ...             def apply(self, r): return r * 0.5
        ...             def cleanup(self): pass
        ...         return MockPredictor()
        >>> builder = NeuralBuilder(adapter=MockAdapter())
        >>> # Testable without real checkpoint!
    """

    def __init__(self, adapter: PredictorAdapter | None = None) -> None:
        """Initialize with optional predictor adapter for DIP compliance.

        Args:
            adapter: Adapter for creating neural predictors.
                    If None, defaults to DLKitAdapter.

        Notes:
            Adapter is lazily imported to avoid hard dependency on DLKit.
        """
        if adapter is not None:
            self._adapter = adapter
        else:
            # Lazy import to avoid hard dependency
            from neuralls.preconditioner.adapters import DLKitAdapter

            self._adapter = DLKitAdapter()

    def build(
        self, matrix: NDArray, config: PreconditionerConfig
    ) -> "PreconditionerFn":
        """Create neural preconditioner function with resource management.

        Args:
            matrix: System matrix A (unused - kept for API compatibility)
            config: Neural preconditioner configuration with checkpoint path

        Returns:
            Function implementing z = f_θ(r) using trained network

        Notes:
            The predictor is created once and captured in closure.
            Resources are cleaned up when function goes out of scope.

        Raises:
            TypeError: If config is not NeuralPreconditionerConfig
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If model loading fails
        """
        # Type-safe cast for accessing neural-specific fields
        from neuralls.configuration.preconditioner import NeuralPreconditionerConfig

        if not isinstance(config, NeuralPreconditionerConfig):
            raise TypeError(
                f"NeuralBuilder requires NeuralPreconditionerConfig, got {type(config)}"
            )

        # Create predictor from adapter (loads model)
        predictor = self._adapter.create_predictor(
            checkpoint_path=config.checkpoint_path,
            config_path=config.config_path,
            data_config_path=config.data_config_path,
        )

        def _precondition(residual: NDArray) -> NDArray:
            """Apply neural preconditioner.

            Args:
                residual: Current residual vector

            Returns:
                Predicted solution/correction from neural network

            Raises:
                RuntimeError: If predictor unloaded or GPU error
            """
            return predictor.apply(residual)

        # Store cleanup reference for potential manual cleanup
        _precondition._cleanup = predictor.cleanup  # type: ignore[attr-defined]

        return _precondition

    def supported_types(self) -> set[str]:
        """Return supported type names.

        Returns:
            {"neural"}
        """
        return {"neural"}
