"""Registry pattern for preconditioner builders - enables extensibility.

This module implements the Registry Pattern combined with the Builder Pattern
to provide extensible preconditioner creation following the Open/Closed Principle.

The registry maintains a mapping from preconditioner type strings to builder
instances. New preconditioner types can be added by registering new builders
without modifying the registry code.

Design Principles:
    - Open/Closed: Open for extension (register new builders), closed for modification
    - Single Responsibility: Registry only manages lookup and dispatch, builders create
    - Dependency Inversion: Registry depends on PreconditionerBuilder protocol

Example:
    >>> registry = PreconditionerRegistry()
    >>> registry.register(JacobiBuilder())
    >>> registry.register(CustomBuilder())  # Extend without modifying!
    >>>
    >>> precond_fn = registry.create(A, jacobi_config)
    >>> z = precond_fn(residual)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from neuralls.preconditioner.builders import (
        PreconditionerBuilder,
        PreconditionerFn,
    )
    from neuralls.configuration.preconditioner import PreconditionerConfig


class PreconditionerRegistry:
    """Registry for preconditioner builders.

    The registry implements the Registry Pattern: it maintains a type-to-builder
    mapping and dispatches creation requests to the appropriate builder.

    This follows the Open/Closed Principle:
    - **Open for extension**: New types added via `register(builder)`
    - **Closed for modification**: Registry code never changes

    Usage:
        >>> # Create registry
        >>> registry = PreconditionerRegistry()
        >>>
        >>> # Register standard builders
        >>> registry.register(IdentityBuilder())
        >>> registry.register(JacobiBuilder())
        >>> registry.register(ILUBuilder())
        >>>
        >>> # Extend with custom builder (no modification needed!)
        >>> registry.register(CustomBuilder())
        >>>
        >>> # Create preconditioner
        >>> precond_fn = registry.create(A, config)
        >>> z = precond_fn(residual)

    Thread Safety:
        This registry is NOT thread-safe. If using in multi-threaded contexts,
        wrap access in locks or create separate registries per thread.
    """

    def __init__(self) -> None:
        """Initialize empty registry.

        Starts with no builders registered. Call `register()` to add builders.
        """
        self._builders: dict[str, PreconditionerBuilder] = {}

    def register(self, builder: PreconditionerBuilder) -> None:
        """Register a builder for one or more preconditioner types.

        The builder declares which types it supports via `supported_types()`.
        Each type is mapped to this builder instance.

        Args:
            builder: Builder instance implementing PreconditionerBuilder protocol

        Raises:
            ValueError: If any supported type is already registered

        Example:
            >>> registry = PreconditionerRegistry()
            >>> registry.register(IdentityBuilder())  # Registers "identity" and "none"
            >>> registry.register(JacobiBuilder())    # Registers "jacobi"

        Notes:
            Builders that support multiple types (e.g., IdentityBuilder supports
            both "identity" and "none") will be registered under all type names.
        """
        for ptype in builder.supported_types():
            if ptype in self._builders:
                existing = self._builders[ptype]
                raise ValueError(
                    f"Type '{ptype}' already registered. "
                    f"Existing builder: {type(existing).__name__}, "
                    f"New builder: {type(builder).__name__}"
                )
            self._builders[ptype] = builder

    def create(
        self, matrix: NDArray, config: PreconditionerConfig
    ) -> PreconditionerFn:
        """Create preconditioner function from configuration.

        Looks up the appropriate builder for the config's type and delegates
        creation to that builder.

        Args:
            matrix: System matrix A (n × n)
            config: Preconditioner configuration (validated Pydantic model)

        Returns:
            Preconditioner function: residual → preconditioned_residual

        Raises:
            ValueError: If preconditioner type not registered

        Example:
            >>> config = StandardPreconditionerConfig(name="jacobi", type="jacobi")
            >>> jacobi_fn = registry.create(A, config)
            >>> z = jacobi_fn(residual)

        Notes:
            The configuration must have a `type` field matching a registered type.
            The config is passed to the builder, which may extract additional
            type-specific parameters (e.g., checkpoint path for neural).
        """
        builder = self._builders.get(config.type)
        if builder is None:
            available = sorted(self._builders.keys())
            raise ValueError(
                f"Unsupported preconditioner type: '{config.type}'. "
                f"Available types: {available}. "
                f"Register a builder for '{config.type}' via registry.register()"
            )
        return builder.build(matrix, config)

    def available_types(self) -> set[str]:
        """Return all registered preconditioner types.

        Returns:
            Set of type strings currently registered

        Example:
            >>> registry = create_default_registry()
            >>> types = registry.available_types()
            >>> print(sorted(types))
            ['identity', 'ilu', 'jacobi', 'neural', 'none']
        """
        return set(self._builders.keys())


def create_default_registry() -> PreconditionerRegistry:
    """Create registry with standard preconditioner builders.

    Convenience function that creates a registry and registers the four
    standard builders: Identity, Jacobi, ILU, and Neural.

    Returns:
        Registry with identity, jacobi, ilu, and neural builders registered

    Example:
        >>> registry = create_default_registry()
        >>> # Ready to use with all standard preconditioners
        >>> jacobi_fn = registry.create(A, jacobi_config)
        >>> neural_fn = registry.create(A, neural_config)

    Notes:
        The neural builder is registered with the default DLKitPredictorFactory.
        For testing or alternative ML frameworks, create a custom registry:

        >>> registry = PreconditionerRegistry()
        >>> registry.register(IdentityBuilder())
        >>> registry.register(JacobiBuilder())
        >>> registry.register(NeuralBuilder(predictor_factory=MockFactory()))
    """
    from neuralls.preconditioner.builders import (
        IdentityBuilder,
        JacobiBuilder,
        ILUBuilder,
        NeuralBuilder,
    )

    registry = PreconditionerRegistry()
    registry.register(IdentityBuilder())
    registry.register(JacobiBuilder())
    registry.register(ILUBuilder())
    registry.register(NeuralBuilder())
    return registry
