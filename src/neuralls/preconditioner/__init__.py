"""Preconditioner creation with builder pattern.

This package implements preconditioner creation using the Builder and Registry
patterns, following SOLID principles for extensibility and maintainability.

Public API:
    - create_default_registry(): Get standard preconditioner registry
    - PreconditionerRegistry: Register and create preconditioners
    - PreconditionerBuilder: Protocol for custom builders
    - PredictorFactory: Protocol for neural predictor factories
    - PreconditionerFn: Type alias for preconditioner functions

Architecture:
    - builders.py: Concrete builder implementations (Identity, Jacobi, ILU, Neural)
    - registry.py: Registry pattern for type-to-builder dispatch
    - predictor.py: Predictor factory protocol for dependency injection

Example Usage:
    >>> from neuralls.preconditioner import create_default_registry
    >>> from neuralls.configuration.preconditioner import StandardPreconditionerConfig
    >>>
    >>> # Create registry with standard builders
    >>> registry = create_default_registry()
    >>>
    >>> # Create Jacobi preconditioner
    >>> jacobi_config = StandardPreconditionerConfig(name="jacobi", type="jacobi")
    >>> jacobi_fn = registry.create(A, jacobi_config)
    >>> z = jacobi_fn(residual)
    >>>
    >>> # Create neural preconditioner
    >>> neural_config = NeuralPreconditionerConfig(
    ...     name="neural",
    ...     type="neural",
    ...     checkpoint_path=Path("model.ckpt"),
    ... )
    >>> neural_fn = registry.create(A, neural_config)
    >>> z = neural_fn(residual)

Extending with Custom Preconditioners:
    >>> from neuralls.preconditioner import PreconditionerRegistry
    >>>
    >>> class CustomBuilder:
    ...     def build(self, matrix, config):
    ...         return lambda r: custom_transform(r, matrix)
    ...     def supported_types(self):
    ...         return {"custom"}
    >>>
    >>> registry = create_default_registry()
    >>> registry.register(CustomBuilder())  # Extend without modifying core code!
    >>> custom_fn = registry.create(A, custom_config)
"""

from neuralls.preconditioner.registry import (
    PreconditionerRegistry,
    create_default_registry,
)
from neuralls.preconditioner.builders import (
    PreconditionerBuilder,
    PreconditionerFn,
)
from neuralls.preconditioner.predictor import PredictorFactory

__all__ = [
    "PreconditionerRegistry",
    "create_default_registry",
    "PreconditionerBuilder",
    "PreconditionerFn",
    "PredictorFactory",
]
