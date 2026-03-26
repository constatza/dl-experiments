"""Preconditioner abstractions and implementations for iterative solvers.

This package provides the algorithm layer for preconditioning:
- Base abstractions (Preconditioner, LinearPreconditioner, etc.)
- Concrete implementations (Identity, Jacobi, ILU, Neural, etc.)
- Framework adapters for neural preconditioners

This package has no dependency on neuralls.configuration.  The TOML-driven
factory (create_preconditioner) lives in neuralls.assembly.preconditioner,
which is the explicit glue between configuration and this package.

Public API:
    Base classes:
        - Preconditioner: Abstract base for all preconditioners
        - LinearPreconditioner: Base for matrix-based preconditioners
        - NonLinearPreconditioner: Base for non-linear preconditioners
        - ContextualPreconditioner: Base for iteration-dependent preconditioners
        - PreconditionerContext: Dataclass with iteration state

    Implementations:
        - Identity: No preconditioning (z = r)
        - JacobiPreconditioner: Diagonal scaling (z = D^{-1}r)
        - ILUPreconditioner: Incomplete LU factorization
        - ICholeskyPreconditioner: Incomplete Cholesky factorization
        - NeuralPreconditioner: Neural network preconditioner
        - CallablePreconditioner: Wrap arbitrary function
        - ScheduledPreconditioner: Switch preconditioners based on iteration
        - LinearOperatorPreconditioner: Wrap SciPy LinearOperator

    Adapters (for advanced usage):
        - PredictorPort: Framework-agnostic interface
        - PredictorAdapter: Framework adapter protocol

Example:
    >>> # Direct instantiation (preferred)
    >>> precond = JacobiPreconditioner(matrix)
    >>> z = precond.apply(residual)
    >>>
    >>> # Factory from TOML — import from assembly layer
    >>> from neuralls.assembly.preconditioner import create_preconditioner
    >>> config = load_comparison_config("comparison.toml")
    >>> precond = create_preconditioner(matrix, config.preconditioner)
"""

from .base import (
    Preconditioner,
    PreconditionerContext,
    LinearPreconditioner,
    NonLinearPreconditioner,
    ContextualPreconditioner,
)
from .implementations import (
    Identity,
    JacobiPreconditioner,
    ILUPreconditioner,
    IC0Preconditioner,
    ICholeskyPreconditioner,
    NeuralPreconditioner,
    ScheduledPreconditioner,
)
from .callable import CallablePreconditioner
from .linear_operator import LinearOperatorPreconditioner
from .ports import PredictorPort, PredictorAdapter

__all__ = [
    # Base classes
    "Preconditioner",
    "PreconditionerContext",
    "LinearPreconditioner",
    "NonLinearPreconditioner",
    "ContextualPreconditioner",
    # Implementations
    "Identity",
    "JacobiPreconditioner",
    "ILUPreconditioner",
    "IC0Preconditioner",
    "ICholeskyPreconditioner",
    "NeuralPreconditioner",
    "CallablePreconditioner",
    "ScheduledPreconditioner",
    "LinearOperatorPreconditioner",
    # Adapters
    "PredictorPort",
    "PredictorAdapter",
]
