"""Preconditioner abstractions and implementations for iterative solvers.

This package provides the algorithm layer for preconditioning:
- Base abstractions (Preconditioner, LinearPreconditioner, etc.)
- Concrete implementations (Identity, Jacobi, ILU, Neural, etc.)
- Framework adapters for neural preconditioners

This package has no dependency on platform config parsing. The TOML-driven
factory lives in `neuralls.composition.preconditioners.factory`, which is the
explicit glue between config models and this package.

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
    >>> # Factory from TOML — import from composition layer
    >>> from neuralls.composition.preconditioners.factory import create_preconditioner
    >>> config = load_comparison_config("comparison.toml")
    >>> precond = create_preconditioner(matrix, config.preconditioner)
"""

from .base import (
    Preconditioner,
    PreconditionerContext,
    LinearPreconditioner,
    NonLinearPreconditioner,
    ContextualPreconditioner,
    BindableInputs,
)
from .implementations import (
    Identity,
    JacobiPreconditioner,
    ILUPreconditioner,
    IC0Preconditioner,
    ICholeskyPreconditioner,
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
    "BindableInputs",
    # Implementations
    "Identity",
    "JacobiPreconditioner",
    "ILUPreconditioner",
    "IC0Preconditioner",
    "ICholeskyPreconditioner",
    "CallablePreconditioner",
    "ScheduledPreconditioner",
    "LinearOperatorPreconditioner",
    # Adapters
    "PredictorPort",
    "PredictorAdapter",
]
