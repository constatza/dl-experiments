"""Preconditioner abstractions and implementations for iterative solvers.

This package provides all preconditioner functionality:
- Base abstractions (Preconditioner, LinearPreconditioner, etc.)
- Concrete implementations (Identity, Jacobi, ILU, Neural, etc.)
- Factory function for TOML workflow
- Framework adapters for neural preconditioners

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

    Factory:
        - create_preconditioner: Create from TOML configuration

    Adapters (for advanced usage):
        - PredictorPort: Framework-agnostic interface
        - PredictorAdapter: Framework adapter protocol

Example:
    >>> # Direct instantiation (preferred)
    >>> precond = JacobiPreconditioner(matrix)
    >>> z = precond.apply(residual)
    >>>
    >>> # Factory from TOML
    >>> config = load_solver_config("solver.toml")
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
from .builders import create_preconditioner, create_scheduled_preconditioner
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
    # Factory
    "create_preconditioner",
    "create_scheduled_preconditioner",
    # Adapters
    "PredictorPort",
    "PredictorAdapter",
]
