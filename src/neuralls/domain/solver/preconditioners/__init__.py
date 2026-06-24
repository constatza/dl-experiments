"""Preconditioner abstractions and implementations for iterative solvers.

This package provides the algorithm layer for preconditioning:
- Base abstractions (Preconditioner, LinearPreconditioner, etc.)
- Concrete implementations (Identity, Jacobi, ILU, AMG, Neural, etc.)
- Framework adapters for neural preconditioners

This package has no dependency on platform config parsing. The TOML-driven
factory lives in `neuralls.composition.preconditioners.factory`, which is the
explicit glue between config models and this package.

Public API:
    Base classes:
        - Preconditioner: Abstract base for all preconditioners
        - LinearPreconditioner: Base for matrix-based preconditioners
        - NonLinearPreconditioner: Base for non-linear / iteration-varying preconditioners
        - PreconditionerContext: Dataclass with iteration state (always passed by CG solver)

    Implementations:
        - Identity: No preconditioning (z = r)
        - JacobiPreconditioner: Diagonal scaling (z = D^{-1}r)
        - ILUPreconditioner: Incomplete LU factorization
        - ICholeskyPreconditioner: Incomplete Cholesky factorization
        - AMGPreconditioner: Algebraic Multigrid (smoothed aggregation or neural P/R)
        - CallablePreconditioner: Wrap arbitrary function
        - ScheduledPreconditioner: Switch preconditioners based on iteration
        - LinearOperatorPreconditioner: Wrap SciPy LinearOperator

    Adapters (for advanced usage):
        - PredictorPort: Minimal framework-agnostic interface (residual only)
        - ExtraInputPredictorPort: Port for models needing named extra inputs
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
    BindableInputs,
    LinearPreconditioner,
    NonLinearPreconditioner,
    Preconditioner,
    PreconditionerContext,
)
from .callable import CallablePreconditioner
from .implementations import (
    IC0Preconditioner,
    ICholeskyPreconditioner,
    Identity,
    ILUPreconditioner,
    JacobiPreconditioner,
    ScheduledPreconditioner,
)
from .implementations.amg import AMGPreconditioner
from .linear_operator import LinearOperatorPreconditioner
from .ports import ExtraInputPredictorPort, PredictorAdapter, PredictorPort

__all__ = [
    # Base classes
    "Preconditioner",
    "PreconditionerContext",
    "LinearPreconditioner",
    "NonLinearPreconditioner",
    "BindableInputs",
    # Implementations
    "Identity",
    "JacobiPreconditioner",
    "ILUPreconditioner",
    "IC0Preconditioner",
    "ICholeskyPreconditioner",
    "AMGPreconditioner",
    "CallablePreconditioner",
    "ScheduledPreconditioner",
    "LinearOperatorPreconditioner",
    # Adapters
    "PredictorPort",
    "ExtraInputPredictorPort",
    "PredictorAdapter",
]
