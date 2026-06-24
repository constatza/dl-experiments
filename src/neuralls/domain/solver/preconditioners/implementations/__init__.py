"""Core preconditioner implementations.

This module exports concrete preconditioner implementations:
- Identity: No preconditioning
- JacobiPreconditioner: Diagonal scaling
- ILUPreconditioner: Incomplete LU factorization
- IC0Preconditioner: Zero-fill incomplete Cholesky
- ICholeskyPreconditioner: Parameterized incomplete Cholesky
- ScheduledPreconditioner: Switch preconditioners based on iteration

Note: Adapter classes (CallablePreconditioner, LinearOperatorPreconditioner)
are in the parent module as they wrap external interfaces.
"""

from .amg import AMGPreconditioner
from .ic0 import IC0Preconditioner
from .icholesky import ICholeskyPreconditioner
from .identity import Identity
from .ilu import ILUPreconditioner
from .jacobi import JacobiPreconditioner
from .scheduled import ScheduledPreconditioner

__all__ = [
    "Identity",
    "JacobiPreconditioner",
    "ILUPreconditioner",
    "IC0Preconditioner",
    "ICholeskyPreconditioner",
    "AMGPreconditioner",
    "ScheduledPreconditioner",
]
