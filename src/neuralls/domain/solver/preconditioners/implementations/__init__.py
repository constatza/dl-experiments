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

from .identity import Identity
from .jacobi import JacobiPreconditioner
from .ilu import ILUPreconditioner
from .ic0 import IC0Preconditioner
from .icholesky import ICholeskyPreconditioner
from .scheduled import ScheduledPreconditioner

__all__ = [
    "Identity",
    "JacobiPreconditioner",
    "ILUPreconditioner",
    "IC0Preconditioner",
    "ICholeskyPreconditioner",
    "ScheduledPreconditioner",
]
