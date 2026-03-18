"""Configuration models for solver execution."""

from dataclasses import dataclass, field
from typing import Any

from neuralls.solver.monitoring.trace_mode import TraceMode


@dataclass(frozen=True)
class SolverConfig:
    """Structured configuration for solver execution.

    This dataclass captures the parameters used to run a solver,
    enabling reproducible execution and structured export.
    """

    algorithm: str
    """Name of the algorithm (e.g., 'preconditioned_cg')."""

    rtol: float
    """Relative tolerance for convergence."""

    atol: float
    """Absolute tolerance for convergence."""

    maxiter: int
    """Maximum number of iterations allowed."""

    trace_mode: TraceMode
    """Verbosity of iteration tracking."""

    m_max: int | None = None
    """Orthogonalization window size (used by FCG)."""

    preconditioner: str | None = None
    """Name or description of the preconditioner used."""

    extra_params: dict[str, Any] = field(default_factory=dict)
    """Additional algorithm-specific parameters."""
