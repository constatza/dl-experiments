"""Configuration models for solver execution."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from neuralls.domain.solver.monitoring.trace_mode import TraceMode


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


type NormalizeSystem = Literal["none", "matrix", "rhs", "both", "diagonal", "spectral"]


@dataclass(frozen=True)
class SolverParams:
    """Numerical solver parameters for comparison runs.

    Args:
        rtol: Relative convergence tolerance.
        atol: Absolute convergence tolerance.
        max_iterations: Maximum iterations allowed.
        stopping_criterion: When to stop iteration.
        m_max: FCG orthogonalization restart parameter.
        breakdown_tol: Breakdown detection tolerance.
    """

    rtol: float
    atol: float
    max_iterations: int
    stopping_criterion: Literal["residual_norm", "fixed_iterations"]
    m_max: int
    breakdown_tol: float | None


@dataclass(frozen=True)
class ComparisonData:
    """Input data and normalization context for a comparison run.

    Args:
        matrix_path: Path to system matrix file.
        rhs_path: Path to right-hand side vector file.
        rhs_index: Index into rhs array (for multi-column rhs).
        dataset_alias: Optional dataset label.
        normalize_system: Normalization strategy to apply.
    """

    matrix_path: Path
    rhs_path: Path
    rhs_index: int = 0
    dataset_alias: str | None = None
    normalize_system: NormalizeSystem = "matrix"


@dataclass(frozen=True)
class ComparisonGeneral:
    """Grouped comparison context: solver params + data spec.

    Args:
        params: Solver configuration.
        data: Input data and normalization spec.
    """

    params: SolverParams
    data: ComparisonData
