"""Typed specifications for workflow stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.configuration.domain import ExperimentWorkspace


@dataclass(frozen=True)
class ComparisonSpec:
    """Inputs needed to run a comparison."""

    name: str
    model_config: Path
    data_config: Path
    solver_config: Path
    workspace: ExperimentWorkspace
    checkpoint: Path
    matrix_override: Path | None = None
    rhs_override: Path | None = None
    figures_dir: Path | None = None
    output_dir: Path | None = None
    settings: Any | None = None


@dataclass(frozen=True)
class ComparisonParams:
    """Runtime parameters for comparison execution."""

    matrix: Path | None = None
    rhs: Path | None = None
    output_dir: Path | None = None
    figures_dir: Path | None = None
    save_plots: bool = True
    breakdown_tol: float | None = None
    limits: "PreconditionerLimits" = field(default_factory=lambda: PreconditionerLimits())
    reorthogonalize: str = "full"
    reorthog_window: int = 10
    reorthog_threshold: float = 0.0


@dataclass(frozen=True)
class PreconditionerLimits:
    """Limiters and fallback behavior for preconditioners."""

    apply_every: int = 1
    first_n: int | None = None
    neural_iters: int | None = None
    fallback_preconditioner: str = "identity"


@runtime_checkable
class PreconditionerLimitsProtocol(Protocol):
    apply_every: int
    first_n: int | None
    neural_iters: int | None
    fallback_preconditioner: str


@runtime_checkable
class ComparisonParamsProtocol(Protocol):
    matrix: Path | None
    rhs: Path | None
    output_dir: Path | None
    figures_dir: Path | None
    save_plots: bool
    breakdown_tol: float | None
    limits: PreconditionerLimitsProtocol
    reorthogonalize: str
    reorthog_window: int
    reorthog_threshold: float


@dataclass(frozen=True)
class ComparisonOutcome:
    """Result for a single comparison."""

    name: str
    success: bool
    error: str | None = None
    payload: dict[str, Any] | None = None
