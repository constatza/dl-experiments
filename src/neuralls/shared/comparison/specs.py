"""Typed specifications for workflow stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from typing import TYPE_CHECKING

from neuralls.shared.workspace import ExperimentWorkspace

if TYPE_CHECKING:
    from .results import ComparisonResult


type NormalizeSystem = Literal["none", "matrix", "rhs", "both", "diagonal", "spectral"]


@dataclass(frozen=True)
class SolverParams:
    """Numerical solver parameters."""

    rtol: float
    atol: float
    max_iterations: int
    stopping_criterion: Literal["residual_norm", "fixed_iterations"]
    m_max: int
    breakdown_tol: float | None


@dataclass(frozen=True)
class ComparisonData:
    """Input data and normalization context for a comparison run."""

    matrix_path: Path
    rhs_path: Path
    rhs_index: int = 0
    dataset_alias: str | None = None
    normalize_system: NormalizeSystem = "matrix"


@dataclass(frozen=True)
class ComparisonGeneral:
    """Grouped comparison context."""

    params: SolverParams
    data: ComparisonData


@dataclass(frozen=True)
class ComparisonSpec:
    """Inputs needed to run a comparison."""

    comparison_id: str
    comparison_display_name: str
    model_config: Path
    data_config: Path
    comparison_config: Path
    workspace: ExperimentWorkspace
    checkpoint: Path
    matrix_override: Path | None = None
    rhs_override: Path | None = None
    figures_dir: Path | None = None
    output_dir: Path | None = None


@dataclass(frozen=True)
class ComparisonParams:
    """Runtime parameters for comparison execution."""


@dataclass(frozen=True)
class ComparisonOutcome:
    """Result for a single comparison."""

    comparison_id: str
    comparison_display_name: str
    success: bool
    error: str | None = None
    payload: ComparisonResult | None = None
    warnings: tuple[str, ...] = ()
