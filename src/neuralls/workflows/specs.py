"""Typed specifications for workflow stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from neuralls.configuration.domain import ExperimentWorkspace

if TYPE_CHECKING:
    from .results import ComparisonResult


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


@dataclass(frozen=True)
class ComparisonParams:
    """Runtime parameters for comparison execution."""

    save_plots: bool = True


@dataclass(frozen=True)
class ComparisonOutcome:
    """Result for a single comparison."""

    name: str
    success: bool
    error: str | None = None
    payload: ComparisonResult | None = None
