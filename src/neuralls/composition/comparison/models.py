"""Composition-layer orchestration models for comparison workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neuralls.domain.solver.models.result import ComparisonResult
from neuralls.platform.config.models.workspace import ExperimentWorkspace
from neuralls.shared.types import ComparisonRhsGenerationKind, RhsKind

__all__ = [
    "ComparisonResult",
    "ComparisonSpec",
    "ComparisonParams",
    "ComparisonOutcome",
    "ComparisonPaths",
    "LinearSystem",
    "ResolvedComparisonInput",
]


@dataclass(frozen=True)
class ComparisonSpec:
    """Inputs needed to run a single comparison.

    Bundles all paths and resolved artifacts required by the comparison
    orchestrator. Immutable — constructed once by the batch runner.

    Args:
        comparison_id: Stable identifier for this comparison.
        comparison_display_name: Human-readable label.
        model_config: Path to model configuration TOML.
        data_config: Path to data configuration TOML.
        comparison_config: Path to comparison configuration TOML.
        workspace: Resolved experiment workspace.
        checkpoint: Path to model checkpoint.
        matrix_override: Optional matrix file override.
        rhs_override: Optional rhs file override.
        figures_dir: Optional figures directory override.
        output_dir: Optional output directory override.
    """

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
class ComparisonPaths:
    """Resolved paths for a single comparison run.

    Attributes:
        matrix: Path to system matrix file (.txt or .npy).
        rhs: Path to right-hand side vector file (.txt or .npy).
        output: Root output directory for comparison results.
        figures: Directory for diagnostic plots.
    """

    matrix: Path
    rhs: Path
    output: Path
    figures: Path


@dataclass(frozen=True)
class LinearSystem:
    """Loaded and validated linear system (A, b) pair.

    Attributes:
        matrix: System matrix A in Ax=b (shape: n x n).
        rhs: Right-hand side vector b in Ax=b (shape: n,).
    """

    matrix: np.ndarray
    rhs: np.ndarray


@dataclass(frozen=True)
class ResolvedComparisonInput:
    """Resolved comparison system and provenance used for execution and tracking."""

    matrix: np.ndarray
    rhs: np.ndarray
    matrix_dataset_id: str
    matrix_index: int
    rhs_source_type: str
    rhs_dataset_id: str | None = None
    rhs_index: int | None = None
    rhs_kind: RhsKind | None = None
    generator_kind: ComparisonRhsGenerationKind | None = None
    generator_params: dict[str, object] | None = None


@dataclass(frozen=True)
class ComparisonParams:
    """Runtime parameters for comparison execution (reserved for future use)."""


@dataclass(frozen=True)
class ComparisonOutcome:
    """Result for a single comparison run.

    Args:
        comparison_id: Stable identifier for this comparison.
        comparison_display_name: Human-readable label.
        success: Whether the comparison completed without error.
        error: Error message if failed.
        payload: Full comparison result if successful.
        warnings: Non-fatal warning messages.
    """

    comparison_id: str
    comparison_display_name: str
    success: bool
    error: str | None = None
    payload: ComparisonResult | None = None
    warnings: tuple[str, ...] = ()
