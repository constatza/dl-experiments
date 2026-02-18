"""Typed result models for workflow outputs.

This module provides Pydantic models and dataclasses for workflow return values,
replacing untyped dict[str, Any] returns with structured, validated types.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ConfigDict, Field


@dataclass(frozen=True)
class PlotPaths:
    """Paths to generated plots from comparison workflow."""

    convergence: Path | None = None
    condition_numbers: Path | None = None
    parity: Path | None = None
    residuals: Path | None = None


class SolverRecommendations(BaseModel):
    """Best solver/preconditioner recommendations.

    Structure matches output from summarize_best_combinations().
    """

    best_overall: dict[str, Any] = Field(
        default_factory=dict,
        description="Best overall solver/preconditioner combination",
    )
    best_by_metric: dict[str, Any] = Field(
        default_factory=dict,
        description="Best combinations grouped by metric",
    )

    model_config = ConfigDict(
        frozen=True,
    )


class ComparisonResult(BaseModel):
    """Typed result from compare_preconditioners workflow.

    Replaces the untyped dict[str, Any] return value with a structured model.
    """

    results: dict[str, Any] = Field(
        ...,
        description="Raw solver results from run_cg_comparison",
    )
    summary: str = Field(
        ...,
        description="Formatted summary text of results",
    )
    plot_paths: dict[str, Path] = Field(
        default_factory=dict,
        description="Paths to generated plot files",
    )
    preconditioners: list[str] = Field(
        default_factory=list,
        description="List of preconditioner names tested",
    )
    condition_numbers: dict[str, float] = Field(
        default_factory=dict,
        description="Condition numbers of preconditioned matrices",
    )
    solver_params: Any = Field(
        ...,
        description="Solver parameters used (GeneralSolverParams)",
    )
    recommendations: dict[str, Any] = Field(
        default_factory=dict,
        description="Best solver recommendations",
    )
    output_dir: Path | None = Field(
        default=None,
        description="Output directory where comparison artefacts were saved",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow GeneralSolverParams
        frozen=True,
    )


@dataclass(frozen=True)
class TrainingResult:
    """Typed result from training workflow."""

    checkpoint_path: Path
    metrics: dict[str, float]
    best_epoch: int
    final_val_loss: float | None = None


@dataclass(frozen=True)
class PredictionResult:
    """Typed result from prediction workflow."""

    predictions_path: Path
    parity_plot: Path | None = None
    metrics: dict[str, float] | None = None
    num_samples: int = 0
