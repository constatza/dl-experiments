"""Typed dataclass models for workflow outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .specs import ComparisonGeneral


@dataclass(frozen=True)
class PlotPaths:
    """Paths to generated plots from comparison workflow."""

    convergence: Path | None = None
    condition_numbers: Path | None = None
    parity: Path | None = None
    residuals: Path | None = None
    iterations_barplot: Path | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Path] | None) -> PlotPaths:
        """Build typed plot paths from a loose mapping."""
        if mapping is None:
            return cls()
        return cls(
            convergence=mapping.get("convergence"),
            condition_numbers=mapping.get("condition_numbers"),
            parity=mapping.get("parity"),
            residuals=mapping.get("residuals"),
            iterations_barplot=mapping.get("iterations_barplot"),
        )

    def to_mapping(self) -> dict[str, Path]:
        """Return only populated plot paths."""
        values = {
            "convergence": self.convergence,
            "condition_numbers": self.condition_numbers,
            "parity": self.parity,
            "residuals": self.residuals,
            "iterations_barplot": self.iterations_barplot,
        }
        return {key: path for key, path in values.items() if path is not None}


@dataclass(frozen=True)
class RankedRecommendation:
    """Typed recommendation entry for a solver/preconditioner combination."""

    label: str
    iterations: int
    residual: float
    residual_abs: float
    breakdown: bool


@dataclass(frozen=True)
class ComparisonRecommendations:
    """Typed comparison recommendations."""

    ranked: tuple[RankedRecommendation, ...] = ()
    overall_best: RankedRecommendation | None = None


@dataclass(frozen=True)
class ArrayArtifact:
    """Reference to a persisted numpy array artifact."""

    path: Path
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ComparisonArtifactManifest:
    """Manifest of files emitted by comparison artifact writing."""

    comparison_toml: Path
    comparison_json: Path
    recommendations_json: Path
    summary_txt: Path
    config_copy: Path
    arrays: tuple[ArrayArtifact, ...] = ()


@dataclass(frozen=True)
class FallbackComparisonResultEntry:
    """Minimal typed fallback for artifact writing in tests."""

    iterations: int
    residual: float
    residual_abs: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class ComparisonArtifactFallback:
    """Typed fallback source for artifact writing when tests patch the workflow."""

    summary: str = ""
    preconditioners: tuple[str, ...] = ()
    condition_numbers: dict[str, float] = field(default_factory=dict)
    plot_paths: PlotPaths = field(default_factory=PlotPaths)
    recommendations: ComparisonRecommendations = field(default_factory=ComparisonRecommendations)
    results: dict[str, FallbackComparisonResultEntry] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonResult:
    """Typed result from compare_preconditioners workflow."""

    results: dict[str, Any]
    summary: str
    solver_params: ComparisonGeneral
    plot_paths: PlotPaths = field(default_factory=PlotPaths)
    preconditioners: tuple[str, ...] = ()
    condition_numbers: dict[str, float] = field(default_factory=dict)
    recommendations: ComparisonRecommendations = field(default_factory=ComparisonRecommendations)
    output_dir: Path | None = None


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
