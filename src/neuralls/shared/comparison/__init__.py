"""Comparison-related shared DTOs."""

from neuralls.shared.comparison.results import (
    ArrayArtifact,
    ComparisonArtifactFallback,
    ComparisonArtifactManifest,
    ComparisonRecommendations,
    ComparisonResult,
    FallbackComparisonResultEntry,
    PlotPaths,
    PredictionResult,
    RankedRecommendation,
    TrainingResult,
)
from neuralls.shared.comparison.specs import ComparisonOutcome, ComparisonParams, ComparisonSpec

__all__ = [
    "ArrayArtifact",
    "ComparisonArtifactFallback",
    "ComparisonArtifactManifest",
    "ComparisonOutcome",
    "ComparisonParams",
    "ComparisonRecommendations",
    "ComparisonResult",
    "ComparisonSpec",
    "FallbackComparisonResultEntry",
    "PlotPaths",
    "PredictionResult",
    "RankedRecommendation",
    "TrainingResult",
]
