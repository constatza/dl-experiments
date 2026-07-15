"""Workflow configuration and reporting models for solver orchestration."""

from .config import ComparisonData, ComparisonGeneral, SolverParams
from .result import (
    CGComparisonResult,
    ComparisonRecommendations,
    ComparisonResult,
    IterationContext,
    PlotPaths,
    RankedRecommendation,
    SolverResult,
)

__all__ = [
    "SolverResult",
    "CGComparisonResult",
    "IterationContext",
    "PlotPaths",
    "RankedRecommendation",
    "ComparisonRecommendations",
    "ComparisonResult",
    "SolverParams",
    "ComparisonData",
    "ComparisonGeneral",
]
