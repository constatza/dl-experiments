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
    "CGComparisonResult",
    "ComparisonData",
    "ComparisonGeneral",
    "ComparisonRecommendations",
    "ComparisonResult",
    "IterationContext",
    "PlotPaths",
    "RankedRecommendation",
    "SolverParams",
    "SolverResult",
]
