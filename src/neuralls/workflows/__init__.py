"""Workflow package entry point with lazy imports to avoid cycles."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuralls.workflows.reporting import ExperimentResult
    from neuralls.solver.models.result import CGComparisonResult
    from neuralls.workflows.results import ComparisonRecommendations

__all__ = [
    "run_experiment_matrix",
    "run_cg_comparison",
    "format_results_summary",
    "summarize_best_combinations",
]


def run_experiment_matrix(
    experiments_config_path: Path,
    *,
    force: bool = False,
    project_root: Path | None = None,
    **kwargs: Any,
) -> list["ExperimentResult"]:
    from neuralls.workflows.runner import run_experiment_matrix as _run

    return _run(
        experiments_config_path,
        force=force,
        project_root=project_root,
        **kwargs,
    )


def run_cg_comparison(*args: Any, **kwargs: Any) -> dict[str, "CGComparisonResult"]:
    """Run CG with multiple preconditioners for comparison."""
    from neuralls.workflows.cg_runner import run_cg_comparison as _run

    return _run(*args, **kwargs)


def format_results_summary(*args: Any, **kwargs: Any) -> str:
    """Format CG results into a readable summary."""
    from neuralls.workflows.cg_runner import format_results_summary as _format

    return _format(*args, **kwargs)


def summarize_best_combinations(*args: Any, **kwargs: Any) -> "ComparisonRecommendations":
    """Summarize best-performing CG combinations."""
    from neuralls.workflows.cg_runner import summarize_best_combinations as _summarize

    return _summarize(*args, **kwargs)
