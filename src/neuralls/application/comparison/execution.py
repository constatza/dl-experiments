"""Redirect stub — logic moved to domain/solver/comparison.py."""

from neuralls.domain.solver.comparison import (
    run_cg_comparison,
    format_results_summary,
    summarize_best_combinations,
    _requires_flexible_cg,
)

__all__ = [
    "run_cg_comparison",
    "format_results_summary",
    "summarize_best_combinations",
    "_requires_flexible_cg",
]
