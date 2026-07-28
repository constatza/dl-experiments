"""Redirect stub — logic moved to domain/solver/comparison.py."""

from neuralls.domain.solver.comparison import (
    _requires_flexible_cg,
    format_results_summary,
    run_cg_comparison,
    summarize_best_combinations,
)

__all__ = [
    "_requires_flexible_cg",
    "format_results_summary",
    "run_cg_comparison",
    "summarize_best_combinations",
]
