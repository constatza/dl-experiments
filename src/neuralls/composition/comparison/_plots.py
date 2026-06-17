"""Comparison diagnostic plot generation."""

from __future__ import annotations

from neuralls.composition.comparison.models import ComparisonPaths
from neuralls.domain.analysis.spectra import plot_condition_numbers
from neuralls.domain.solver.models.result import CGComparisonResult, PlotPaths
from neuralls.platform.reporting.plots import plot_convergence_comparison, plot_metric_comparison


def _generate_comparison_plots(
    results: dict[str, CGComparisonResult],
    cond_numbers: dict[str, float],
    paths: ComparisonPaths,
    display_name: str | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    max_iterations: int | None = None,
) -> PlotPaths:
    """Generate diagnostic plots for a comparison run.

    Args:
        results: CG comparison results keyed by preconditioner name.
        cond_numbers: Effective condition numbers keyed by preconditioner name.
        paths: Resolved comparison paths (figures directory used for output).
        display_name: Optional title shown on all plots.
        rtol: Relative tolerance displayed as a reference line.
        atol: Absolute tolerance displayed as a reference line.
        max_iterations: Maximum iterations displayed as a reference line.

    Returns:
        Typed PlotPaths with paths to all generated figures.
    """
    suffix = paths.matrix.stem or "comparison"
    cond_path = plot_condition_numbers(
        cond_numbers,
        save_dir=paths.figures,
        suffix=suffix,
        title=display_name,
        rtol=rtol,
        atol=atol,
    )

    convergence_path = paths.figures / f"preconditioner_convergence_{suffix}.png"
    plot_convergence_comparison(
        results,
        metadata=None,
        save_path=convergence_path,
        title=display_name,
        rtol=rtol,
        atol=atol,
        max_iterations=max_iterations,
    )

    iter_path = paths.figures / f"preconditioner_iterations_{suffix}.png"
    plot_metric_comparison(
        list(results.keys()),
        [r.iterations for r in results.values()],
        metric_name="CG Iterations",
        title=display_name,
        horizontal=True,
        save_path=iter_path,
    )

    return PlotPaths(
        convergence=convergence_path,
        condition_numbers=cond_path,
        iterations_barplot=iter_path,
    )
