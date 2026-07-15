"""Single preconditioner comparison run — pure orchestration entry point."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from neuralls.composition.comparison._linear_system import (
    _load_linear_system,
    _log_matrix_condition_number,
)
from neuralls.composition.comparison._plots import _generate_comparison_plots
from neuralls.composition.comparison._preconditioner_setup import (
    PreconditionerService,
    _create_scheduled_preconditioners,
    _load_and_bind_extra_inputs,
)
from neuralls.composition.comparison.models import ComparisonPaths
from neuralls.composition.comparison.models import ResolvedComparisonInput
from neuralls.domain.analysis.spectra import PreconditionerCallable, compute_condition_numbers
from neuralls.domain.solver.comparison import (
    _to_numpy,
    format_results_summary,
    run_cg_comparison,
)
from neuralls.domain.solver.models.result import (
    ComparisonRecommendations,
    ComparisonResult,
)
from neuralls.platform.config.models.comparison import ComparisonGeneral
from neuralls.platform.config.models.preconditioner import PreconditionerConfig
from neuralls.platform.config.resolution import resolve_user_path
from neuralls.platform.storage.filesystem import ensure_dir


def _resolve_comparison_paths(
    *,
    general_params: ComparisonGeneral,
    output_root: Path | None,
    figures_root: Path | None,
) -> ComparisonPaths:
    """Resolve all paths for a comparison run.

    Args:
        general_params: Comparison general configuration with params+data context.
        output_root: Optional override for output root directory.
        figures_root: Optional override for figures directory.

    Returns:
        ComparisonPaths with all resolved and validated paths.

    Raises:
        ValueError: If matrix_path or rhs_path are missing.
    """
    matrix_file = Path(general_params.data.matrix_path)
    rhs_file = (
        Path(general_params.data.rhs_path)
        if general_params.data.rhs_path is not None
        else matrix_file
    )

    if output_root is not None:
        output_base = resolve_user_path(output_root)
    else:
        output_base = (Path.cwd() / "comparison" / matrix_file.stem).resolve()

    figs_base = Path(figures_root) if figures_root else output_base / "figures"

    return ComparisonPaths(
        matrix=matrix_file,
        rhs=rhs_file,
        output=output_base,
        figures=figs_base,
    )


def _ensure_comparison_directories(paths: ComparisonPaths) -> None:
    """Ensure the figures directory exists.

    Only the figures subdirectory needs explicit creation; the output root
    is either caller-supplied or an MLflow-managed artifact directory.
    mkdir(parents=True) inside ensure_dir creates any intermediate directories
    (including paths.output) as a side effect.

    Args:
        paths: Comparison paths with output and figures directories.
    """
    ensure_dir(paths.figures)


def compare_preconditioners(
    *,
    general_params: ComparisonGeneral,
    preconditioner_configs: Sequence[PreconditionerConfig],
    output_root: Path | None = None,
    figures_root: Path | None = None,
    display_name: str | None = None,
    resolved_input: ResolvedComparisonInput | None = None,
) -> ComparisonResult:
    """Run CG comparisons and generate diagnostics.

    Orchestrates an 11-step workflow:
    1. Validate inputs
    2. Resolve paths (matrix, rhs, output, figures)
    3. Load and validate linear system
    4. Create preconditioners via service
    5. Wrap preconditioners with scheduling
    6. Bind extra inputs (neural preconditioners)
    7. Compute condition numbers
    8. Run CG comparison
    9. Generate diagnostic plots
    10. Package and return result

    Args:
        general_params: Comparison general configuration.
        preconditioner_configs: Sequence of preconditioner configurations.
        output_root: Optional override for output root directory.
        figures_root: Optional override for figures directory.
        display_name: Optional display name for plots.

    Returns:
        ComparisonResult with results, summary, plot paths, and solver metadata.

    Raises:
        ValueError: If no preconditioner configs provided or required paths missing.
        FileNotFoundError: If matrix/rhs files don't exist.
    """
    if not preconditioner_configs:
        raise ValueError("At least one preconditioner config must be provided.")

    paths = _resolve_comparison_paths(
        general_params=general_params,
        output_root=output_root,
        figures_root=figures_root,
    )
    _ensure_comparison_directories(paths)

    resolved_matrix_index = (
        general_params.data.matrix_index if general_params.data.matrix_index is not None else 0
    )

    system = _load_linear_system(
        paths,
        rhs_sample_index=0,
        matrix_index=resolved_matrix_index,
        normalize_system=general_params.data.normalize_system,
        resolved_input=resolved_input,
    )
    _log_matrix_condition_number(
        _to_numpy(system.matrix),
        matrix_path=paths.matrix,
        display_name=display_name,
    )

    service = PreconditionerService()
    preconditioners = service.create_preconditioner_set(system.matrix, preconditioner_configs)

    scheduled_preconditioners = _create_scheduled_preconditioners(
        preconditioner_configs=preconditioner_configs,
        matrix=system.matrix,
        base_preconditioners=preconditioners,
    )

    _load_and_bind_extra_inputs(
        scheduled_preconditioners,
        matrix=system.matrix,
        matrix_path=paths.matrix,
        matrix_index=resolved_matrix_index,
    )

    cond_callables: dict[str, PreconditionerCallable] = {
        name: p for name, p in scheduled_preconditioners.items()
    }
    cond_numbers = compute_condition_numbers(_to_numpy(system.matrix), cond_callables)

    results = run_cg_comparison(
        system.matrix,
        system.rhs,
        preconditioners=scheduled_preconditioners,
        rtol=general_params.params.rtol,
        atol=general_params.params.atol,
        maxiter=general_params.params.max_iterations,
        m_max=general_params.params.m_max,
        breakdown_tol=general_params.params.breakdown_tol,
    )
    recommendations = ComparisonRecommendations()

    plot_paths = _generate_comparison_plots(
        results,
        cond_numbers,
        paths,
        scheduled_preconditioners,
        display_name=display_name,
        rtol=general_params.params.rtol,
        atol=general_params.params.atol,
        max_iterations=general_params.params.max_iterations,
    )

    return ComparisonResult(
        results=results,
        summary=format_results_summary(results),
        plot_paths=plot_paths,
        preconditioners=tuple(preconditioners.keys()),
        condition_numbers=cond_numbers,
        solver_params=general_params,
        recommendations=recommendations,
        output_dir=paths.output,
    )
