"""Simplified preconditioner comparison workflow driven by solver config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from collections.abc import Callable, Sequence

import numpy as np

from neuralls.configuration.preconditioner import PreconditionerConfig
from neuralls.configuration.comparison import GeneralSolverConfig
from ..constants import REORTHOG_STRICT_THRESHOLD
from ..diagnostics import compute_condition_numbers, plot_condition_numbers
from ..file_operations import ensure_dir
from ..io.comparison import load_system_arrays
from ..plotting import plot_convergence_comparison
from ..preconditioner import create_default_registry
from ..solver import (
    create_reorthogonalization_strategy,
    format_results_summary,
    run_cg_comparison,
    summarize_best_combinations,
)
from ..validation import validate_matrix, validate_rhs_vector
from .results import ComparisonResult


StoppingCriterion = Literal["tolerance", "fixed_iterations"]


def _map_stopping_criterion(name: str) -> StoppingCriterion:
    normalized = name.lower()
    if normalized in {"fixed_iterations", "fixed"}:
        return "fixed_iterations"
    return "tolerance"


def _resolve_paths(
    *,
    general_params: GeneralSolverConfig,
    output_root: Path | None,
    figures_root: Path | None,
) -> tuple[Path, Path, Path, Path]:
    if general_params.matrix_path is None or general_params.rhs_path is None:
        raise ValueError("Matrix and RHS must be provided in solver config.")
    matrix_file = Path(general_params.matrix_path)
    rhs_file = Path(general_params.rhs_path)

    base_root = output_root or getattr(general_params, "output_root", None)
    if base_root is None:
        raise ValueError("output_root must be set in solver general config.")
    output_base = Path(base_root).expanduser().resolve()
    figs_base = figures_root or output_base / "figures"
    ensure_dir(output_base)
    ensure_dir(figs_base)
    return matrix_file, rhs_file, output_base, figs_base


def _resolve_fallback_callable(
    name: str, A: np.ndarray, preconditioners: dict[str, Any]
) -> Callable[[np.ndarray], np.ndarray]:
    """Resolve fallback preconditioner by name.

    This uses the registry pattern to create preconditioners on-demand,
    eliminating code duplication from the old factory approach.
    """
    # Check if already created
    if name in preconditioners:
        return preconditioners[name]

    # Create on-demand using registry
    from neuralls.configuration.preconditioner import StandardPreconditionerConfig

    registry = create_default_registry()
    config = StandardPreconditionerConfig(name=name, type=name)
    return registry.create(A, config)


def compare_preconditioners(
    *,
    general_params: GeneralSolverConfig,
    preconditioner_configs: Sequence[PreconditionerConfig],
    output_root: Path | None = None,
    figures_root: Path | None = None,
    save_plots: bool = True,
) -> ComparisonResult:
    """Run CG comparisons once per solver config and emit shared diagnostics."""
    if not preconditioner_configs:
        raise ValueError("At least one preconditioner config must be provided.")

    matrix_file, rhs_file, output_root, figures_root = _resolve_paths(
        general_params=general_params,
        output_root=output_root,
        figures_root=figures_root,
    )

    A, b = load_system_arrays(matrix_file, rhs_file)
    validate_matrix(A)
    validate_rhs_vector(b, A)

    # Create preconditioners using registry pattern
    registry = create_default_registry()
    preconditioners = {cfg.name: registry.create(A, cfg) for cfg in preconditioner_configs}
    solver_types = {cfg.name: cfg.type for cfg in preconditioner_configs}

    cond_numbers = compute_condition_numbers(A, preconditioners)

    solver_options: dict[str, dict[str, Any]] = {}
    for cfg in preconditioner_configs:
        limit = cfg.limit_iters if cfg.limit_iters >= 0 else None
        solver_options[cfg.name] = {
            "limit_iters": limit,
            "fallback": _resolve_fallback_callable(cfg.fallback, A, preconditioners),
        }

    # Create fallback identity preconditioner using registry
    from neuralls.configuration.preconditioner import StandardPreconditionerConfig

    fallback_config = StandardPreconditionerConfig(name="identity", type="identity")
    fallback_precond = registry.create(A, fallback_config)
    stopping_criterion = _map_stopping_criterion(general_params.stopping_criterion)
    reorthogonalize = create_reorthogonalization_strategy(
        "full",
        A=A,
        window_size=10,
        threshold=REORTHOG_STRICT_THRESHOLD,
    )

    results = run_cg_comparison(
        A,
        b,
        preconditioners=preconditioners,
        rtol=general_params.rtol,
        atol=general_params.atol,
        max_iter=general_params.max_iterations,
        stopping_criterion=stopping_criterion,
        breakdown_tol=0.0,
        precond_iters=None,
        fallback_preconditioner=fallback_precond,
        precond_every=1,
        precond_first_n=None,
        reorthogonalize=reorthogonalize,
        combination_plan=None,
        limited_preconditioner=None,
        solver_types=solver_types,
        solver_options=solver_options,
    )

    recommendations = summarize_best_combinations(results)

    plot_paths: dict[str, Path] = {}
    if save_plots:
        suffix = matrix_file.stem or "comparison"
        plot_condition_numbers(cond_numbers, save_dir=figures_root, suffix=suffix)
        convergence_path = figures_root / f"preconditioner_convergence_{suffix}.png"
        plot_convergence_comparison(results, metadata=None, save_path=convergence_path)
        plot_paths = {"convergence": convergence_path}

    return ComparisonResult(
        results=results,
        summary=format_results_summary(results),
        plot_paths=plot_paths,
        preconditioners=list(preconditioners.keys()),
        solver_params=general_params,
        recommendations=recommendations,
    )
