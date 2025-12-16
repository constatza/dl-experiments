"""Simplified preconditioner comparison workflow driven by solver config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from collections.abc import Callable, Sequence

import numpy as np

from ..constants import REORTHOG_STRICT_THRESHOLD
from ..diagnostics import compute_condition_numbers, plot_condition_numbers
from ..file_operations import ensure_dir
from ..io.comparison import load_system_arrays
from ..plotting import plot_convergence_comparison
from ..preconditioner_factory import (
    BasePreconditionerConfig,
    build_preconditioner_configs,
    build_preconditioners,
    make_ilu_preconditioner,
    make_identity_preconditioner,
    make_jacobi_preconditioner,
)
from ..solver import (
    create_reorthogonalization_strategy,
    format_results_summary,
    run_cg_comparison,
    summarize_best_combinations,
)
from ..validation import validate_matrix, validate_rhs_vector
from ..configuration.solver import GeneralSolverParams


StoppingCriterion = Literal["tolerance", "fixed_iterations"]


def _map_stopping_criterion(name: str) -> StoppingCriterion:
    normalized = name.lower()
    if normalized in {"fixed_iterations", "fixed"}:
        return "fixed_iterations"
    return "tolerance"


def _resolve_paths(
    *,
    general_params: GeneralSolverParams,
    output_root: Path | None,
    figures_root: Path | None,
) -> tuple[Path, Path, Path, Path]:
    if general_params.matrix_path is None or general_params.rhs_path is None:
        raise ValueError("Matrix and RHS must be provided in solver config.")
    matrix_file = Path(general_params.matrix_path)
    rhs_file = Path(general_params.rhs_path)

    output_base = output_root or Path.cwd() / "output"
    figs_base = figures_root or output_base / "figures"
    ensure_dir(output_base)
    ensure_dir(figs_base)
    return matrix_file, rhs_file, output_base, figs_base


def _update_neural_metadata_with_limits(
    metadata: dict[str, NeuralPreconditionerMetadata | None],
    configs: Sequence[BasePreconditionerConfig],
) -> None:
    for cfg in configs:
        if cfg.type != "neural":
            continue
        limit = cfg.limit_iters if cfg.limit_iters >= 0 else None
        meta = metadata.get(cfg.name)
        if meta is not None:
            metadata[cfg.name] = NeuralPreconditionerMetadata(
                residual_iters=meta.residual_iters,
                applied_iters=limit,
            )


def _resolve_fallback_callable(
    name: str, A: np.ndarray, preconditioners: dict[str, Any]
) -> Callable[[np.ndarray], np.ndarray]:
    if name == "identity":
        return make_identity_preconditioner()
    if name == "jacobi":
        return make_jacobi_preconditioner(A)
    if name == "ilu":
        return make_ilu_preconditioner(A)
    if name in preconditioners:
        return preconditioners[name]
    return make_identity_preconditioner()


def compare_preconditioners(
    *,
    general_params: GeneralSolverParams,
    preconditioner_configs: Sequence[BasePreconditionerConfig],
    output_root: Path | None = None,
    figures_root: Path | None = None,
) -> dict[str, Any]:
    """Run CG comparisons using structured solver config (no file I/O here)."""
    matrix_file, rhs_file, output_root, figures_root = _resolve_paths(
        general_params=general_params,
        output_root=output_root,
        figures_root=figures_root,
    )

    A, b = load_system_arrays(matrix_file, rhs_file)
    validate_matrix(A)
    validate_rhs_vector(b, A)

    precond_configs = list(preconditioner_configs) or build_preconditioner_configs(
        [{"name": "none", "type": "none"}]
    )
    if not precond_configs:
        precond_configs = build_preconditioner_configs(
            [{"name": "none", "type": "none"}]
        )
    preconditioners, solver_types = build_preconditioners(A, precond_configs)

    cond_numbers = compute_condition_numbers(A, preconditioners)

    solver_options: dict[str, dict[str, Any]] = {}
    for cfg in precond_configs:
        limit = cfg.limit_iters if cfg.limit_iters >= 0 else None
        solver_options[cfg.name] = {
            "limit_iters": limit,
            "fallback": _resolve_fallback_callable(cfg.fallback, A, preconditioners),
        }

    fallback_precond = make_identity_preconditioner()
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

    suffix = matrix_file.stem or "comparison"
    plot_condition_numbers(cond_numbers, save_dir=figures_root, suffix=suffix)
    convergence_path = figures_root / f"preconditioner_convergence_{suffix}.png"
    plot_convergence_comparison(
        results, metadata=None, save_path=convergence_path
    )

    plot_paths = {"convergence": convergence_path}

    return {
        "results": results,
        "summary": format_results_summary(results),
        "plot_paths": plot_paths,
        "preconditioners": list(preconditioners.keys()),
        "solver_params": general_params,
        "recommendations": recommendations,
    }
