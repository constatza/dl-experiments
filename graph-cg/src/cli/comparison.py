"""Preconditioner comparison helper shared by CLI and workflows."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..constants import (
    NOISE_STRATEGY_NONE,
    DEFAULT_NOISE_LEVEL,
    DEFAULT_NOISE_SEED,
)
from ..common import (
    DEFAULT_RESULTS_DIR,
    ensure_dir,
    get_paths_from_config,
    get_solver_params,
    load_config_with_context,
    load_system_data,
    derive_model_identifier,
    sanitize_identifier,
)
from ..experiment_manifest import update_manifest
from ..cg_algorithms import (
    run_cg_comparison,
    format_results_summary,
    summarize_best_combinations,
)
from ..preconditioner_factory import create_all_preconditioners
from ..noise_generators import create_noise_strategy
from ..plotting import plot_convergence_comparison
from ..validation import validate_matrix, validate_rhs, validate_solver_params


def compare_preconditioners(
    *,
    config_path: str | Path,
    data_config_path: str | Path | None = None,
    matrix_path: str | Path | None = None,
    rhs_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    warm_start_checkpoint_path: str | Path | None = None,
    step_helper_checkpoint_path: str | Path | None = None,
    pca_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    noise_strategy: str = NOISE_STRATEGY_NONE,
    noise_level: float = DEFAULT_NOISE_LEVEL,
    noise_seed: int | None = DEFAULT_NOISE_SEED,
    save_plots: bool = True,
    figures_dir: str | Path | None = None,
    breakdown_tol: float | None = None,
    custom_combinations: Sequence[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """Compare preconditioner, warm-start, and helper strategies."""

    settings, context = load_config_with_context(config_path, data_config_path)
    paths = get_paths_from_config(settings, context)
    solver_params = get_solver_params(settings)

    matrix_file = matrix_path or (context.test_matrix if context.test_matrix else paths["matrix_path"])
    rhs_file = rhs_path or (context.test_rhs if context.test_rhs else paths["rhs_path"])
    checkpoint_file = checkpoint_path or paths.get("checkpoint_path")
    output_path = Path(output_dir or paths.get("results_dir", DEFAULT_RESULTS_DIR))

    dataset_id = getattr(getattr(context, "data", None), "dataset_id", None)
    if dataset_id is None and data_config_path is not None:
        dataset_id = Path(data_config_path).stem
    dataset_slug = sanitize_identifier(str(dataset_id or "dataset"))

    model_id = derive_model_identifier(settings, context, config_path)
    model_slug = sanitize_identifier(model_id)
    suffix = f"{dataset_slug}-{model_slug}"

    if matrix_file is None or rhs_file is None:
        raise ValueError("Matrix and RHS paths must be specified")

    ensure_dir(output_path)

    A, b = load_system_data(matrix_file, rhs_file)
    validate_matrix(A)
    validate_rhs(b, A)

    if context.test_solutions_path is not None:
        print(f"Loading test solutions from: {context.test_solutions_path}")
        solution_files = sorted(glob.glob(str(context.test_solutions_path)))
        if not solution_files:
            raise FileNotFoundError(f"No test solution files found: {context.test_solutions_path}")

        x_test = np.loadtxt(solution_files[0])
        if x_test.ndim == 1:
            x_test = x_test.reshape(-1, 1)

        b = A @ x_test.flatten()

        print("✓ Computed test RHS from solution (b_test = A @ x_test)")
        print(f"  Test solution shape: {x_test.shape}")
        print(f"  Test RHS shape: {b.shape}")

    print("Loaded pre-normalized data artifacts")

    if noise_strategy != NOISE_STRATEGY_NONE:
        noise_fn = create_noise_strategy(
            noise_strategy, noise_level, seed=noise_seed, A=A
        )
        b = noise_fn(b)

    tol = solver_params["tolerance"]
    max_iter = solver_params["max_iterations"]
    stopping_criterion = solver_params["stopping_criterion"]
    validate_solver_params(tol, max_iter, stopping_criterion)

    preconditioners, warm_starts, step_helpers = create_all_preconditioners(
        A,
        checkpoint_path=checkpoint_file,
        config_path=config_path if checkpoint_file else None,
        data_config_path=data_config_path if checkpoint_file else None,
        warm_start_checkpoint_path=warm_start_checkpoint_path,
        warm_start_config_path=config_path if (warm_start_checkpoint_path or checkpoint_file) else None,
        warm_start_data_config_path=data_config_path if (warm_start_checkpoint_path or checkpoint_file) else None,
        step_helper_checkpoint_path=step_helper_checkpoint_path,
        step_helper_config_path=config_path if (step_helper_checkpoint_path or checkpoint_file) else None,
        step_helper_data_config_path=data_config_path if (step_helper_checkpoint_path or checkpoint_file) else None,
        pca_path=pca_path,
    )

    def build_default_combinations() -> list[tuple[str, str, str]]:
        combos: list[tuple[str, str, str]] = []
        warm_names = list(warm_starts.keys())

        for warm_name in warm_names:
            combos.append((warm_name, "none", "none"))

        for classical in ("jacobi", "ilu"):
            if classical in preconditioners:
                combos.append(("none", classical, "none"))

        for precond_name in preconditioners.keys():
            if precond_name in {"none", "jacobi", "ilu"}:
                continue
            combos.append(("none", precond_name, "none"))
            for warm_name in warm_names:
                if warm_name == "none":
                    continue
                combos.append((warm_name, precond_name, "none"))

        deduped: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for combo in combos:
            if combo not in seen:
                deduped.append(combo)
                seen.add(combo)
        return deduped

    combination_plan = build_default_combinations()
    if custom_combinations:
        for combo in custom_combinations:
            if combo not in combination_plan:
                combination_plan.append(combo)

    effective_breakdown_tol = breakdown_tol if breakdown_tol is not None else tol * 1e-4

    results = run_cg_comparison(
        A,
        b,
        preconditioners=preconditioners,
        warm_starts=warm_starts,
        step_helpers=step_helpers,
        tol=tol,
        max_iter=max_iter,
        stopping_criterion=stopping_criterion,
        breakdown_tol=effective_breakdown_tol,
        combination_plan=combination_plan,
    )

    recommendations = summarize_best_combinations(results)

    plot_paths: dict[str, Path] = {}
    if save_plots:
        from ..common import DEFAULT_FIGURES_DIR

        try:
            figures_root = Path(figures_dir) if figures_dir is not None else Path(
                paths.get("figures_dir", DEFAULT_FIGURES_DIR)
            )
            figures_root.mkdir(parents=True, exist_ok=True)

            convergence_path = figures_root / f"preconditioner_convergence_{suffix}.png"
            plot_convergence_comparison(results, save_path=convergence_path)
            plot_paths["convergence"] = convergence_path
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Could not generate plots: {exc}")

    if checkpoint_file is not None:
        checkpoint_path_obj = Path(checkpoint_file)
        if checkpoint_path_obj.parent.name == "checkpoints":
            experiment_dir = checkpoint_path_obj.parent.parent
            update_manifest(
                experiment_dir,
                "comparison",
                {
                    "checkpoint_path": str(checkpoint_path_obj.relative_to(experiment_dir)),
                },
            )

    return {
        "results": results,
        "summary": format_results_summary(results),
        "plot_paths": plot_paths,
        "preconditioners": list(preconditioners.keys()),
        "warm_starts": list(warm_starts.keys()),
        "step_helpers": list(step_helpers.keys()),
        "solver_params": solver_params,
        "recommendations": recommendations,
    }
