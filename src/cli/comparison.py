"""Preconditioner comparison helper shared by CLI and workflows."""

from __future__ import annotations

import glob
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal
from collections.abc import Callable, Sequence

import numpy as np

from ..constants import (
    NOISE_STRATEGY_NONE,
    DEFAULT_NOISE_LEVEL,
    DEFAULT_NOISE_SEED,
    REORTHOG_STRICT_THRESHOLD,
    ConfigKeys,
)
from ..diagnostics import compute_condition_numbers, plot_condition_numbers
from ..file_operations import ensure_dir, sanitize_identifier
from ..configuration.loader import load_data_context
from ..configuration.solver import SolverSpec
from ..io.comparison import load_solver_config, resolve_system_paths, load_system_arrays
from ..solver import (
    run_cg_comparison,
    format_results_summary,
    summarize_best_combinations,
    create_reorthogonalization_strategy,
)
from ..preconditioner_factory import create_all_preconditioners, make_neural_preconditioner
# TODO: Re-implement noise generators
# from ..noise_generators import create_noise_strategy
def create_noise_strategy(*args, **kwargs):
    """Stub for noise strategy - to be re-implemented."""
    return None
from ..plotting import plot_convergence_comparison
from ..validation import validate_matrix, validate_rhs_vector, validate_solver_params


NoiseStrategyName = Literal[
    "none",
    "global",
    "single_dim",
    "blockwise",
    "worst_case",
    "load_redistribution",
    "missing_data",
    "corrupted_data",
    "extreme_magnitude",
]
ReorthogonalizationName = Literal["none", "full", "partial", "selective"]
StoppingCriterionName = Literal["tolerance", "fixed_iterations"]


def _coerce_stopping_criterion(value: str) -> StoppingCriterionName:
    if value in ("tolerance", "fixed_iterations"):
        return value
    return "tolerance"


def compare_preconditioners(
    *,
    config_path: str | Path | None,
    data_config_path: str | Path | None = None,
    matrix_path: str | Path | None = None,
    rhs_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    pca_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    solver_config_path: str | Path | None = None,
    noise_strategy: NoiseStrategyName = NOISE_STRATEGY_NONE,
    noise_level: float = DEFAULT_NOISE_LEVEL,
    noise_seed: int | None = DEFAULT_NOISE_SEED,
    save_plots: bool = True,
    figures_dir: str | Path | None = None,
    breakdown_tol: float | None = None,
    neural_precond_iters: int | None = None,
    fallback_preconditioner: str = "identity",
    precond_every: int = 1,
    precond_first_n: int | None = None,
    reorthogonalize: ReorthogonalizationName = "full",
    reorthog_window: int = 10,
    reorthog_threshold: float = REORTHOG_STRICT_THRESHOLD,
    custom_combinations: Sequence[tuple[str, str, str]] | None = None,
    ) -> dict[str, Any]:
        """Compare preconditioner, warm-start, and helper strategies."""
        # Optional data context (no model config required)
        context, data_config, data_config_path_resolved = load_data_context(data_config_path)
        if context is None:
            raise ValueError("Data config is required to locate normalized.npz for comparison.")
        general_params, solver_entries = load_solver_config(solver_config_path)

        config_matrix = Path(general_params.matrix_path) if general_params.matrix_path else None
        config_rhs = Path(general_params.rhs_path) if general_params.rhs_path else None
        supplied_matrix = Path(matrix_path) if matrix_path is not None else None
        supplied_rhs = Path(rhs_path) if rhs_path is not None else None

        checkpoint_file = checkpoint_path
        # Resolve output root
        if output_dir is not None:
            output_path = Path(output_dir)
        elif context is not None:
            output_path = Path(context.flow.output_root)
        else:
            output_path = Path.cwd()

        dataset_id = (
            getattr(context.data, "dataset_id", None)
            if context is not None
            else "manual"
        )
        dataset_slug = sanitize_identifier(str(dataset_id))

        model_slug = "manual"
        suffix = f"{dataset_slug}-{model_slug}"

        ensure_dir(output_path)

        matrix_file, rhs_file = resolve_system_paths(
            matrix_path=supplied_matrix,
            rhs_path=supplied_rhs,
            config_matrix=config_matrix,
            config_rhs=config_rhs,
        )
        A, b = load_system_arrays(matrix_file, rhs_file)
        if b is None and context is not None and context.test_solutions_path is not None and config_rhs is None and supplied_rhs is None:
            print(f"Loading test solutions from: {context.test_solutions_path}")
            solution_files = sorted(glob.glob(str(context.test_solutions_path)))
            if not solution_files:
                raise FileNotFoundError(
                    f"No test solution files found: {context.test_solutions_path}"
                )
            x_test = np.loadtxt(solution_files[0], dtype=np.float64)
            if x_test.ndim == 1:
                x_test = x_test.reshape(-1, 1)
            b = A @ x_test.flatten()
            print(f"✓ Recomputed RHS from test solution: {solution_files[0]}")

        validate_matrix(A)
        validate_rhs_vector(b, A)
        print("Loaded pre-normalized data artifacts")
        try:
            cond_num = float(np.linalg.cond(A))
            print(f"Matrix condition number (2-norm): {cond_num:.3e}")
        except Exception as exc:  # noqa: BLE001
            print(f"Matrix condition number unavailable: {exc}")

        if noise_strategy != NOISE_STRATEGY_NONE:
            noise_fn = create_noise_strategy(
                noise_strategy, noise_level, seed=noise_seed, A=A
            )
            b = noise_fn(b)

        rtol = general_params.rtol
        atol = general_params.atol
        max_iter = general_params.max_iterations
        stopping_criterion = _coerce_stopping_criterion(general_params.stopping_criterion)
        validate_solver_params(rtol, atol, max_iter, stopping_criterion)

        preconditioners, preconditioner_metadata = create_all_preconditioners(
            A,
            checkpoint_path=checkpoint_file,
            config_path=config_path if checkpoint_file else None,
            data_config_path=data_config_path if checkpoint_file else None,
            pca_path=pca_path,
        )
        # Select solvers from config (names are display labels)
        solver_types: dict[str, str] = {}
        solver_options: dict[str, dict[str, Any]] = {}
        selected_preconditioners: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
        if not solver_entries:
            solver_entries = [SolverSpec(name="none", type="none", args={})]
        for entry in solver_entries:
            solver_types[entry.name] = entry.type
            args = dict(entry.args)
            opts: dict[str, Any] = {}
            if "limit_iters" in args:
                opts["limit_iters"] = args.pop("limit_iters")
            if "apply_every" in args:
                opts["apply_every"] = args.pop("apply_every")
            if "first_n" in args:
                opts["first_n"] = args.pop("first_n")
            if "fallback" in args:
                opts["fallback"] = args.pop("fallback")
            solver_options[entry.name] = opts

            if entry.type == "none":
                # Baseline CG without preconditioning
                selected_preconditioners[entry.name] = lambda residual: residual
                continue
            if entry.type == "neural" and "checkpoint_path" in args:
                ckpt_path = Path(args.pop("checkpoint_path"))
                cfg_path_override = args.pop("config_path", None)
                data_cfg_override = args.pop("data_config_path", None)
                precond_fn, meta = make_neural_preconditioner(
                    A,
                    ckpt_path,
                    cfg_path_override or config_path,
                    data_cfg_override or data_config_path,
                )
                selected_preconditioners[entry.name] = precond_fn
                preconditioner_metadata[entry.name] = meta
                continue
            if entry.type not in preconditioners:
                raise ValueError(f"Unknown solver/preconditioner type '{entry.type}' requested in solver config.")
            selected_preconditioners[entry.name] = preconditioners[entry.type]

        cond_numbers = compute_condition_numbers(A, preconditioners)

        def _infer_neural_iters_from_data_config() -> int | None:
            if data_config_path is None:
                return None
            try:
                with Path(data_config_path).open("rb") as f:
                    config = tomllib.load(f)
                strategies = config.get("generation", {}).get("strategy", [])
                candidates: list[int] = []
                for strategy in strategies:
                    if not isinstance(strategy, dict):
                        continue
                    if "residual_iters" in strategy:
                        candidates.append(int(strategy["residual_iters"]))
                    if (
                        hasattr(ConfigKeys, "RESIDUAL_ITERS")
                        and ConfigKeys.RESIDUAL_ITERS in strategy
                    ):
                        candidates.append(int(strategy[ConfigKeys.RESIDUAL_ITERS]))

                generation = config.get("generation", {})
                if isinstance(generation, dict):
                    if (
                        hasattr(ConfigKeys, "RESIDUAL_ITERS")
                        and ConfigKeys.RESIDUAL_ITERS in generation
                    ):
                        candidates.append(int(generation[ConfigKeys.RESIDUAL_ITERS]))
                    if "residual_iters" in generation:
                        candidates.append(int(generation["residual_iters"]))

                return min(candidates) if candidates else None
            except Exception:
                return None

        neural_iters_to_apply: int | None = None
        if "neural" in preconditioner_metadata:
            neural_meta = preconditioner_metadata["neural"]
            if neural_meta is not None:
                inferred_iters = (
                    neural_precond_iters
                    or neural_meta.residual_iters
                    or _infer_neural_iters_from_data_config()
                )
                neural_iters_to_apply = inferred_iters
                preconditioner_metadata["neural"] = replace(
                    neural_meta, applied_iters=neural_iters_to_apply
                )
        elif neural_precond_iters is not None:
            neural_iters_to_apply = neural_precond_iters

        if neural_iters_to_apply is None and "neural" in preconditioners:
            print(
                "Warning: No residual_iters found in data config; neural preconditioner will run for all iterations."
            )

        fallback_precond = None
        if neural_iters_to_apply is not None:
            if fallback_preconditioner == "identity":
                from ..preconditioner_factory import make_identity_preconditioner

                fallback_precond = make_identity_preconditioner()
            elif fallback_preconditioner == "jacobi":
                from ..preconditioner_factory import make_jacobi_preconditioner

                fallback_precond = make_jacobi_preconditioner(A)
            elif fallback_preconditioner == "ilu":
                from ..preconditioner_factory import make_ilu_preconditioner

                fallback_precond = make_ilu_preconditioner(A)
            else:
                raise ValueError(
                    f"Unknown fallback preconditioner: {fallback_preconditioner}"
                )

        def build_preconditioner_combinations() -> list[tuple[str, str, str]]:
            combos: list[tuple[str, str, str]] = []
            for precond_name in selected_preconditioners.keys():
                combos.append(("none", precond_name, "none"))
            return combos

        combination_plan = build_preconditioner_combinations()
        if custom_combinations:
            for combo in custom_combinations:
                if combo not in combination_plan:
                    combination_plan.append(combo)

        effective_breakdown_tol = breakdown_tol if breakdown_tol is not None else 0.0

        reorthog_strategy = create_reorthogonalization_strategy(
            reorthogonalize,
            A=A,
            window_size=reorthog_window,
            threshold=reorthog_threshold,
        )

        results = run_cg_comparison(
            A,
            b,
            preconditioners=selected_preconditioners,
            rtol=rtol,
            atol=atol,
            max_iter=max_iter,
            stopping_criterion=stopping_criterion,
            breakdown_tol=effective_breakdown_tol,
            precond_iters=neural_iters_to_apply,
            fallback_preconditioner=fallback_precond,
            precond_every=precond_every,
            precond_first_n=precond_first_n,
            reorthogonalize=reorthog_strategy,
            combination_plan=combination_plan,
            limited_preconditioner="neural" if neural_iters_to_apply is not None else None,
            solver_types=solver_types,
            solver_options=solver_options,
        )

        recommendations = summarize_best_combinations(results)

        plot_paths: dict[str, Path] = {}
        if save_plots:
            try:
                figures_root = (
                    Path(figures_dir)
                    if figures_dir is not None
                    else Path(output_path) / "figures"
                )
                figures_root.mkdir(parents=True, exist_ok=True)

                spectra_suffix = suffix
                plot_condition_numbers(cond_numbers, save_dir=figures_root, suffix=spectra_suffix)

                convergence_path = figures_root / f"preconditioner_convergence_{suffix}.png"
                plot_convergence_comparison(
                    results, metadata=preconditioner_metadata, save_path=convergence_path
                )
                plot_paths["convergence"] = convergence_path
            except Exception as exc:  # noqa: BLE001
                from loguru import logger
                logger.warning(f"Could not generate plots: {exc}")
                logger.exception("Full traceback:")
        else:
            plot_condition_numbers(cond_numbers)

        return {
            "results": results,
            "summary": format_results_summary(results),
            "plot_paths": plot_paths,
            "preconditioners": list(preconditioners.keys()),
            "preconditioner_metadata": preconditioner_metadata,
            "solver_params": general_params,
            "recommendations": recommendations,
        }
