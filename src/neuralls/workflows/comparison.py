"""Backend helpers for comparison workflows.

This module provides orchestration for running CG solver comparisons.
Neural preconditioner checkpoints are resolved from a ``ComparisonRun``
produced by ``train_batch()`` — no ``experiments.toml`` needed here.

Architecture:
    ``run_comparison()`` is the single entry point.  It supports two modes:

    - Pipeline mode (``comparison_run`` provided): resolves neural checkpoint
      references from ``checkpoint_map``; uses ``comparison_run.tracking_uri``.
    - Standalone mode (``comparison_run=None``): neural specs must carry explicit
      ``checkpoint_path``; uses ``solver_cfg.general.comparisons.tracking_uri``.

    Comparison runs are tagged with ``batch_run_id`` for grouping (pipeline mode)
    or ``phase=direct_comparison`` (standalone mode).

Key Functions:
    - ``run_comparison()``: Unified entry point for both modes.
    - ``_resolve_preconditioner()``: Pure: inject checkpoint into a neural solver spec.
    - ``_resolve_neural_preconditioners()``: Pure map over all solver specs.
    - ``_validate_neural_preconditioner()``: Pure validation helper.

Example:
    >>> from neuralls.workflows.comparison import run_comparison
    >>> from neuralls.workflows.comparison_run import load_comparison_run
    >>> comparison_run = load_comparison_run(Path("output/training/comparison_run.json"))
    >>> outcomes = run_comparison(
    ...     Path("configs/solvers/default.toml"),
    ...     ComparisonParams(save_plots=True),
    ...     comparison_run,
    ... )
    >>> print(f"{sum(o.success for o in outcomes)}/{len(outcomes)} comparisons succeeded")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import tomli_w
from loguru import logger

from neuralls.configuration.comparison import ComparisonsTrackingConfig
from neuralls.configuration.preconditioner import PreconditionerConfig
from neuralls.io.toml_loader import load_solver_config
from neuralls.workflows.comparison_run import (
    ComparisonRun,
    _artifact_uri_to_local_path,
    resolve_checkpoint_from_run,
    setup_comparison_tracking,
)
from neuralls.workflows.compare import compare_preconditioners
from neuralls.workflows.specs import ComparisonOutcome, ComparisonParams
from neuralls.workflows.results import ComparisonResult


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _validate_neural_preconditioner(spec: Any) -> None:
    """Pure: validate that a neural solver spec has a resolvable checkpoint source.

    Neural preconditioners must specify either:
    - ``checkpoint_path`` (explicit path), or
    - ``experiment`` (ID to resolve from ``ComparisonRun``).

    Args:
        spec: Solver specification (``SolverSpecConfig`` from solver.toml).

    Raises:
        ValueError: If neural solver lacks both ``checkpoint_path`` and ``experiment``.
    """
    if spec.type != "neural":
        return
    if not spec.checkpoint_path and not spec.experiment:
        raise ValueError(
            f"Neural solver '{spec.name}' must specify "
            "either 'checkpoint_path' or 'experiment'."
        )


def _resolve_preconditioner(
    spec: Any,
    comparison_run: ComparisonRun,
) -> Any:
    """Pure: inject checkpoint_path into a neural solver spec from ComparisonRun.

    Three cases:
    1. Non-neural solver → returned unchanged.
    2. Neural with explicit ``checkpoint_path`` → returned unchanged.
    3. Neural with ``experiment`` ref → checkpoint resolved from ComparisonRun.

    Args:
        spec: Solver specification (``SolverSpecConfig``).
        comparison_run: ComparisonRun produced by ``train_batch()``.

    Returns:
        Resolved solver spec (unchanged or with ``checkpoint_path`` injected).

    Raises:
        ValueError: If neural solver config is invalid.
        KeyError: If experiment ref not found in ``comparison_run.checkpoint_map``.
    """
    if spec.type != "neural":
        return spec
    if spec.checkpoint_path:
        return spec

    _validate_neural_preconditioner(spec)
    checkpoint = resolve_checkpoint_from_run(spec, comparison_run)
    return spec.model_copy(update={"checkpoint_path": checkpoint})


def _resolve_neural_preconditioners(
    solver_specs: list[Any],
    comparison_run: ComparisonRun,
) -> list[PreconditionerConfig]:
    """Pure: resolve checkpoints for all neural solver specs.

    Non-neural and already-resolved specs pass through unchanged.

    Args:
        solver_specs: List of ``SolverSpecConfig`` from solver.toml ``[solvers]``.
        comparison_run: ComparisonRun with checkpoint_map from training phase.

    Returns:
        List of solver specs with all neural ``experiment`` refs resolved to
        ``checkpoint_path`` values.

    Raises:
        ValueError: If any neural spec is misconfigured.
        KeyError: If any experiment ref is missing from ``comparison_run.checkpoint_map``.
    """
    return [_resolve_preconditioner(spec, comparison_run) for spec in solver_specs]


def _save_comparison_toml(result: ComparisonResult, output_path: Path) -> None:
    """Save comparison diagnostics to a TOML file.

    Args:
        result: The comparison result.
        output_path: Path to the output TOML file.
    """
    data: dict[str, Any] = {
        "condition_number": result.condition_numbers,
    }

    # Add other scalar metrics
    iterations: dict[str, int] = {}
    residuals: dict[str, float] = {}
    for name, res in result.results.items():
        iterations[name] = res.iterations
        residuals[name] = res.residual
    
    data["iterations"] = iterations
    data["final_residual"] = residuals

    with open(output_path, "wb") as f:
        tomli_w.dump(data, f)


# ---------------------------------------------------------------------------
# Side-effect orchestration
# ---------------------------------------------------------------------------


def run_comparison(
    solver_config: Path,
    params: ComparisonParams,
    comparison_run: ComparisonRun | None = None,
) -> list[ComparisonOutcome]:
    """Run a preconditioner comparison.

    Two modes depending on whether a ComparisonRun is provided:

    - Pipeline mode (``comparison_run`` provided): resolves neural checkpoint
      references from ``checkpoint_map``; uses ``comparison_run.tracking_uri``.
    - Standalone mode (``comparison_run=None``): neural specs must carry explicit
      ``checkpoint_path``; uses ``solver_cfg.general.comparisons.tracking_uri``.

    Args:
        solver_config: Path to solver TOML (e.g. ``configs/solvers/default.toml``).
        params: Comparison parameters (save_plots, etc.).
        comparison_run: Optional handshake from ``train_batch()``. When ``None``,
            solver_config must have ``[general.comparisons]`` configured and all
            neural specs must have explicit ``checkpoint_path``.

    Returns:
        List with a single ``ComparisonOutcome`` for the solver config.

    Raises:
        ValueError: If standalone mode is used without ``[general.comparisons]``
            or a neural spec is missing ``checkpoint_path``.
        KeyError: If a neural spec references an experiment not in ``checkpoint_map``.
    """
    try:
        solver_cfg = load_solver_config(solver_config)

        if comparison_run is not None:
            resolved_specs = _resolve_neural_preconditioners(solver_cfg.solvers, comparison_run)
            tracking = ComparisonsTrackingConfig(
                tracking_uri=comparison_run.tracking_uri,
                artifact_location=comparison_run.artifact_location,
            )
            tags: dict[str, str] = {
                "phase": "comparison",
                "batch_run_id": comparison_run.mlflow_run_id,
            }
        else:
            for spec in solver_cfg.solvers:
                _validate_neural_preconditioner(spec)
                if spec.type == "neural" and not spec.checkpoint_path:
                    raise ValueError(
                        f"Neural solver '{spec.name}' requires explicit checkpoint_path "
                        "in standalone mode (no ComparisonRun provided)."
                    )
            if solver_cfg.general.comparisons is None:
                raise ValueError(
                    "[general.comparisons] is required in solver config for standalone mode."
                )
            resolved_specs = solver_cfg.solvers
            tracking = solver_cfg.general.comparisons
            tags = {"phase": "direct_comparison"}

        setup_comparison_tracking(tracking.tracking_uri, tracking.artifact_location)
        run_name = f"comparison-{solver_config.stem}"
        with mlflow.start_run(run_name=run_name, tags=tags) as comp_run:
            comp_run_id = comp_run.info.run_id
            output_root = _artifact_uri_to_local_path(mlflow.get_artifact_uri())

            result = compare_preconditioners(
                general_params=solver_cfg.general,
                preconditioner_configs=resolved_specs,
                output_root=output_root,
                save_plots=params.save_plots,
            )

            # Save comparison.toml
            output_root.mkdir(parents=True, exist_ok=True)
            toml_path = output_root / "comparison.toml"
            _save_comparison_toml(result, toml_path)

            # Log all artifacts in output_root to MLflow
            mlflow.log_artifacts(str(output_root))
            
            # Explicitly log the solver config file
            mlflow.log_artifact(str(solver_config))
            
            mlflow.log_param("solver_config", solver_config.stem)
            mlflow.log_param("comp_run_id", comp_run_id)
            best = (result.recommendations or {}).get("best_overall")
            if best:
                mlflow.log_metrics({
                    "best_iterations": float(best.get("iterations", 0)),
                    "best_residual": float(best.get("residual", 0)),
                })

    except (ValueError, RuntimeError, OSError, FileNotFoundError, KeyError) as exc:
        logger.error(f"Comparison failed: {exc}")
        return [ComparisonOutcome(name=solver_config.stem, success=False, error=str(exc))]

    return [ComparisonOutcome(name=solver_config.stem, success=True, payload=result)]
