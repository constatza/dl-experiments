"""Backend helpers for comparison workflows."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import mlflow
from loguru import logger

from neuralls.configuration.comparison import ComparisonConfig
from neuralls.configuration.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerConfig,
)
from neuralls.io.toml_loader import load_comparison_config
from neuralls.workflows.comparison_artifacts import (
    coerce_comparison_result_payload,
    write_comparison_artifacts,
)
from neuralls.workflows.comparison_run import (
    ComparisonRun,
    _artifact_uri_to_local_path,
    setup_comparison_tracking,
)
from neuralls.workflows.compare import compare_preconditioners
from neuralls.workflows.model_resolution import resolve_preconditioner_models
from neuralls.workflows.results import ComparisonResult
from neuralls.workflows.specs import ComparisonOutcome, ComparisonParams


def _validate_neural_preconditioner(spec: Any) -> None:
    """Validate strict neural preconditioner requirements for schema_version=3."""
    if not isinstance(spec, NeuralPreconditionerConfig):
        return
    if spec.model_ref is None:
        raise ValueError(
            f"Neural preconditioner '{spec.name}' must define model_ref."
        )
    if spec.checkpoint_path is not None or spec.experiment is not None:
        raise ValueError(
            f"Neural preconditioner '{spec.name}' cannot use checkpoint_path/experiment "
            "in schema_version=3."
        )


def _resolve_preconditioner(
    spec: Any,
    comparison_run: ComparisonRun,  # noqa: ARG001
) -> Any:
    """Strict schema keeps model resolution in model_ref path (no checkpoint-map rewrite)."""
    _validate_neural_preconditioner(spec)
    return spec


def _resolve_neural_preconditioners(
    solver_specs: list[Any],
    comparison_run: ComparisonRun,  # noqa: ARG001
) -> list[PreconditionerConfig]:
    """Validate neural preconditioners and return unchanged specs."""
    return [_resolve_preconditioner(spec, comparison_run) for spec in solver_specs]


def _needs_model_resolution(specs: tuple[PreconditionerConfig, ...]) -> bool:
    """Return True when any neural preconditioner needs model_ref lookup."""
    for spec in specs:
        if not isinstance(spec, NeuralPreconditionerConfig):
            continue
        if spec.model_ref is not None:
            return True
    return False


def _resolve_tracking(
    cfg: ComparisonConfig,
    comparison_run: ComparisonRun | None,
) -> tuple[str, str, str, dict[str, str]]:
    """Resolve comparison tracking coordinates and run tags."""
    if cfg.general.tracking is None:
        raise ValueError("general.tracking is required for comparison runs.")
    tags: dict[str, str] = {"phase": "comparison"}
    if comparison_run is not None:
        tags["batch_run_id"] = comparison_run.mlflow_run_id
    return (
        cfg.general.tracking.tracking_uri,
        str(cfg.general.tracking.artifact_location),
        cfg.general.tracking.experiment_name,
        tags,
    )


def _resolve_specs(
    cfg: ComparisonConfig,
    work_root: Path,
) -> list[PreconditionerConfig]:
    """Resolve model_ref preconditioners into concrete checkpoint paths."""
    specs = list(cfg.preconditioners)
    if not _needs_model_resolution(cfg.preconditioners):
        return specs

    model_store = cfg.general.model_store
    if model_store is None:
        raise ValueError(
            "general.model_store.tracking_uri is required when neural preconditioners use model_ref."
        )
    return resolve_preconditioner_models(
        specs=specs,
        tracking_uri=model_store.tracking_uri,
        download_root=work_root / "models",
        dataset_alias=cfg.general.data.dataset_alias,
    )


def run_comparison(
    comparison_config: Path,
    params: ComparisonParams,
    comparison_run: ComparisonRun | None = None,
    experiments_config_path: Path | None = None,
) -> list[ComparisonOutcome]:
    """Run a preconditioner comparison from a schema_version=3 config."""
    try:
        if experiments_config_path is not None:
            raise ValueError(
                "experiments_config_path is not supported in comparison schema_version=3."
            )

        cfg = load_comparison_config(comparison_config)
        tracking_uri, artifact_location, experiment_name, tags = _resolve_tracking(
            cfg, comparison_run
        )

        setup_comparison_tracking(
            tracking_uri=tracking_uri,
            artifact_location=artifact_location,
            experiment_name=experiment_name,
        )
        run_name = cfg.run_name or f"comparison-{comparison_config.stem}"

        with mlflow.start_run(run_name=run_name, tags=tags) as comp_run:
            comp_run_id = comp_run.info.run_id
            _ = _artifact_uri_to_local_path(mlflow.get_artifact_uri())

            with tempfile.TemporaryDirectory() as _tmp:
                work_root = Path(_tmp)
                resolved_specs = _resolve_specs(cfg, work_root)
                raw_result = compare_preconditioners(
                    general_params=cfg.general,
                    preconditioner_configs=resolved_specs,
                    output_root=work_root,
                    save_plots=params.save_plots,
                )
                artifact_source = coerce_comparison_result_payload(raw_result)
                write_comparison_artifacts(
                    result=artifact_source,
                    work_root=work_root,
                    comparison_config=comparison_config,
                )
                mlflow.log_artifacts(str(work_root))

            mlflow.log_param("comparison_config", comparison_config.stem)
            mlflow.log_param("comp_run_id", comp_run_id)
            best = artifact_source.recommendations.overall_best
            if best is not None:
                mlflow.log_metrics(
                    {
                        "best_iterations": float(best.iterations),
                        "best_residual": float(best.residual),
                    }
                )

    except (ValueError, RuntimeError, OSError, FileNotFoundError, KeyError) as exc:
        logger.error(f"Comparison failed: {exc}")
        return [
            ComparisonOutcome(
                name=comparison_config.stem, success=False, error=str(exc)
            )
        ]

    payload = raw_result if isinstance(raw_result, ComparisonResult) else None
    return [ComparisonOutcome(name=comparison_config.stem, success=True, payload=payload)]
