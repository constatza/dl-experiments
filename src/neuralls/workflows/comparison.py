"""Backend helpers for comparison workflows."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mlflow
from loguru import logger

from neuralls.configuration.dataset_identity import resolve_dataset_identity
from neuralls.configuration.comparison import ComparisonConfig
from neuralls.configuration.experiments import (
    ComparisonRegistryEntry,
    ExperimentEntry,
    ExperimentsConfig,
    resolve_display_name,
)
from neuralls.configuration.master_registry import (
    get_experiment_binding,
    load_validated_master_config,
    resolve_comparison_config_path,
)
from neuralls.configuration.mlflow_normalization import build_mlflow_environment
from neuralls.configuration.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerConfig,
    PreconditionerType,
    RegisteredModelRefConfig,
)
from neuralls.io.toml_loader import load_data_config
from neuralls.io.toml_loader import load_comparison_config
from neuralls.workflows.comparison_artifacts import (
    coerce_comparison_result_payload,
    write_comparison_artifacts,
)
from neuralls.workflows.comparison_run import setup_comparison_tracking
from neuralls.workflows.compare import compare_preconditioners
from neuralls.workflows.model_resolution import (
    ExperimentModelContext,
    resolve_preconditioner_models_with_warnings,
)
from neuralls.workflows.results import ComparisonResult
from neuralls.workflows.run_specs import build_comparison_run_spec, build_child_comparison_tags
from neuralls.workflows.specs import ComparisonOutcome, ComparisonParams


@dataclass(frozen=True)
class ComparisonTopology:
    """Shared MLflow topology for comparison execution."""

    tracking_uri: str
    artifact_location: str | None
    experiment_name: str
    model_store_tracking_uri: str


def _validate_neural_preconditioner(spec: Any) -> None:
    """Validate strict neural preconditioner requirements for comparisons."""
    if not isinstance(spec, NeuralPreconditionerConfig):
        return
    if spec.model_ref is None:
        raise ValueError(
            f"Neural preconditioner '{spec.name}' must define model_ref."
        )
    if spec.checkpoint_path is not None:
        raise ValueError(
            f"Neural preconditioner '{spec.name}' cannot use checkpoint_path/experiment "
            "legacy resolution in comparison configs."
        )


def _resolve_preconditioner(
    spec: Any,
) -> Any:
    """Validate one preconditioner and any experiment-bound references."""
    _validate_neural_preconditioner(spec)
    return spec


def _resolve_neural_preconditioners(
    solver_specs: list[Any],
) -> list[PreconditionerConfig]:
    """Validate neural preconditioners and return unchanged specs."""
    return [_resolve_preconditioner(spec) for spec in solver_specs]


def _needs_model_resolution(specs: tuple[PreconditionerConfig, ...]) -> bool:
    """Return True when any neural preconditioner needs model_ref lookup."""
    for spec in specs:
        if not isinstance(spec, NeuralPreconditionerConfig):
            continue
        if spec.model_ref is not None:
            return True
    return False


def _referenced_experiment_ids(
    specs: tuple[PreconditionerConfig, ...],
) -> tuple[str, ...]:
    """Return experiment ids explicitly referenced by neural specs."""
    ids: list[str] = []
    for spec in specs:
        if not isinstance(spec, NeuralPreconditionerConfig):
            continue
        experiment_id = spec.experiment
        if experiment_id is None or experiment_id in ids:
            continue
        ids.append(experiment_id)
    return tuple(ids)


def _build_master_experiment_contexts(
    cfg_path: Path,
    experiment_ids: tuple[str, ...],
) -> dict[str, ExperimentModelContext]:
    """Resolve dataset/model identity for all master-config experiments."""
    master_cfg, config_dir = load_validated_master_config(cfg_path)
    contexts: dict[str, ExperimentModelContext] = {}
    for experiment_id in experiment_ids:
        binding = get_experiment_binding(master_cfg, config_dir, experiment_id)
        data_cfg = load_data_config(binding.data_config_path)
        dataset_id = resolve_dataset_identity(
            data_cfg=data_cfg,
            config_path=binding.data_config_path,
        ).name
        contexts[experiment_id] = ExperimentModelContext(
            dataset_alias=dataset_id,
            model_name=experiment_id,
        )
    return contexts


def _build_experiment_contexts(
    *,
    experiments_config_path: Path,
    experiment_ids: tuple[str, ...],
) -> dict[str, ExperimentModelContext]:
    """Collect per-experiment model resolution context from master config."""
    return _build_master_experiment_contexts(experiments_config_path, experiment_ids)


def _load_master_config(
    experiments_config_path: Path,
) -> tuple[ExperimentsConfig, Path]:
    """Load the master config and its resolution root."""
    return load_validated_master_config(experiments_config_path)


def _existing_experiment_ids(specs: tuple[PreconditionerConfig, ...]) -> set[str]:
    """Return experiment ids already claimed by explicit neural preconditioners."""
    return {
        spec.experiment
        for spec in specs
        if isinstance(spec, NeuralPreconditionerConfig) and spec.experiment is not None
    }


def neural_specs_from_experiments(
    entries: Sequence[ExperimentEntry],
    claimed_ids: set[str],
) -> list[NeuralPreconditionerConfig]:
    """Generate NeuralPreconditionerConfig stubs from experiment entries.

    Args:
        entries: Experiment entries to convert.
        claimed_ids: Experiment ids already covered by explicit preconditioners.

    Returns:
        List of auto-generated neural preconditioner configs.
    """
    return [
        NeuralPreconditionerConfig(
            name=entry.effective_display_name,
            type=PreconditionerType.NEURAL,
            experiment=entry.id,
            model_ref=RegisteredModelRefConfig(latest=True),
        )
        for entry in entries
        if entry.id not in claimed_ids
    ]


def _resolve_specs(
    cfg: ComparisonConfig,
    work_root: Path,
    experiments_config_path: Path,
    model_store_tracking_uri: str,
) -> tuple[list[PreconditionerConfig], tuple[str, ...]]:
    """Resolve model_ref preconditioners into concrete checkpoint paths."""
    specs = _resolve_neural_preconditioners(list(cfg.preconditioners))
    if not _needs_model_resolution(cfg.preconditioners):
        return specs, ()

    experiment_ids = _referenced_experiment_ids(cfg.preconditioners)
    experiment_contexts = None
    if experiment_ids:
        experiment_contexts = _build_experiment_contexts(
            experiments_config_path=experiments_config_path,
            experiment_ids=experiment_ids,
        )
    resolution = resolve_preconditioner_models_with_warnings(
        specs=specs,
        tracking_uri=model_store_tracking_uri,
        download_root=work_root / "models",
        dataset_alias=cfg.general.data.dataset_alias,
        experiment_contexts=experiment_contexts,
        skip_unresolved=True,
    )
    return resolution.specs, resolution.warnings


def _resolve_comparison_topology(
    experiments_config_path: Path,
) -> ComparisonTopology:
    """Resolve comparison MLflow topology from the master config."""
    master_cfg, _ = _load_master_config(experiments_config_path)
    env = build_mlflow_environment(
        tracking_uri=master_cfg.mlflow.tracking_uri,
        artifacts_destination=master_cfg.mlflow.artifacts_destination,
        config_path=experiments_config_path,
    )
    return ComparisonTopology(
        tracking_uri=env["MLFLOW_TRACKING_URI"],
        artifact_location=env.get("MLFLOW_ARTIFACT_URI"),
        experiment_name=master_cfg.names.comparison,
        model_store_tracking_uri=env["MLFLOW_TRACKING_URI"],
    )


def run_comparison(
    comparison_config: Path,
    params: ComparisonParams,
    experiments_config_path: Path,
    comparison_id: str | None = None,
    comparison_display_name: str | None = None,
    experiment_entries: Sequence[ExperimentEntry] | None = None,
) -> list[ComparisonOutcome]:
    """Run a preconditioner comparison from the current config schema."""
    _ = params
    resolved_comparison_id = comparison_id or comparison_config.stem
    resolved_comparison_display_name = resolve_display_name(
        resolved_comparison_id,
        comparison_display_name,
    )
    try:
        cfg = load_comparison_config(comparison_config)
        if experiment_entries:
            claimed_ids = _existing_experiment_ids(cfg.preconditioners)
            auto_specs = neural_specs_from_experiments(experiment_entries, claimed_ids)
            if auto_specs:
                cfg = replace(cfg, preconditioners=cfg.preconditioners + tuple(auto_specs))
        topology = _resolve_comparison_topology(experiments_config_path)
        resolution_warnings: tuple[str, ...] = ()

        setup_comparison_tracking(
            tracking_uri=topology.tracking_uri,
            artifact_location=topology.artifact_location,
            experiment_name=topology.experiment_name,
        )
        comparison_entry = ComparisonRegistryEntry(
            id=resolved_comparison_id,
            path=comparison_config,
            display_name=comparison_display_name,
        )
        run_name, comp_tags = build_comparison_run_spec(entry=comparison_entry)

        with mlflow.start_run(run_name=run_name, tags=comp_tags.as_mlflow_tags()) as comp_run:
            comp_run_id = comp_run.info.run_id
            mlflow.log_param("artifact_uri", mlflow.get_artifact_uri())

            with tempfile.TemporaryDirectory() as _tmp:
                work_root = Path(_tmp)
                resolved_specs, resolution_warnings = _resolve_specs(
                    cfg,
                    work_root,
                    experiments_config_path,
                    topology.model_store_tracking_uri,
                )
                if not resolved_specs:
                    raise ValueError(
                        "No runnable preconditioners remain after model resolution."
                    )
                if resolution_warnings:
                    mlflow.log_param(
                        "skipped_preconditioners",
                        str(len(resolution_warnings)),
                    )
                raw_result = compare_preconditioners(
                    general_params=cfg.general,
                    preconditioner_configs=resolved_specs,
                    output_root=work_root,
                    display_name=resolved_comparison_display_name,
                )
                artifact_source = coerce_comparison_result_payload(raw_result)
                write_comparison_artifacts(
                    result=artifact_source,
                    work_root=work_root,
                    comparison_config=comparison_config,
                )
                mlflow.log_artifacts(str(work_root))

                # Create nested child runs for each preconditioner result
                if isinstance(raw_result, ComparisonResult):
                    for name, entry in raw_result.results.items():
                        child_tags = build_child_comparison_tags(
                            preconditioner_name=name,
                            comparison_id=resolved_comparison_id,
                            parent_run_name=run_name,
                        )
                        with mlflow.start_run(run_name=name, nested=True, tags=child_tags.as_mlflow_tags()):
                            for step, residual in enumerate(entry.residual_history):
                                mlflow.log_metric("residual", residual, step=step)

            mlflow.log_param("comparison_config", comparison_config.stem)
            mlflow.log_param("comparison_id", resolved_comparison_id)
            mlflow.log_param("comparison_display_name", resolved_comparison_display_name)
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
                comparison_id=resolved_comparison_id,
                comparison_display_name=resolved_comparison_display_name,
                success=False,
                error=str(exc),
                warnings=(),
            )
        ]

    payload = raw_result if isinstance(raw_result, ComparisonResult) else None
    return [
        ComparisonOutcome(
            comparison_id=resolved_comparison_id,
            comparison_display_name=resolved_comparison_display_name,
            success=True,
            payload=payload,
            warnings=resolution_warnings,
        )
    ]


def run_comparison_batch(
    experiments_config_path: Path,
    params: ComparisonParams,
) -> list[ComparisonOutcome]:
    """Run all configured comparison profiles from the master config."""
    master_cfg, config_dir = _load_master_config(experiments_config_path)
    if not master_cfg.comparisons:
        raise ValueError(
            "Experiments config must define at least one [[comparisons]] entry."
        )

    outcomes: list[ComparisonOutcome] = []
    for entry in master_cfg.comparisons:
        if entry.experiments:
            experiment_entries: list[ExperimentEntry] = [
                e for e in master_cfg.experiments if e.id in entry.experiments
            ]
        else:
            experiment_entries = list(master_cfg.experiments)
        comparison_path = resolve_comparison_config_path(
            master_cfg,
            config_dir,
            entry.id,
        )
        outcomes.extend(
            run_comparison(
                comparison_config=comparison_path,
                params=params,
                experiments_config_path=experiments_config_path,
                comparison_id=entry.id,
                comparison_display_name=entry.effective_display_name,
                experiment_entries=experiment_entries,
            )
        )
    return outcomes
