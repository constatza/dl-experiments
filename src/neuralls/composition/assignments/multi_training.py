"""Batch training workflow: train multiple (model, dataset) pairs and aggregate metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dlkit.common import ChildFailure, ChildSuccess
from dlkit.engine.workflows.multi_run import MultiRunSpec, RunSpec
from dlkit.interfaces.api import run_multirun_spec
from loguru import logger
from mlflow.tracking import MlflowClient

from neuralls.composition.assignments._registry_lookup import (
    _find_registry_entry,
    _resolve_config_paths,
)
from neuralls.composition.assignments.training import (
    PreparedTraining,
    cleanup_prepared_training,
    finalize_prepared_training,
    prepare_training_settings,
    to_run_spec,
)
from neuralls.composition.tracking.run_specs import (
    build_registration_tags,
    build_session_run_spec,
)
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.config.models.dataset_identity import resolve_dataset_identity
from neuralls.platform.config.models.experiments import (
    AssignmentEntry,
    CaseConfig,
    resolve_display_name,
)
from neuralls.platform.config.resolution import (
    derive_output_root_from_tracking_uri,
    is_sqlite_tracking_uri,
)
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.reporting.plots import plot_metric_comparison
from neuralls.platform.tracking.environment import scoped_mlflow_environment
from neuralls.platform.tracking.mlflow import (
    build_workflow_environment,
    finalize_session_parent_run,
)
from neuralls.platform.tracking.mlflow_client import fetch_mlflow_metrics
from neuralls.platform.tracking.model_registry import read_model_class_name


@dataclass(frozen=True)
class TrainingRunResult:
    """Immutable result for a single trained assignment.

    Attributes:
        label: Short numeric label ("1", "2", "3" …) for axis display.
        assignment_id: Stable assignment identifier.
        assignment_display_name: Human-facing assignment label.
        mlflow_run_id: MLflow run UUID the checkpoint was uploaded to.
        metrics: All eval/* metrics returned by MLflow for this run.
    """

    label: str
    assignment_id: str
    assignment_display_name: str
    mlflow_run_id: str
    metrics: dict[str, float]
    resolved_dataset_id: str | None = None
    dataset_display_name: str | None = None
    model_class: str | None = None
    job_display_name: str | None = None
    dataset_registry_id: str | None = None
    job_registry_id: str | None = None
    job_config_path: Path | None = None
    data_config_path: Path | None = None


@dataclass(frozen=True)
class BatchResult:
    """Immutable result for a completed batch of training runs.

    Attributes:
        results: Ordered list of per-assignment results (same order as TOML).
        label_map: Maps short label → assignment metadata.
        output_dir: Directory where batch artefacts (plot, label JSON) are saved.
    """

    results: list[TrainingRunResult]
    label_map: dict[str, dict[str, str | None]]
    output_dir: Path


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _make_label_map(
    results: list[TrainingRunResult],
) -> dict[str, dict[str, str | None]]:
    """Build a mapping from short label to full assignment identity.

    Args:
        results: List of completed training run results.

    Returns:
        Dict ``{label: {"assignment_id": …, "mlflow_run_id": …}}``.
    """
    return {
        r.label: {
            "assignment_id": r.assignment_id,
            "assignment_display_name": r.assignment_display_name,
            "mlflow_run_id": r.mlflow_run_id,
        }
        for r in results
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _annotate_mlflow_run(
    *,
    label: str,
    run_id: str,
    tracking_uri: str,
    job_config_path: Path,
    assignment_id: str,
    assignment_display_name: str,
    resolved_dataset_id: str,
    dataset_display_name: str,
    dataset_registry_id: str | None,
    job_registry_id: str | None,
    job_display_name: str,
) -> None:
    """Log batch params and tag one completed training run for later lookup.

    The checkpoint itself is already uploaded to this run by train_model() —
    this only adds batch-level bookkeeping params and identifying tags
    (assignment_id, dataset_id, job_id, ...) so the run can later be found by
    tag-filtered lookup (see ``LoggedModelRefConfig``). No model registration
    happens here — the registry is reserved for deliberate/manual promotion.

    Args:
        label: Short numeric label for log messages.
        run_id: MLflow run UUID for the completed training run.
        tracking_uri: MLflow tracking server URI.
        job_config_path: Path to job config (provides [model].class as model_class tag).
        assignment_id: Assignment identifier tagged on the run for later lookup.
    """
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        client.log_param(run_id, "batch_label", label)
        client.log_param(run_id, "assignment_id", assignment_id)
        client.log_param(run_id, "assignment_display_name", assignment_display_name)
        client.log_param(run_id, "dataset_id", resolved_dataset_id)
        client.log_param(run_id, "dataset_display_name", dataset_display_name)
        client.log_param(run_id, "job_display_name", job_display_name)
        if dataset_registry_id is not None:
            client.log_param(run_id, "dataset_registry_id", dataset_registry_id)
        if job_registry_id is not None:
            client.log_param(run_id, "job_registry_id", job_registry_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[{label}] Could not log params to MLflow: {exc}")

    model_class = read_model_class_name(job_config_path)
    entry = AssignmentEntry(
        id=assignment_id,
        dataset=dataset_registry_id or resolved_dataset_id,
        job=job_registry_id or assignment_id,
        display_name=assignment_display_name,
    )
    reg_tags = build_registration_tags(
        entry=entry,
        model_class=model_class,
    )

    try:
        for key, value in reg_tags.as_mlflow_tags().items():
            client.set_tag(run_id, key, value)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[{label}] Could not tag MLflow run {run_id}: {exc}")


def _finalize_batch_child(
    *,
    label: str,
    prepared: PreparedTraining,
    execution_result: object,
    settings: NeurallsSettings,
) -> TrainingRunResult:
    """Finalize one successful multirun child: durability, metrics, annotation.

    Mirrors what the old ``_train_single`` did after ``train_model()``
    returned — everything needed (assignment/dataset/job identity) is already
    resolved onto ``prepared.assignment.spec`` from the prepare pass.

    Args:
        label: Short numeric label for log messages.
        prepared: This assignment's prepared training inputs.
        execution_result: The multirun child's dispatch result.
        settings: Resolved neuralls settings.

    Returns:
        ``TrainingRunResult`` with the MLflow run ID and metrics.
    """
    spec = prepared.assignment.spec
    run_id, tracking_uri = finalize_prepared_training(prepared, execution_result)
    data_cfg = load_data_config(prepared.resolved_data_config_path, settings)
    resolved_dataset_id = resolve_dataset_identity(
        data_cfg=data_cfg,
        config_path=prepared.resolved_data_config_path,
    ).name
    model_class = read_model_class_name(prepared.config_path)
    resolved_dataset_display_name = resolve_display_name(
        resolved_dataset_id, spec.dataset_display_name
    )
    resolved_job_display_name = resolve_display_name(
        model_class or spec.job_id or spec.assignment_id,
        spec.job_display_name,
    )

    metrics: dict[str, float] = {}
    try:
        metrics = fetch_mlflow_metrics(run_id, tracking_uri)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[{label}] Could not fetch MLflow metrics: {exc}")

    _annotate_mlflow_run(
        label=label,
        run_id=run_id,
        tracking_uri=tracking_uri,
        job_config_path=prepared.config_path,
        assignment_id=spec.assignment_id,
        assignment_display_name=spec.assignment_display_name,
        resolved_dataset_id=resolved_dataset_id,
        dataset_display_name=resolved_dataset_display_name,
        dataset_registry_id=spec.dataset_id,
        job_registry_id=spec.job_id,
        job_display_name=resolved_job_display_name,
    )

    logger.info(f"[{label}] Done. run_id={run_id}, metrics={list(metrics.keys())}")
    return TrainingRunResult(
        label=label,
        assignment_id=spec.assignment_id,
        assignment_display_name=spec.assignment_display_name,
        mlflow_run_id=run_id,
        metrics=metrics,
        resolved_dataset_id=resolved_dataset_id,
        dataset_display_name=resolved_dataset_display_name,
        model_class=model_class,
        job_display_name=resolved_job_display_name,
        dataset_registry_id=spec.dataset_id,
        job_registry_id=spec.job_id,
        job_config_path=prepared.config_path,
        data_config_path=prepared.resolved_data_config_path,
    )


def _resolve_training_entry_metadata(
    *,
    entry: AssignmentEntry,
    cfg: CaseConfig,
) -> tuple[str, str | None, str | None, str | None]:
    """Return registry/display metadata for one training entry."""
    dataset_entry = _find_registry_entry(cfg.datasets, entry.dataset_id)
    job_entry = _find_registry_entry(cfg.jobs, entry.job_id)
    dataset_display_name = (
        dataset_entry.effective_display_name if dataset_entry is not None else entry.dataset_id
    )
    job_display_name = job_entry.effective_display_name if job_entry is not None else entry.job_id
    return (
        entry.dataset_id,
        dataset_display_name,
        entry.job_id,
        job_display_name,
    )


def _resolve_case_config_path(case_config_path: Path | None, configs_dir: Path) -> Path:
    """Return a stable case-config identity path for training session parents."""
    if case_config_path is not None:
        return case_config_path.resolve()
    return (configs_dir / "case.toml").resolve()


def train_batch(
    cfg: CaseConfig,
    configs_dir: Path,
    settings: NeurallsSettings | None = None,
    output_root: Path | None = None,
    case_config_path: Path | None = None,
) -> BatchResult:
    """Train all assignments defined in a case config and collect metrics.

    Phase 1: Trains each assignment sequentially. Each batch invocation opens
    one MLflow session parent, while dlkit still manages the individual child
    training runs independently.

    Args:
        cfg: Validated case configuration.
        configs_dir: Parent directory of the case TOML (for resolving
            relative config paths).
        output_root: Optional override for the training output root directory.

    Returns:
        ``BatchResult`` with all run results, label map, and batch output directory.

    Raises:
        ValueError: If the TOML contains no ``[[assignments]]`` entries.
        FileNotFoundError: If any resolved config path does not exist.
    """
    settings = require_settings(settings)
    assignment_entries = list(cfg.assignments)
    if not assignment_entries:
        raise ValueError("No [[assignments]] entries found in case config.")

    tracking_uri = cfg.mlflow.tracking_uri
    if output_root is not None:
        base_output = Path(output_root)
    elif tracking_uri is not None and is_sqlite_tracking_uri(tracking_uri):
        derived_output = derive_output_root_from_tracking_uri(tracking_uri)
        if derived_output is None:
            raise ValueError("MLflow tracking_uri must resolve to a local sqlite database path.")
        base_output = derived_output
    else:
        base_output = settings.output_dir

    training_mlflow_env = build_workflow_environment(
        tracking_uri=tracking_uri,
        artifact_location=cfg.mlflow.artifacts_destination,
        default_output_root=base_output,
    )
    mlflow_experiment_name = cfg.names.training
    resolved_case_config_path = _resolve_case_config_path(case_config_path, configs_dir)

    # ------------------------------------------------------------------
    # Phase 1: Prepare every assignment's settings, then run them as one
    # dlkit multirun sweep — dlkit owns the parent run's lifecycle, child
    # tagging, and per-child failure isolation natively.
    # ------------------------------------------------------------------
    results: list[TrainingRunResult] = []
    with scoped_mlflow_environment(training_mlflow_env.env):
        session_run_name, session_tags = build_session_run_spec(
            case_config_path=resolved_case_config_path,
            experiment_name=mlflow_experiment_name,
            phase="session_training",
        )

        prepared_by_child_id: dict[str, tuple[PreparedTraining, str]] = {}
        run_specs: list[RunSpec] = []
        session_failed = False

        for i, entry in enumerate(assignment_entries, start=1):
            label = str(i)
            try:
                job_config_path, data_config = _resolve_config_paths(entry, configs_dir, cfg)
                (
                    dataset_registry_id,
                    dataset_display_name,
                    job_registry_id,
                    job_display_name,
                ) = _resolve_training_entry_metadata(entry=entry, cfg=cfg)
                prepared = prepare_training_settings(
                    config_path=job_config_path,
                    data_config_path=data_config,
                    settings=settings,
                    output_root=base_output,
                    assignment_id=entry.id,
                    assignment_display_name=entry.effective_display_name,
                    dataset_registry_id=dataset_registry_id,
                    dataset_display_name=dataset_display_name,
                    job_registry_id=job_registry_id,
                    job_display_name=job_display_name,
                    mlflow_experiment_name=mlflow_experiment_name,
                    batched=True,
                )
            except Exception as exc:  # noqa: BLE001
                # Mirror run_assignment()'s resilience: one assignment's
                # failure must never abort the rest of the batch.
                session_failed = True
                logger.error(f"[{label}] Assignment '{entry.id}' failed to prepare: {exc}")
                continue
            prepared_by_child_id[prepared.assignment.spec.assignment_id] = (prepared, label)
            run_specs.append(to_run_spec(prepared))

        sweep_result = None
        if run_specs:
            sweep_result = run_multirun_spec(
                MultiRunSpec(
                    experiment_name=mlflow_experiment_name,
                    parent_run_name=session_run_name,
                    parent_tags=dict(session_tags.as_mlflow_tags()),
                    failure_policy="continue",
                    children=tuple(run_specs),
                )
            )
            for outcome in sweep_result.children:
                prepared, label = prepared_by_child_id[outcome.child_id]
                try:
                    match outcome:
                        case ChildSuccess():
                            results.append(
                                _finalize_batch_child(
                                    label=label,
                                    prepared=prepared,
                                    execution_result=outcome.result,
                                    settings=settings,
                                )
                            )
                            logger.info(f"Completed {label}/{len(run_specs)} assignments")
                        case ChildFailure():
                            session_failed = True
                            logger.error(
                                f"[{label}] Assignment '{outcome.child_id}' failed: {outcome.message}"
                            )
                finally:
                    cleanup_prepared_training(prepared)

        if sweep_result is not None:
            finalize_session_parent_run(
                tracking_uri=sweep_result.tracking_uri or training_mlflow_env.tracking_uri,
                run_id=sweep_result.parent_run_id,
                status="FAILED" if session_failed else "FINISHED",
            )

    label_map = _make_label_map(results)
    output_dir = base_output / "training"
    return BatchResult(
        results=results,
        label_map=label_map,
        output_dir=output_dir,
    )


def write_metric_report(
    batch: BatchResult,
    *,
    metric: str,
    output_dir: Path | None = None,
) -> Path:
    """Write aggregate batch-training artifacts for one metric."""
    report_dir = output_dir if output_dir is not None else batch.output_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    labels: list[str] = []
    values: list[float] = []
    missing: list[str] = []

    for result in batch.results:
        if metric in result.metrics:
            labels.append(result.label)
            values.append(result.metrics[metric])
            continue
        missing.append(f"{result.label} ({result.assignment_display_name})")

    if missing:
        logger.warning("Metric '{}' missing for assignments: {}", metric, ", ".join(missing))

    legend = {
        result.label: (
            f"{result.assignment_display_name} (run: {result.mlflow_run_id})"
            if result.mlflow_run_id
            else result.assignment_display_name
        )
        for result in batch.results
        if result.label in labels
    }

    if labels:
        plot_metric_comparison(
            labels=labels,
            values=values,
            metric_name=metric,
            legend=legend,
            save_path=report_dir / f"batch_metric_{metric.replace('/', '_')}.png",
        )

    (report_dir / "batch_training_labels.json").write_text(
        json.dumps(batch.label_map, indent=2),
        encoding="utf-8",
    )
    return report_dir
