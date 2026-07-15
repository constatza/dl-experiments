"""Batch training workflow: train multiple (model, dataset) pairs and aggregate metrics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from loguru import logger
from mlflow.tracking import MlflowClient

from neuralls.platform.config.models.dataset_identity import resolve_dataset_identity
from neuralls.platform.config.models.experiments import (
    CaseConfig,
    AssignmentEntry,
    RegistryEntry,
    resolve_display_name,
)
from neuralls.platform.config.registry import (
    resolve_dataset_config_path,
    resolve_job_config_path,
)
from neuralls.platform.config.resolution import (
    derive_output_root_from_tracking_uri,
    is_sqlite_tracking_uri,
)
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.reporting.plots import plot_metric_comparison
from neuralls.platform.tracking.environment import scoped_mlflow_environment
from neuralls.platform.tracking.mlflow import build_workflow_environment
from neuralls.platform.tracking.mlflow_client import fetch_mlflow_metrics
from neuralls.platform.tracking.model_registry import read_model_class_name
from neuralls.composition.tracking.run_specs import (
    build_registration_tags,
    build_session_run_spec,
)
from neuralls.composition.tracking.session import session_parent_run
from neuralls.composition.assignments.training import train_model


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


def _resolve_config_paths(
    assignment: AssignmentEntry,
    configs_dir: Path,
    cfg: CaseConfig,
) -> tuple[Path, Path]:
    """Resolve job and dataset config paths from an assignment entry.

    Args:
        assignment: Single ``[[assignments]]`` entry.
        configs_dir: Parent directory of the assignments TOML.

    Returns:
        Tuple of ``(job_config_path, data_config_path)``.

    Raises:
        FileNotFoundError: If either resolved config path does not exist.
    """
    job_path = resolve_job_config_path(
        cfg,
        configs_dir,
        assignment.job_id,
        assignment_id=assignment.id,
    )
    dataset_path = resolve_dataset_config_path(
        cfg,
        configs_dir,
        assignment.dataset_id,
        assignment_id=assignment.id,
    )

    if not job_path.exists():
        raise FileNotFoundError(f"Job config not found: {job_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_path}")

    return job_path, dataset_path


_BATCH_METRIC_PREFIXES = ("eval/", "val/", "test/")


def _collect_batch_metrics(
    results: list[TrainingRunResult],
) -> dict[str, float]:
    """Aggregate metrics from all completed training runs.

    Reads metrics that were already fetched and stored in TrainingRunResult.
    Returns averaged metrics across all runs that have them.
    Only metrics whose keys start with ``eval/``, ``val/``, or ``test/`` are
    averaged — counter metrics like epoch counts are excluded.

    Args:
        results: List of completed training run results.

    Returns:
        Dict of aggregated metric summaries.
    """
    all_metrics: dict[str, list[float]] = {}
    for result in results:
        for key, value in result.metrics.items():
            if any(key.startswith(p) for p in _BATCH_METRIC_PREFIXES):
                all_metrics.setdefault(key, []).append(value)

    return {f"avg_{key}": sum(vals) / len(vals) for key, vals in all_metrics.items() if vals}


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


def _train_single(
    settings: NeurallsSettings | None,
    assignment_id: str,
    assignment_display_name: str,
    job_config_path: Path,
    data_config_path: Path,
    label: str,
    output_root: Path | None,
    mlflow_experiment_name: str,
    parent_run_id: str | None = None,
    dataset_registry_id: str | None = None,
    dataset_display_name: str | None = None,
    job_registry_id: str | None = None,
    job_display_name: str | None = None,
) -> TrainingRunResult:
    """Train one assignment and fetch MLflow metrics.

    Args:
        assignment_id: The ``id`` field from the TOML entry.
        job_config_path: Path to job config TOML.
        data_config_path: Path to dataset config TOML.
        label: Short numeric label for this run ("1", "2", …).
        output_root: Optional custom output root directory.

    Returns:
        ``TrainingRunResult`` with the MLflow run ID and metrics.
    """
    logger.info(f"[{label}] Training assignment: {assignment_display_name}")
    settings = require_settings(settings)

    run_id, tracking_uri = train_model(
        config_path=job_config_path,
        settings=settings,
        data_config_path=data_config_path,
        output_root=output_root,
        assignment_id=assignment_id,
        assignment_display_name=assignment_display_name,
        dataset_registry_id=dataset_registry_id,
        dataset_display_name=dataset_display_name,
        job_registry_id=job_registry_id,
        job_display_name=job_display_name,
        mlflow_experiment_name=mlflow_experiment_name,
        parent_run_id=parent_run_id,
    )
    data_cfg = load_data_config(data_config_path, settings)
    resolved_dataset_id = resolve_dataset_identity(
        data_cfg=data_cfg,
        config_path=data_config_path,
    ).name
    model_class = read_model_class_name(job_config_path)
    resolved_dataset_display_name = resolve_display_name(resolved_dataset_id, dataset_display_name)
    resolved_job_display_name = resolve_display_name(
        model_class or job_registry_id or assignment_id,
        job_display_name,
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
        job_config_path=job_config_path,
        assignment_id=assignment_id,
        assignment_display_name=assignment_display_name,
        resolved_dataset_id=resolved_dataset_id,
        dataset_display_name=resolved_dataset_display_name,
        dataset_registry_id=dataset_registry_id,
        job_registry_id=job_registry_id,
        job_display_name=resolved_job_display_name,
    )

    logger.info(f"[{label}] Done. run_id={run_id}, metrics={list(metrics.keys())}")
    return TrainingRunResult(
        label=label,
        assignment_id=assignment_id,
        assignment_display_name=assignment_display_name,
        mlflow_run_id=run_id,
        metrics=metrics,
        resolved_dataset_id=resolved_dataset_id,
        dataset_display_name=resolved_dataset_display_name,
        model_class=model_class,
        job_display_name=resolved_job_display_name,
        dataset_registry_id=dataset_registry_id,
        job_registry_id=job_registry_id,
        job_config_path=job_config_path,
        data_config_path=data_config_path,
    )


def _find_registry_entry(
    entries: list[RegistryEntry],
    registry_id: str,
) -> RegistryEntry | None:
    """Look up one registry entry by id."""
    for entry in entries:
        if entry.id == registry_id:
            return entry
    return None


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
    # Phase 1: Train — one session parent groups the independent child runs
    # ------------------------------------------------------------------
    results: list[TrainingRunResult] = []
    n = len(assignment_entries)
    with scoped_mlflow_environment(training_mlflow_env.env):
        session_run_name, session_tags = build_session_run_spec(
            case_config_path=resolved_case_config_path,
            experiment_name=mlflow_experiment_name,
            phase="session_training",
        )
        with session_parent_run(
            tracking_uri=training_mlflow_env.tracking_uri,
            artifact_uri=training_mlflow_env.artifact_uri,
            run_name=session_run_name,
            tags=session_tags.as_mlflow_tags(),
            experiment_name=mlflow_experiment_name,
        ) as handle:
            for i, entry in enumerate(assignment_entries, start=1):
                label = str(i)
                try:
                    assignment_id = entry.id
                    assignment_display_name = entry.effective_display_name
                    job_config_path, data_config = _resolve_config_paths(entry, configs_dir, cfg)
                    (
                        dataset_registry_id,
                        dataset_display_name,
                        job_registry_id,
                        job_display_name,
                    ) = _resolve_training_entry_metadata(
                        entry=entry,
                        cfg=cfg,
                    )

                    result = _train_single(
                        settings=settings,
                        assignment_id=assignment_id,
                        assignment_display_name=assignment_display_name,
                        job_config_path=job_config_path,
                        data_config_path=data_config,
                        label=label,
                        output_root=base_output,
                        mlflow_experiment_name=mlflow_experiment_name,
                        parent_run_id=handle.parent_run_id,
                        dataset_registry_id=dataset_registry_id,
                        dataset_display_name=dataset_display_name,
                        job_registry_id=job_registry_id,
                        job_display_name=job_display_name,
                    )
                    results.append(result)
                    logger.info(f"Completed {i}/{n} assignments")
                except Exception as exc:  # noqa: BLE001
                    # Mirror run_assignment()'s resilience: one assignment's
                    # failure must never abort the rest of the batch.
                    handle.mark_failed()
                    logger.error(f"[{label}] Assignment '{entry.id}' failed: {exc}")

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
