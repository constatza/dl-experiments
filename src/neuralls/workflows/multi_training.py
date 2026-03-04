"""Batch training workflow: train multiple (model, dataset) pairs and aggregate metrics.

This module provides functions for training a batch of experiments defined in an
experiments TOML, collecting MLflow metrics from each run, and returning composable
results suitable for comparison plots.

Architecture:
    The batch workflow parses experiments.toml, resolves config paths, trains each
    experiment sequentially (dlkit manages all individual MLflow runs independently),
    then opens OUR own batch MLflow run AFTER all training completes. This avoids
    the "stack desynchronization" warnings from nesting runs with dlkit.

Key Functions:
    - ``train_batch()``: Main entry point; trains experiments, then opens batch MLflow run.
    - ``_train_single()``: Train one experiment and collect its MLflow metrics.
    - ``_read_sidecar()``: Parse ``mlflow_run.json`` written next to checkpoint.
    - ``_fetch_mlflow_metrics()``: Query MLflow for all metrics of a run.
    - ``_make_label_map()``: Build short-label → experiment identity mapping.
    - ``_collect_batch_metrics()``: Aggregate metrics across all training runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

from neuralls.configuration.comparison import ComparisonsTrackingConfig
from neuralls.configuration.experiments import ExperimentsConfig, RunEntry
from neuralls.io.toml_loader import load_raw_toml
from neuralls.workflows.comparison_run import (
    ComparisonRun,
    _artifact_uri_to_local_path,
    setup_comparison_tracking,
)
from neuralls.workflows.model_catalog import assign_dataset_alias_to_registered_model
from neuralls.workflows.training import train_model


@dataclass(frozen=True)
class TrainingRunResult:
    """Immutable result for a single trained experiment.

    Attributes:
        label: Short numeric label ("1", "2", "3" …) for axis display.
        experiment_id: The ``id`` field from the TOML ``[[experiment]]`` entry.
        checkpoint_path: Path to the saved model checkpoint (.ckpt file).
        mlflow_run_id: MLflow run UUID from sidecar JSON, or None if unavailable.
        metrics: All eval/* metrics returned by MLflow for this run.
    """

    label: str
    experiment_id: str
    checkpoint_path: Path
    mlflow_run_id: str | None
    metrics: dict[str, float]


@dataclass(frozen=True)
class BatchResult:
    """Immutable result for a completed batch of training runs.

    Attributes:
        results: Ordered list of per-experiment results (same order as TOML).
        label_map: Maps short label → ``{"experiment_id": …, "mlflow_run_id": …}``.
        output_dir: Directory where batch artefacts (plot, label JSON) are saved.
        comparison_run: Typed handshake for the comparison phase.
    """

    results: list[TrainingRunResult]
    label_map: dict[str, dict[str, str | None]]
    output_dir: Path
    comparison_run: ComparisonRun


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _read_sidecar(checkpoint_path: Path) -> dict[str, str] | None:
    """Read ``mlflow_run.json`` written next to the checkpoint.

    Args:
        checkpoint_path: Path to the saved ``.ckpt`` file.

    Returns:
        Parsed sidecar dict with keys ``run_id``, ``tracking_uri``, etc.,
        or ``None`` if the sidecar file does not exist.
    """
    sidecar = checkpoint_path.parent / "mlflow_run.json"
    if not sidecar.exists():
        return None
    return json.loads(sidecar.read_text())  # type: ignore[return-value]


def _fetch_mlflow_metrics(run_id: str, tracking_uri: str) -> dict[str, float]:
    """Query MLflow for all metrics belonging to a run.

    Args:
        run_id: MLflow run UUID.
        tracking_uri: SQLite or HTTP tracking URI.

    Returns:
        Flat dict of ``{metric_key: value}`` for the run.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    return dict(run.data.metrics)


def _make_label_map(
    results: list[TrainingRunResult],
) -> dict[str, dict[str, str | None]]:
    """Build a mapping from short label to full experiment identity.

    Args:
        results: List of completed training run results.

    Returns:
        Dict ``{label: {"experiment_id": …, "mlflow_run_id": …}}``.
    """
    return {
        r.label: {
            "experiment_id": r.experiment_id,
            "mlflow_run_id": r.mlflow_run_id,
        }
        for r in results
    }


def _resolve_config_paths(
    experiment: Any,
    configs_dir: Path,
) -> tuple[Path, Path]:
    """Resolve model and dataset config paths from an experiment or run entry.

    Supports two formats:
    - ``ExperimentEntry``: uses short names → ``models/{model}.toml``, ``datasets/{dataset}.toml``
    - ``RunEntry``: uses direct relative paths → ``{model_config}``, ``{data_config}``

    Args:
        experiment: Single ``[[experiment]]`` or ``[[run]]`` entry.
        configs_dir: Parent directory of the experiments TOML.

    Returns:
        Tuple of ``(model_config_path, data_config_path)``.

    Raises:
        KeyError: If required keys are missing.
        FileNotFoundError: If either resolved config path does not exist.
    """
    if isinstance(experiment, RunEntry):
        model_path = configs_dir / experiment.model
        dataset_path = configs_dir / experiment.data
    elif isinstance(experiment, dict):
        model_path = configs_dir / "models" / f"{experiment['model']}.toml"
        dataset_path = configs_dir / "datasets" / f"{experiment['dataset']}.toml"
    else:
        model_path = configs_dir / "models" / f"{experiment.model}.toml"
        dataset_path = configs_dir / "datasets" / f"{experiment.dataset}.toml"

    if not model_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_path}")

    return model_path, dataset_path


def _read_registered_model_name(model_config_path: Path) -> str | None:
    """Read model registry name from model config ([MODEL].name)."""
    try:
        raw = load_raw_toml(model_config_path)
    except (FileNotFoundError, OSError, ValueError):
        return None
    model_section = raw.get("MODEL")
    if not isinstance(model_section, dict):
        return None
    model_name = model_section.get("name")
    if not isinstance(model_name, str):
        return None
    model_name = model_name.strip()
    return model_name or None


_BATCH_METRIC_PREFIXES = ("eval/", "val/", "test/")


def _collect_batch_metrics(
    results: list[TrainingRunResult],
    tracking_uri: str | None,
) -> dict[str, float]:
    """Aggregate metrics from all completed training runs.

    Reads metrics that were already fetched and stored in TrainingRunResult.
    Returns averaged metrics across all runs that have them.
    Only metrics whose keys start with ``eval/``, ``val/``, or ``test/`` are
    averaged — counter metrics like epoch counts are excluded.

    Args:
        results: List of completed training run results.
        tracking_uri: MLflow tracking URI (unused, metrics already in results).

    Returns:
        Dict of aggregated metric summaries.
    """
    all_metrics: dict[str, list[float]] = {}
    for result in results:
        for key, value in result.metrics.items():
            if any(key.startswith(p) for p in _BATCH_METRIC_PREFIXES):
                all_metrics.setdefault(key, []).append(value)

    return {
        f"avg_{key}": sum(vals) / len(vals)
        for key, vals in all_metrics.items()
        if vals
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _train_single(
    experiment_id: str,
    model_config_path: Path,
    data_config_path: Path,
    label: str,
    output_root: Path | None,
) -> TrainingRunResult:
    """Train one experiment, read sidecar, and fetch MLflow metrics.

    dlkit manages its own MLflow run independently — no parent run nesting here.

    Args:
        experiment_id: The ``id`` field from the TOML entry.
        model_config_path: Path to model config TOML.
        data_config_path: Path to dataset config TOML.
        label: Short numeric label for this run ("1", "2", …).
        output_root: Optional custom output root directory.

    Returns:
        ``TrainingRunResult`` with checkpoint path, run ID, and metrics.
    """
    logger.info(f"[{label}] Training experiment: {experiment_id}")

    checkpoint_path = train_model(
        config_path=model_config_path,
        data_config_path=data_config_path,
        output_root=output_root,
    )

    sidecar = _read_sidecar(checkpoint_path)
    run_id = sidecar["run_id"] if sidecar else None
    tracking_uri = sidecar.get("tracking_uri") if sidecar else None

    metrics: dict[str, float] = {}
    if run_id and tracking_uri:
        try:
            metrics = _fetch_mlflow_metrics(run_id, tracking_uri)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{label}] Could not fetch MLflow metrics: {exc}")

    if run_id and tracking_uri:
        try:
            client = MlflowClient(tracking_uri=tracking_uri)
            client.log_param(run_id, "experiment_id", experiment_id)
            client.log_param(run_id, "checkpoint_path", str(checkpoint_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{label}] Could not log params to MLflow: {exc}")

    if run_id and tracking_uri:
        dataset_alias = data_config_path.stem
        model_name = _read_registered_model_name(model_config_path)
        if model_name is None:
            logger.warning(
                "[{}] Could not determine [MODEL].name from {}. "
                "Skipping dataset alias assignment.",
                label,
                model_config_path,
            )
        else:
            try:
                assigned = assign_dataset_alias_to_registered_model(
                    tracking_uri=tracking_uri,
                    registered_model_name=model_name,
                    run_id=run_id,
                    dataset_alias=dataset_alias,
                )
                if assigned is not None:
                    logger.info(
                        "[{}] Assigned alias '{}' -> {} v{}",
                        label,
                        dataset_alias,
                        model_name,
                        assigned,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[{}] Could not assign dataset alias '{}' for run {}: {}",
                    label,
                    dataset_alias,
                    run_id,
                    exc,
                )

    logger.info(
        f"[{label}] Done. checkpoint={checkpoint_path}, run_id={run_id}, "
        f"metrics={list(metrics.keys())}"
    )
    return TrainingRunResult(label, experiment_id, checkpoint_path, run_id, metrics)


def train_batch(
    experiments_config_path: Path,
    output_root: Path | None = None,
) -> BatchResult:
    """Train all experiments defined in an experiments TOML and collect metrics.

    Phase 1: Trains each experiment sequentially. dlkit manages all individual
    MLflow runs independently — no parent run is opened during training.

    Phase 2: After all training completes, opens OUR own independent batch MLflow
    run on the comparisons DB (no nesting with dlkit).

    Args:
        experiments_config_path: Path to the experiments TOML (e.g.
            ``configs/experiments.toml``).
        output_root: Optional override for the training output root directory.

    Returns:
        ``BatchResult`` with all run results, label map, output directory, and
        a ``ComparisonRun`` for the comparison phase.

    Raises:
        ValueError: If the TOML contains no ``[[experiment]]`` entries.
        FileNotFoundError: If any resolved config path does not exist.
    """
    experiments_config_path = Path(experiments_config_path).resolve()
    configs_dir = experiments_config_path.parent
    raw = load_raw_toml(experiments_config_path)

    cfg = ExperimentsConfig.model_validate(raw)

    # Support both [[run]] and [[experiment]] entry formats
    run_entries: list[Any] = list(cfg.run) or list(cfg.experiment)
    if not run_entries:
        raise ValueError(
            f"No [[run]] or [[experiment]] entries found in {experiments_config_path}"
        )

    # Derive output_dir from mlflow tracking_uri when not explicitly configured
    tracking_uri = cfg.mlflow.client.tracking_uri
    if output_root is not None:
        base_output = Path(output_root)
    elif cfg.output_dir is not None:
        base_output = cfg.output_dir
    else:
        db_path = Path(tracking_uri.removeprefix("sqlite:///"))
        base_output = db_path.parent.parent

    # Derive comparisons tracking config from mlflow when not explicitly configured
    if cfg.comparisons is not None:
        comp = cfg.comparisons
    else:
        db_path = Path(tracking_uri.removeprefix("sqlite:///"))
        artifact_location = str(db_path.parent.parent / "mlartifacts")
        comp = ComparisonsTrackingConfig(
            tracking_uri=tracking_uri,
            artifact_location=artifact_location,
        )

    experiments = run_entries

    # ------------------------------------------------------------------
    # Phase 1: Train — dlkit manages all individual MLflow runs independently
    # ------------------------------------------------------------------
    results: list[TrainingRunResult] = []
    n = len(experiments)
    for i, entry in enumerate(experiments, start=1):
        label = str(i)
        experiment_id = entry.id
        model_config, data_config = _resolve_config_paths(entry, configs_dir)

        result = _train_single(
            experiment_id=experiment_id,
            model_config_path=model_config,
            data_config_path=data_config,
            label=label,
            output_root=base_output,
        )
        results.append(result)
        logger.info(f"Completed {i}/{n} experiments")

    # ------------------------------------------------------------------
    # Phase 2: Open OUR batch run on the comparisons DB (no nesting with dlkit)
    # ------------------------------------------------------------------
    setup_comparison_tracking(comp.tracking_uri, comp.artifact_location)

    with mlflow.start_run(run_name="batch-comparison", tags={"phase": "training_batch"}) as batch_run:
        batch_run_id = batch_run.info.run_id
        batch_experiment_id = batch_run.info.experiment_id
        artifact_uri = mlflow.get_artifact_uri()

        mlflow.log_param("artifact_uri", artifact_uri)
        mlflow.log_param("artifact_location", comp.artifact_location)
        for result in results:
            mlflow.log_param(f"checkpoint_{result.experiment_id}", str(result.checkpoint_path))

        agg_metrics = _collect_batch_metrics(results, comp.tracking_uri)
        if agg_metrics:
            mlflow.log_metrics(agg_metrics)

        checkpoint_map = {r.experiment_id: r.checkpoint_path for r in results}
        comparison_run = ComparisonRun(
            mlflow_run_id=batch_run_id,
            mlflow_experiment_id=batch_experiment_id,
            tracking_uri=comp.tracking_uri,
            artifact_location=comp.artifact_location,
            checkpoint_map=checkpoint_map,
            artifact_uri=artifact_uri,
        )
        mlflow.log_dict(
            {
                "mlflow_run_id": batch_run_id,
                "mlflow_experiment_id": batch_experiment_id,
                "tracking_uri": comp.tracking_uri,
                "artifact_location": comp.artifact_location,
                "checkpoint_map": {k: str(v) for k, v in checkpoint_map.items()},
                "artifact_uri": artifact_uri,
            },
            "comparison_run.json",
        )
        logger.info(f"Opened MLflow batch run: {batch_run_id}")

    output_dir = _artifact_uri_to_local_path(artifact_uri)
    label_map = _make_label_map(results)
    return BatchResult(
        results=results,
        label_map=label_map,
        output_dir=output_dir,
        comparison_run=comparison_run,
    )
