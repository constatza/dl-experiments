"""Pure builders for MLflow run specs and structured tag dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

from neuralls.platform.config.models.experiments import ComparisonRegistryEntry, ExperimentEntry
from neuralls.platform.config.resolution import MlflowPaths
from neuralls.platform.tracking.mlflow import MlflowRunConfig


def iso_timestamp(dt: datetime | None = None) -> str:
    """Return local ISO 8601 timestamp at second precision (injectable for tests).

    Args:
        dt: Optional datetime to format. Defaults to now().

    Returns:
        ISO 8601 formatted timestamp string at second precision.
    """
    return (dt or datetime.now()).strftime("%Y-%m-%dT%H:%M:%S")


def format_run_timestamp(dt: datetime | None = None) -> str:
    """Return a readable local timestamp for MLflow run names."""
    return (dt or datetime.now()).strftime("%a %d %b %Y - %H:%M:%S")


@dataclass(frozen=True)
class TrainingRunTags:
    """Structured tags for a training run.

    Attributes:
        phase: Always "training".
        experiment_id: Stable experiment identifier.
        dataset_id: Dataset registry id.
        model_id: Model registry id.
        experiment_display_name: Human-facing experiment label.
    """

    phase: Literal["training"]
    experiment_id: str
    dataset_id: str
    model_id: str
    experiment_display_name: str

    def as_mlflow_tags(self) -> dict[str, str]:
        """Serialize to MLflow-compatible string tag dict."""
        return asdict(self)


@dataclass(frozen=True)
class ComparisonRunTags:
    """Structured tags for a comparison outer run.

    Attributes:
        phase: Always "comparison".
        comparison_id: Stable comparison identifier.
        comparison_display_name: Human-facing comparison label.
        comparison_config: Stem of comparison config file.
        comparison_path: Full path string.
        started_at: ISO timestamp (local).
        run_name: Timestamped run name (for child back-linking).
    """

    phase: Literal["comparison"]
    comparison_id: str
    comparison_display_name: str
    comparison_config: str
    comparison_path: str
    started_at: str
    run_name: str

    def as_mlflow_tags(self) -> dict[str, str]:
        """Serialize to MLflow-compatible string tag dict."""
        return asdict(self)


@dataclass(frozen=True)
class ChildComparisonRunTags:
    """Structured tags for a per-preconditioner nested CG run.

    Attributes:
        phase: Always "preconditioner_run".
        preconditioner: Preconditioner name.
        comparison_id: Parent comparison identifier.
        parent_run_name: Timestamped parent run name (for back-linking).
    """

    phase: Literal["preconditioner_run"]
    preconditioner: str
    comparison_id: str
    parent_run_name: str

    def as_mlflow_tags(self) -> dict[str, str]:
        """Serialize to MLflow-compatible string tag dict."""
        return asdict(self)


@dataclass(frozen=True)
class TrainingSessionRunTags:
    """Structured tags for a batch-training session parent run."""

    phase: Literal["session_training"]
    case_config: str
    case_config_path: str
    started_at: str
    training_experiment_name: str

    def as_mlflow_tags(self) -> dict[str, str]:
        """Serialize to MLflow-compatible string tag dict."""
        return asdict(self)


@dataclass(frozen=True)
class RegistrationTags:
    """Structured tags for a registered model version.

    Note: registered_at UTC timestamp is added as an invariant inside
    register_logged_model() — not here — so callers cannot accidentally omit it.

    Attributes:
        experiment_id: Stable experiment identifier.
        dataset_id: Dataset registry id.
        model_id: Model registry id.
        experiment_display_name: Human-facing experiment label.
        model_class: Optional model class name from [MODEL].name.
    """

    experiment_id: str
    dataset_id: str
    model_id: str
    experiment_display_name: str
    model_class: str | None = None

    def as_mlflow_tags(self) -> dict[str, str]:
        """Serialize to MLflow-compatible string tag dict, omitting None values."""
        tags: dict[str, str] = {
            "experiment_id": self.experiment_id,
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "experiment_display_name": self.experiment_display_name,
        }
        if self.model_class is not None:
            tags["model_class"] = self.model_class
        return tags


def build_training_run_spec(
    *,
    entry: ExperimentEntry,
    experiment_name: str,
    paths: MlflowPaths,
    workspace_root: Path,
    timestamp: str | None = None,
    include_timestamp: bool = True,
) -> MlflowRunConfig:
    """Build complete MlflowRunConfig for a training run.

    Run name format: ``{display_name} | {readable_timestamp}`` when
    ``include_timestamp`` is True (standalone runs), or just ``{display_name}``
    when False (batch runs where the session parent already carries the timestamp).

    Args:
        entry: Experiment registry entry.
        experiment_name: MLflow experiment name to group runs under.
        paths: Resolved MLflow tracking/artifact URIs.
        workspace_root: Workspace root directory.
        timestamp: Optional fixed timestamp (injectable for tests).
        include_timestamp: When False, omit the timestamp from the run name.

    Returns:
        Complete MlflowRunConfig ready for training.
    """
    ts = timestamp or format_run_timestamp()
    tags = TrainingRunTags(
        phase="training",
        experiment_id=entry.id,
        dataset_id=entry.dataset_id,
        model_id=entry.model_id,
        experiment_display_name=entry.effective_display_name,
    )
    display = entry.effective_display_name
    run_name = f"{display} | {ts}" if include_timestamp else display
    return MlflowRunConfig(
        experiment_name=experiment_name,
        run_name=run_name,
        tags=tags.as_mlflow_tags(),
        paths=paths,
        workspace_root=workspace_root,
    )


def build_comparison_run_spec(
    *,
    entry: ComparisonRegistryEntry,
    timestamp: str | None = None,
) -> tuple[str, ComparisonRunTags]:
    """Build (run_name, ComparisonRunTags) for a comparison outer run.

    Returns structured tags — caller calls ``.as_mlflow_tags()`` at the MLflow boundary.

    Args:
        entry: Comparison registry entry.
        timestamp: Optional fixed timestamp (injectable for tests).

    Returns:
        Tuple of ``(run_name, ComparisonRunTags)``.
    """
    ts = timestamp or iso_timestamp()
    run_name = f"{entry.effective_display_name} | {ts}"
    method_label = entry.method.stem if entry.method is not None else entry.id
    tags = ComparisonRunTags(
        phase="comparison",
        comparison_id=entry.id,
        comparison_display_name=entry.effective_display_name,
        comparison_config=method_label,
        comparison_path=entry.method.as_posix() if entry.method is not None else entry.id,
        started_at=ts,
        run_name=run_name,
    )
    return run_name, tags


def build_training_session_run_spec(
    *,
    case_config_path: Path,
    training_experiment_name: str,
    timestamp: str | None = None,
) -> tuple[str, TrainingSessionRunTags]:
    """Build (run_name, TrainingSessionRunTags) for one batch-training invocation."""
    ts = timestamp or iso_timestamp()
    run_name = f"{case_config_path.stem} | {ts}"
    tags = TrainingSessionRunTags(
        phase="session_training",
        case_config=case_config_path.stem,
        case_config_path=case_config_path.as_posix(),
        started_at=ts,
        training_experiment_name=training_experiment_name,
    )
    return run_name, tags


def build_child_comparison_tags(
    *,
    preconditioner_name: str,
    comparison_id: str,
    parent_run_name: str,
) -> ChildComparisonRunTags:
    """Build structured tags for a nested per-preconditioner CG run.

    Args:
        preconditioner_name: Name of the preconditioner.
        comparison_id: Parent comparison identifier.
        parent_run_name: Timestamped parent run name for back-linking.

    Returns:
        ChildComparisonRunTags with all fields set.
    """
    return ChildComparisonRunTags(
        phase="preconditioner_run",
        preconditioner=preconditioner_name,
        comparison_id=comparison_id,
        parent_run_name=parent_run_name,
    )


def build_registration_tags(
    *,
    entry: ExperimentEntry,
    model_class: str | None,
) -> RegistrationTags:
    """Build structured registration tags for a model version.

    Args:
        entry: Experiment registry entry.
        model_class: Optional model class name from [MODEL].name.

    Returns:
        RegistrationTags with all available fields populated.
    """
    return RegistrationTags(
        experiment_id=entry.id,
        dataset_id=entry.dataset_id,
        model_id=entry.model_id,
        experiment_display_name=entry.effective_display_name,
        model_class=model_class,
    )
