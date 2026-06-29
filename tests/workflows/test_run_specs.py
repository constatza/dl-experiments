"""Unit tests for typed MLflow run/tag builders."""

from __future__ import annotations

from pathlib import Path

from neuralls.platform.config.resolution import MlflowPaths, build_sqlite_tracking_uri
from neuralls.platform.config.models.experiments import ComparisonRegistryEntry, ExperimentEntry
from neuralls.composition.tracking.run_specs import (
    TrainingRunTags,
    build_child_comparison_tags,
    build_comparison_run_spec,
    build_registration_tags,
    build_training_session_run_spec,
    build_training_run_spec,
)


def test_training_tags_contains_all_fields() -> None:
    """Training tags serialize all expected fields."""
    tags = TrainingRunTags(
        phase="training",
        experiment_id="exp-1",
        dataset_id="dataset-1",
        job_id="job-1",
        experiment_display_name="Experiment One",
    )

    assert tags.as_mlflow_tags() == {
        "phase": "training",
        "experiment_id": "exp-1",
        "dataset_id": "dataset-1",
        "job_id": "job-1",
        "experiment_display_name": "Experiment One",
    }


def test_training_run_spec_name_has_readable_timestamp() -> None:
    """Training run specs use the experiment display name plus a readable timestamp."""
    entry = ExperimentEntry(
        id="exp-1",
        dataset="dataset-1",
        job="job-1",
        display_name="Experiment One",
    )

    workspace_root = Path.cwd() / "workspace"
    spec = build_training_run_spec(
        entry=entry,
        experiment_name="Train",
        paths=MlflowPaths(
            build_sqlite_tracking_uri(Path.cwd() / "mlflow.db"),
            str((Path.cwd() / "mlartifacts").resolve()),
        ),
        workspace_root=workspace_root,
        timestamp="Thu 12 Mar 2026 - 12:00:00",
    )

    assert spec.experiment_name == "Train"
    assert spec.run_name == "Experiment One | Thu 12 Mar 2026 - 12:00:00"
    assert dict(spec.tags) == {
        "phase": "training",
        "experiment_id": "exp-1",
        "dataset_id": "dataset-1",
        "job_id": "job-1",
        "experiment_display_name": "Experiment One",
    }


def test_comparison_run_spec_name_and_tags() -> None:
    """Comparison runs capture the timestamp, path, and derived config stem."""
    entry = ComparisonRegistryEntry(
        id="cmp-1",
        matrix_dataset="solutions",
        rhs_dataset="gaussian-rhs",
        method=Path("configs/compare.toml"),
        display_name="Comparison One",
    )

    run_name, tags = build_comparison_run_spec(
        entry=entry,
        timestamp="2026-03-12T12:00:00",
    )

    assert run_name == "Comparison One | 2026-03-12T12:00:00"
    assert tags.as_mlflow_tags() == {
        "phase": "comparison",
        "comparison_id": "cmp-1",
        "comparison_display_name": "Comparison One",
        "comparison_config": "compare",
        "comparison_path": "configs/compare.toml",
        "started_at": "2026-03-12T12:00:00",
        "run_name": "Comparison One | 2026-03-12T12:00:00",
    }


def test_child_tags_no_timestamp_has_parent_run_name() -> None:
    """Child comparison tags link back to the named parent run."""
    tags = build_child_comparison_tags(
        preconditioner_name="jacobi",
        comparison_id="cmp-1",
        parent_run_name="Comparison One | 2026-03-12T12:00:00",
    )

    assert tags.as_mlflow_tags() == {
        "phase": "preconditioner_run",
        "preconditioner": "jacobi",
        "comparison_id": "cmp-1",
        "parent_run_name": "Comparison One | 2026-03-12T12:00:00",
    }


def test_training_session_run_spec_name_and_tags() -> None:
    """Training session parents carry case identity and launch time."""
    run_name, tags = build_training_session_run_spec(
        case_config_path=Path("/tmp/cases/ffnn.toml"),
        training_experiment_name="Train",
        timestamp="2026-03-12T12:00:00",
    )

    assert run_name == "ffnn | 2026-03-12T12:00:00"
    assert tags.as_mlflow_tags() == {
        "phase": "session_training",
        "case_config": "ffnn",
        "case_config_path": "/tmp/cases/ffnn.toml",
        "started_at": "2026-03-12T12:00:00",
        "training_experiment_name": "Train",
    }


def test_registration_tags_optional_model_class() -> None:
    """Registration tags include model_class only when provided."""
    entry = ExperimentEntry(
        id="exp-1",
        dataset="dataset-1",
        job="job-1",
        display_name="Experiment One",
    )

    tags = build_registration_tags(entry=entry, model_class="NormScaledLinearFFNN")

    assert tags.as_mlflow_tags()["model_class"] == "NormScaledLinearFFNN"


def test_registration_tags_as_dict_excludes_none() -> None:
    """Registration tags omit None values from the MLflow payload."""
    entry = ExperimentEntry(
        id="exp-1",
        dataset="dataset-1",
        job="job-1",
        display_name="Experiment One",
    )

    tags = build_registration_tags(entry=entry, model_class=None)

    assert tags.as_mlflow_tags() == {
        "experiment_id": "exp-1",
        "dataset_id": "dataset-1",
        "job_id": "job-1",
        "experiment_display_name": "Experiment One",
    }
