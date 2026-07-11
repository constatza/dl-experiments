"""MLflow-backed reuse: an assignment only skips retraining if find_successful_run

reports a FINISHED run tagged with its own assignment_id — never based on a local
checkpoint or another assignment's run.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neuralls.composition.assignments import training_batch


@pytest.fixture
def run_assignment_dependencies(tmp_path: Path) -> Iterator[dict[str, MagicMock]]:
    """Patch every I/O dependency of ``run_assignment`` except the reuse decision."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with (
        patch.object(training_batch, "load_data_config") as load_data_config,
        patch.object(training_batch, "resolve_dataset_identity") as resolve_dataset_identity,
        patch.object(training_batch, "load_matrix"),
        patch.object(training_batch, "process_config", return_value=data_dir),
        patch.object(training_batch, "resolve_dataset_artifacts") as resolve_dataset_artifacts,
        patch.object(training_batch, "find_successful_run") as find_successful_run,
        patch.object(training_batch, "train_model") as train_model,
    ):
        data_cfg = MagicMock()
        data_cfg.source.matrix_path = tmp_path / "matrix.npy"
        data_cfg.output.data_dir = data_dir
        load_data_config.return_value = data_cfg
        resolve_dataset_identity.return_value.name = "dataset-id"
        resolve_dataset_artifacts.return_value = MagicMock(
            rhs=MagicMock(path=data_dir),
            solutions=MagicMock(path=data_dir),
            matrix=MagicMock(path=data_dir),
        )
        train_model.return_value = ("new-run-id", "sqlite:///new.db")
        yield {"find_successful_run": find_successful_run, "train_model": train_model}


def _run(
    deps: dict[str, MagicMock],
    tmp_path: Path,
    *,
    assignment_id: str = "exp-1",
    force: bool = False,
) -> None:
    training_batch.run_assignment(
        settings=MagicMock(),
        job_config_path=tmp_path / "job.toml",
        data_config_path=tmp_path / "data.toml",
        output_root=tmp_path,
        force=force,
        src_hash="hash",
        assignment_id=assignment_id,
        assignment_display_name="Assignment 1",
        mlflow_experiment_name="Train",
        tracking_uri="sqlite:///tracking.db",
    )


def test_train_skips_when_a_finished_run_already_exists(
    run_assignment_dependencies: dict[str, MagicMock], tmp_path: Path
) -> None:
    """A FINISHED run tagged with this exact assignment_id short-circuits training."""
    run_assignment_dependencies["find_successful_run"].return_value = "existing-run-id"
    _run(run_assignment_dependencies, tmp_path)
    run_assignment_dependencies["train_model"].assert_not_called()


def test_train_runs_when_no_finished_run_exists(
    run_assignment_dependencies: dict[str, MagicMock], tmp_path: Path
) -> None:
    """No matching FINISHED run means training proceeds."""
    run_assignment_dependencies["find_successful_run"].return_value = None
    _run(run_assignment_dependencies, tmp_path)
    run_assignment_dependencies["train_model"].assert_called_once()


def test_force_always_retrains_even_with_a_finished_run(
    run_assignment_dependencies: dict[str, MagicMock], tmp_path: Path
) -> None:
    """force=True bypasses the MLflow reuse check entirely."""
    run_assignment_dependencies["find_successful_run"].return_value = "existing-run-id"
    _run(run_assignment_dependencies, tmp_path, force=True)
    run_assignment_dependencies["train_model"].assert_called_once()
    run_assignment_dependencies["find_successful_run"].assert_not_called()


def test_run_assignment_returns_failed_result_on_unexpected_exception(
    run_assignment_dependencies: dict[str, MagicMock], tmp_path: Path
) -> None:
    """An unexpected exception (e.g. dlkit's leaked-MLflow-run error on a search job)
    must be recorded as a Failed result, not raised — run_assignment_matrix's
    "failed assignments don't stop the batch" guarantee depends on this.
    """
    run_assignment_dependencies["find_successful_run"].return_value = None
    run_assignment_dependencies["train_model"].side_effect = Exception(
        "Run with UUID abc123 is already active."
    )
    result = training_batch.run_assignment(
        settings=MagicMock(),
        job_config_path=tmp_path / "job.toml",
        data_config_path=tmp_path / "data.toml",
        output_root=tmp_path,
        force=False,
        src_hash="hash",
        assignment_id="search-job-2",
        assignment_display_name="Assignment 2",
        mlflow_experiment_name="Train",
        tracking_uri="sqlite:///tracking.db",
    )
    assert result.status == "Failed"
    assert "already active" in (result.error or "")


def test_two_assignments_never_share_a_lookup_key(
    run_assignment_dependencies: dict[str, MagicMock], tmp_path: Path
) -> None:
    """A FINISHED run for one assignment_id must not skip a different assignment_id.

    find_successful_run is the sole reuse signal; querying it per-assignment_id
    (asserted below) is what keeps two assignments sharing an architecture+dataset
    from colliding on the same checkpoint slot.
    """
    run_assignment_dependencies["find_successful_run"].return_value = None
    _run(run_assignment_dependencies, tmp_path, assignment_id="train-job")
    _run(run_assignment_dependencies, tmp_path, assignment_id="search-job")

    queried_ids = [
        call.kwargs["assignment_id"]
        for call in run_assignment_dependencies["find_successful_run"].call_args_list
    ]
    assert queried_ids == ["train-job", "search-job"]
    assert run_assignment_dependencies["train_model"].call_count == 2
