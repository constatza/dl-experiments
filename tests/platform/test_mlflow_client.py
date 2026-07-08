"""Tests for find_successful_run's filter-string construction and result parsing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from neuralls.platform.tracking.mlflow_client import find_successful_run


def _mock_client(experiment_id: str | None, runs: list[MagicMock]) -> MagicMock:
    client = MagicMock()
    if experiment_id is None:
        client.get_experiment_by_name.return_value = None
    else:
        client.get_experiment_by_name.return_value = MagicMock(experiment_id=experiment_id)
    client.search_runs.return_value = runs
    return client


def test_returns_none_when_experiment_does_not_exist() -> None:
    """No MLflow experiment bucket means nothing to search — return None."""
    client = _mock_client(experiment_id=None, runs=[])
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        result = find_successful_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Train",
            assignment_id="exp-1",
        )
    assert result is None
    client.search_runs.assert_not_called()


def test_returns_none_when_no_matching_run_found() -> None:
    """An existing experiment with no matching run also returns None."""
    client = _mock_client(experiment_id="mlflow-exp-1", runs=[])
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        result = find_successful_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Train",
            assignment_id="exp-1",
        )
    assert result is None


def test_returns_run_id_of_most_recent_match() -> None:
    """The first run in the (already start_time-DESC-ordered) result is returned."""
    run = MagicMock()
    run.info.run_id = "run-abc"
    client = _mock_client(experiment_id="mlflow-exp-1", runs=[run])
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        result = find_successful_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Train",
            assignment_id="exp-1",
        )
    assert result == "run-abc"


def test_filter_string_scopes_to_assignment_id_and_finished_status() -> None:
    """The search filter is scoped to this exact assignment_id and FINISHED status —
    the two properties that make the reuse check safe across different assignments
    and against crashed/still-running prior attempts."""
    client = _mock_client(experiment_id="mlflow-exp-1", runs=[])
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        find_successful_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Train",
            assignment_id="my-assignment",
        )
    _, kwargs = client.search_runs.call_args
    assert kwargs["experiment_ids"] == ["mlflow-exp-1"]
    assert "tags.assignment_id = 'my-assignment'" in kwargs["filter_string"]
    assert "attributes.status = 'FINISHED'" in kwargs["filter_string"]


def test_two_different_assignment_ids_produce_different_filters() -> None:
    """Distinct assignment_ids never collide on the same search filter."""
    client = _mock_client(experiment_id="mlflow-exp-1", runs=[])
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        find_successful_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Train",
            assignment_id="train-job",
        )
        find_successful_run(
            tracking_uri="sqlite:///tracking.db",
            mlflow_experiment_name="Train",
            assignment_id="search-job",
        )
    first_filter = client.search_runs.call_args_list[0].kwargs["filter_string"]
    second_filter = client.search_runs.call_args_list[1].kwargs["filter_string"]
    assert "train-job" in first_filter
    assert "search-job" in second_filter
    assert first_filter != second_filter
