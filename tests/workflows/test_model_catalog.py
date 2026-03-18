"""Tests for model catalog registration helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from neuralls.workflows.model_catalog import (
    assign_dataset_alias_to_registered_model,
    build_registered_model_name,
    register_logged_model,
)


def test_build_registered_model_name_uses_architecture_only() -> None:
    """Model registry name should match architecture identifier exactly."""
    assert build_registered_model_name("NormScaledLinearFFNN") == "NormScaledLinearFFNN"


@pytest.fixture
def mock_mlflow_and_client():
    """Patched mlflow module and MlflowClient for registration tests."""
    with (
        patch("neuralls.workflows.model_catalog.mlflow") as mock_mlflow,
        patch("neuralls.workflows.model_catalog.MlflowClient") as mock_client_cls,
    ):
        mock_mlflow.register_model.return_value.version = "3"
        client = mock_client_cls.return_value
        client.get_model_version_by_alias.return_value.version = "3"
        yield mock_mlflow, client


@patch("neuralls.workflows.model_catalog.MlflowClient")
@patch("neuralls.workflows.model_catalog.mlflow")
def test_register_logged_model_applies_aliases_and_tags(
    mock_mlflow: MagicMock,
    mock_client_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Registration normalizes aliases and writes version tags."""
    mock_mlflow.register_model.return_value.version = "7"
    client = mock_client_cls.return_value
    client.get_model_version_by_alias.return_value.version = "7"

    record = register_logged_model(
        run_id="run-1",
        registered_model_name="NormScaledLinearFFNN",
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        aliases=("@solutions", "candidate", "candidate"),
        tags={
            "dataset": "solutions",
            "dataset_config_path": str(tmp_path / "datasets" / "solutions.toml"),
        },
    )

    assert record.name == "NormScaledLinearFFNN"
    assert record.version == 7
    client.set_registered_model_alias.assert_has_calls(
        [
            call(name="NormScaledLinearFFNN", alias="solutions", version="7"),
            call(name="NormScaledLinearFFNN", alias="candidate", version="7"),
        ]
    )
    client.get_model_version_by_alias.assert_has_calls(
        [
            call(name="NormScaledLinearFFNN", alias="solutions"),
            call(name="NormScaledLinearFFNN", alias="candidate"),
        ]
    )
    client.set_model_version_tag.assert_has_calls(
        [
            call(
                name="NormScaledLinearFFNN",
                version="7",
                key="dataset",
                value="solutions",
            ),
            call(
                name="NormScaledLinearFFNN",
                version="7",
                key="dataset_config_path",
                value=str(tmp_path / "datasets" / "solutions.toml"),
            ),
        ]
    )


@patch("neuralls.workflows.model_catalog.MlflowClient")
@patch("neuralls.workflows.model_catalog.mlflow")
def test_register_logged_model_rejects_reserved_alias(
    mock_mlflow: MagicMock,
    _mock_client_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Reserved aliases fail fast before assignment."""
    mock_mlflow.register_model.return_value.version = "7"
    with pytest.raises(ValueError, match="reserved"):
        register_logged_model(
            run_id="run-1",
            registered_model_name="NormScaledLinearFFNN",
            tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            aliases=("latest",),
            tags=None,
        )


@patch("neuralls.workflows.model_catalog.MlflowClient")
def test_assign_dataset_alias_to_registered_model_picks_highest_version(
    mock_client_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Alias assignment picks highest matching version for the same run."""
    client = mock_client_cls.return_value
    mv1 = MagicMock()
    mv1.version = "2"
    mv1.run_id = "run-1"
    mv2 = MagicMock()
    mv2.version = "7"
    mv2.run_id = "run-1"
    mv_other = MagicMock()
    mv_other.version = "9"
    mv_other.run_id = "run-2"
    client.search_model_versions.return_value = [mv1, mv2, mv_other]
    resolved = MagicMock()
    resolved.version = "7"
    client.get_model_version_by_alias.return_value = resolved

    version = assign_dataset_alias_to_registered_model(
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        registered_model_name="NormScaledLinearFFNN",
        run_id="run-1",
        dataset_alias="@solutions",
    )

    assert version == 7
    client.search_model_versions.assert_called_once_with("name='NormScaledLinearFFNN'")
    client.set_registered_model_alias.assert_called_once_with(
        name="NormScaledLinearFFNN",
        alias="solutions",
        version="7",
    )
    client.get_model_version_by_alias.assert_called_once_with(
        name="NormScaledLinearFFNN",
        alias="solutions",
    )


@patch("neuralls.workflows.model_catalog.MlflowClient")
def test_assign_dataset_alias_to_registered_model_returns_none_when_missing(
    mock_client_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """No matching version for run_id returns None without alias assignment."""
    client = mock_client_cls.return_value
    mv = MagicMock()
    mv.version = "3"
    mv.run_id = "another-run"
    client.search_model_versions.return_value = [mv]

    version = assign_dataset_alias_to_registered_model(
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        registered_model_name="NormScaledLinearFFNN",
        run_id="run-1",
        dataset_alias="solutions",
    )

    assert version is None
    client.set_registered_model_alias.assert_not_called()


def test_register_logged_model_uses_experiment_id_as_name(
    mock_mlflow_and_client: tuple,
    tmp_path: Path,
) -> None:
    """Registering under experiment_id produces a model named by experiment, not architecture."""
    mock_mlflow, client = mock_mlflow_and_client
    experiment_id = "spectral-energy"
    model_class = "NormScaledLinearFFNN"

    record = register_logged_model(
        run_id="run-exp",
        registered_model_name=experiment_id,
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        aliases=("candidate",),
        tags={"model_class": model_class},
    )

    assert record.name == experiment_id
    mock_mlflow.register_model.assert_called_once_with(
        model_uri="runs:/run-exp/model",
        name=experiment_id,
    )
    registered_at_value = next(
        call_args.kwargs["value"]
        for call_args in client.set_model_version_tag.call_args_list
        if call_args.kwargs["key"] == "registered_at"
    )
    client.set_model_version_tag.assert_has_calls(
        [
            call(
                name=experiment_id,
                version="3",
                key="registered_at",
                value=registered_at_value,
            ),
            call(
                name=experiment_id,
                version="3",
                key="model_class",
                value=model_class,
            ),
        ],
        any_order=True,
    )
    datetime.fromisoformat(registered_at_value)


@patch("neuralls.workflows.model_catalog.logger")
@patch("neuralls.workflows.model_catalog.MlflowClient")
@patch("neuralls.workflows.model_catalog.mlflow")
def test_register_logged_model_warns_when_name_exists(
    mock_mlflow: MagicMock,
    mock_client_cls: MagicMock,
    mock_logger: MagicMock,
    tmp_path: Path,
) -> None:
    """Existing registered names emit a warning before adding a new version."""
    mock_mlflow.register_model.return_value.version = "5"
    client = mock_client_cls.return_value
    client.get_registered_model.return_value = object()

    register_logged_model(
        run_id="run-1",
        registered_model_name="exp-1",
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        tags={"model_class": "NormScaledLinearFFNN"},
    )

    client.get_registered_model.assert_called_once_with("exp-1")
    mock_logger.warning.assert_called_once_with(
        "Registered model '{}' already exists. Registering a new version.",
        "exp-1",
    )


@patch("neuralls.workflows.model_catalog.MlflowClient")
@patch("neuralls.workflows.model_catalog.mlflow")
def test_register_logged_model_always_sets_registered_at(
    mock_mlflow: MagicMock,
    mock_client_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Every registered model version receives a UTC registration timestamp tag."""
    mock_mlflow.register_model.return_value.version = "5"
    client = mock_client_cls.return_value

    register_logged_model(
        run_id="run-1",
        registered_model_name="exp-1",
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        tags={"model_class": "NormScaledLinearFFNN"},
    )

    registered_at_calls = [
        call_args
        for call_args in client.set_model_version_tag.call_args_list
        if call_args.kwargs["key"] == "registered_at"
    ]
    assert len(registered_at_calls) == 1
    datetime.fromisoformat(registered_at_calls[0].kwargs["value"])


@patch("neuralls.workflows.model_catalog.MlflowClient")
@patch("neuralls.workflows.model_catalog.mlflow")
def test_register_logged_model_registered_at_not_overridden_by_caller(
    mock_mlflow: MagicMock,
    mock_client_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Caller-provided registered_at values must not replace the invariant timestamp."""
    mock_mlflow.register_model.return_value.version = "5"
    client = mock_client_cls.return_value

    register_logged_model(
        run_id="run-1",
        registered_model_name="exp-1",
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        tags={"registered_at": "override"},
    )

    registered_at_calls = [
        call_args
        for call_args in client.set_model_version_tag.call_args_list
        if call_args.kwargs["key"] == "registered_at"
    ]
    assert len(registered_at_calls) == 1
    assert registered_at_calls[0].kwargs["value"] != "override"


def test_register_logged_model_two_experiments_no_alias_collision(
    tmp_path: Path,
) -> None:
    """Two experiments on the same dataset register under separate names without collision."""
    with (
        patch("neuralls.workflows.model_catalog.mlflow") as mock_mlflow,
        patch("neuralls.workflows.model_catalog.MlflowClient") as mock_client_cls,
    ):
        mock_mlflow.register_model.return_value.version = "1"
        client = mock_client_cls.return_value
        client.get_model_version_by_alias.return_value.version = "1"
        tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

        record_a = register_logged_model(
            run_id="run-a",
            registered_model_name="spectral-energy",
            tracking_uri=tracking_uri,
            aliases=("candidate",),
            tags={"model_class": "NormScaledLinearFFNN"},
        )
        record_b = register_logged_model(
            run_id="run-b",
            registered_model_name="spectral-energy-l2",
            tracking_uri=tracking_uri,
            aliases=("candidate",),
            tags={"model_class": "NormScaledLinearFFNN"},
        )

    assert record_a.name != record_b.name
    assert record_a.name == "spectral-energy"
    assert record_b.name == "spectral-energy-l2"
