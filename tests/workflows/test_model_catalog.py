"""Tests for model catalog registration helpers."""

from __future__ import annotations

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
