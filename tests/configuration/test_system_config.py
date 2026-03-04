"""Tests for runtime topology configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuralls.configuration.system import (
    ExperimentNamesConfig,
    MlflowTopologyConfig,
    SystemConfig,
)


def test_system_config_minimal_valid() -> None:
    config = SystemConfig(
        mlflow=MlflowTopologyConfig(client={"tracking_uri": "http://127.0.0.1:5000"})
    )
    assert config.mlflow.client.tracking_uri == "http://127.0.0.1:5000"
    assert config.names == ExperimentNamesConfig()


def test_system_config_uses_default_tracking_uri() -> None:
    config = SystemConfig(mlflow=MlflowTopologyConfig(client={}))
    assert config.mlflow.client.tracking_uri == "http://127.0.0.1:5000"


def test_system_config_is_frozen() -> None:
    config = SystemConfig()
    with pytest.raises(ValidationError):
        config.names = ExperimentNamesConfig(training="x", comparison="y")  # type: ignore[misc]


def test_system_config_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SystemConfig(unknown_field="not-allowed")  # type: ignore[call-arg]
