"""Tests for runtime topology configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neuralls.configuration.system import (
    ExperimentNamesConfig,
    MlflowTopologyConfig,
    SystemConfig,
)


def test_system_config_minimal_valid() -> None:
    config = SystemConfig(
        mlflow=MlflowTopologyConfig(
            tracking_uri=f"sqlite:///{(Path('tests') / 'fixtures' / 'system.db').as_posix()}"
        )
    )
    assert config.mlflow.tracking_uri.endswith("tests/fixtures/system.db")
    assert config.names == ExperimentNamesConfig()


def test_system_config_uses_default_tracking_uri() -> None:
    config = SystemConfig(mlflow=MlflowTopologyConfig())
    assert config.mlflow.tracking_uri == MlflowTopologyConfig().tracking_uri


def test_system_config_is_frozen() -> None:
    config = SystemConfig()
    with pytest.raises(ValidationError):
        config.names = ExperimentNamesConfig(training="x", comparison="y")  # type: ignore[misc]


def test_system_config_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SystemConfig(unknown_field="not-allowed")  # type: ignore[call-arg]
