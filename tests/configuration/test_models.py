"""Tests for Pydantic configuration models.

Tests cover validation of model.toml structure using Pydantic models.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.configuration.models import (
    DataloaderConfig,
    DatamoduleConfig,
    MLflowClientConfig,
    MLflowConfig,
    MLflowServerConfig,
    ModelConfig,
    ModelConfigFile,
    PathsConfig,
    SessionConfig,
)


class TestSessionConfig:
    """Tests for SessionConfig validation."""

    def test_session_config_defaults(self):
        """Test SessionConfig with default values."""
        config = SessionConfig()
        assert config.seed == 42
        assert config.precision == "float64"
        assert config.name is None

    def test_session_config_custom_values(self):
        """Test SessionConfig with custom values."""
        config = SessionConfig(seed=123, precision="float32", name="test_session")
        assert config.seed == 123
        assert config.precision == "float32"
        assert config.name == "test_session"

    def test_session_config_immutable(self):
        """Test that SessionConfig is frozen."""
        config = SessionConfig()
        with pytest.raises((ValidationError, AttributeError)):
            config.seed = 999  # type: ignore


class TestModelConfig:
    """Tests for ModelConfig validation."""

    def test_model_config_required_fields(self):
        """Test ModelConfig with required fields."""
        config = ModelConfig(name="TestModel", module_path="test.module")
        assert config.name == "TestModel"
        assert config.module_path == "test.module"
        assert config.hidden_size is None

    def test_model_config_missing_required_field(self):
        """Test ModelConfig fails without required fields."""
        with pytest.raises(ValidationError):
            ModelConfig(name="TestModel")  # type: ignore

    def test_model_config_with_optional_fields(self):
        """Test ModelConfig with optional fields."""
        config = ModelConfig(
            name="TestModel",
            module_path="test.module",
            hidden_size=256,
            num_layers=3,
            dropout=0.1,
        )
        assert config.hidden_size == 256
        assert config.num_layers == 3
        assert config.dropout == 0.1

    def test_model_config_validation_constraints(self):
        """Test ModelConfig field validation."""
        # Negative hidden_size should fail
        with pytest.raises(ValidationError):
            ModelConfig(
                name="TestModel", module_path="test.module", hidden_size=-1
            )

        # Dropout > 1.0 should fail
        with pytest.raises(ValidationError):
            ModelConfig(
                name="TestModel", module_path="test.module", dropout=1.5
            )


class TestMLflowConfig:
    """Tests for MLflow configuration models."""

    def test_mlflow_config_defaults(self):
        """Test MLflowConfig with defaults."""
        config = MLflowConfig()
        assert config.enabled is False
        assert config.server.backend_store_uri is None
        assert config.client.tracking_uri is None

    def test_mlflow_config_full(self):
        """Test MLflowConfig with all fields."""
        config = MLflowConfig(
            enabled=True,
            server=MLflowServerConfig(
                backend_store_uri="sqlite:///mlflow.db",
                artifacts_destination="/tmp/artifacts",
            ),
            client=MLflowClientConfig(
                tracking_uri="http://localhost:5000",
                experiment_name="test_exp",
            ),
        )
        assert config.enabled is True
        assert config.server.backend_store_uri == "sqlite:///mlflow.db"
        assert config.client.tracking_uri == "http://localhost:5000"


class TestDatamoduleConfig:
    """Tests for DatamoduleConfig validation."""

    def test_datamodule_config_defaults(self):
        """Test DatamoduleConfig with defaults."""
        config = DatamoduleConfig()
        assert config.name == "InMemoryModule"
        assert config.dataloader.num_workers == 0
        assert config.dataloader.batch_size == 32
        assert config.dataloader.pin_memory is True

    def test_dataloader_config_validation(self):
        """Test DataloaderConfig field validation."""
        # Negative num_workers should fail
        with pytest.raises(ValidationError):
            DataloaderConfig(num_workers=-1)

        # Zero batch_size should fail
        with pytest.raises(ValidationError):
            DataloaderConfig(batch_size=0)


class TestModelConfigFile:
    """Tests for complete ModelConfigFile validation."""

    def test_model_config_file_minimal(self):
        """Test ModelConfigFile with minimal required fields."""
        config = ModelConfigFile(
            MODEL=ModelConfig(name="TestModel", module_path="test.module")
        )
        assert config.MODEL.name == "TestModel"
        assert config.SESSION.seed == 42  # Default
        assert config.PATHS.project_root is None  # Default
        assert config.MLFLOW.enabled is False  # Default

    def test_model_config_file_full(self):
        """Test ModelConfigFile with all sections."""
        config = ModelConfigFile(
            SESSION=SessionConfig(seed=123, precision="float32"),
            MODEL=ModelConfig(
                name="TestModel", module_path="test.module", hidden_size=512
            ),
            PATHS=PathsConfig(project_root="/custom/root"),
            MLFLOW=MLflowConfig(enabled=True),
            DATAMODULE=DatamoduleConfig(
                name="CustomModule",
                dataloader=DataloaderConfig(batch_size=64, num_workers=4),
            ),
        )
        assert config.SESSION.seed == 123
        assert config.MODEL.hidden_size == 512
        assert config.PATHS.project_root == "/custom/root"
        assert config.MLFLOW.enabled is True
        assert config.DATAMODULE.dataloader.batch_size == 64

    def test_model_config_file_with_training_dict(self):
        """Test ModelConfigFile allows TRAINING as dict."""
        config = ModelConfigFile(
            MODEL=ModelConfig(name="TestModel", module_path="test.module"),
            TRAINING={"trainer": {"max_epochs": 100}},
        )
        assert config.TRAINING == {"trainer": {"max_epochs": 100}}

    def test_model_config_file_immutable(self):
        """Test that ModelConfigFile is frozen."""
        config = ModelConfigFile(
            MODEL=ModelConfig(name="TestModel", module_path="test.module")
        )
        with pytest.raises((ValidationError, AttributeError)):
            config.SESSION.seed = 999  # type: ignore
