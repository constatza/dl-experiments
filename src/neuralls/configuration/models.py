"""Pydantic models for model configuration validation.

This module provides Pydantic models for validating linear.toml structure.
Separation: Pydantic models (validation) → used directly or converted to runtime types.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SessionConfig(BaseModel):
    """Validates [SESSION] section from linear.toml."""

    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    precision: str = Field(
        default="float64",
        description="Computation precision (float32, float64)",
    )
    name: str | None = Field(
        default=None,
        description="Session/experiment name",
    )

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class ModelConfig(BaseModel):
    """Validates [MODEL] section from linear.toml."""

    name: str = Field(
        ...,
        description="Model class name",
    )
    module_path: str = Field(
        ...,
        description="Python module path to model class",
    )
    hidden_size: int | None = Field(
        default=None,
        description="Hidden layer size",
        ge=1,
    )
    num_layers: int | None = Field(
        default=None,
        description="Number of layers",
        ge=1,
    )
    dropout: float | None = Field(
        default=None,
        description="Dropout rate",
        ge=0.0,
        le=1.0,
    )

    model_config = ConfigDict(
        extra="allow",  # Allow model-specific parameters
        frozen=True,
    )


class PathsConfig(BaseModel):
    """Validates [PATHS] section from linear.toml."""

    project_root: str | None = Field(
        default=None,
        description="Override for project root directory",
    )
    results_dir: str | None = Field(
        default=None,
        description="Override for results directory",
    )
    figures_dir: str | None = Field(
        default=None,
        description="Override for figures directory",
    )
    processed_dir: str | None = Field(
        default=None,
        description="Override for processed data directory",
    )

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class MLflowServerConfig(BaseModel):
    """Validates [MLFLOW.server] subsection."""

    backend_store_uri: str | None = Field(
        default=None,
        description="MLflow backend store URI (database)",
    )
    artifacts_destination: str | None = Field(
        default=None,
        description="MLflow artifacts storage path",
    )

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class MLflowClientConfig(BaseModel):
    """Validates [MLFLOW.client] subsection."""

    tracking_uri: str | None = Field(
        default=None,
        description="MLflow tracking server URI",
    )
    experiment_name: str | None = Field(
        default=None,
        description="MLflow experiment name",
    )
    run_name: str | None = Field(
        default=None,
        description="MLflow run name",
    )

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class MLflowConfig(BaseModel):
    """Validates [MLFLOW] section from linear.toml."""

    enabled: bool = Field(
        default=False,
        description="Enable MLflow tracking",
    )
    server: MLflowServerConfig = Field(
        default_factory=MLflowServerConfig,
        description="MLflow server configuration",
    )
    client: MLflowClientConfig = Field(
        default_factory=MLflowClientConfig,
        description="MLflow client configuration",
    )

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class DataloaderConfig(BaseModel):
    """Validates [DATAMODULE.dataloader] subsection."""

    num_workers: int = Field(
        default=0,
        description="Number of data loader workers",
        ge=0,
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for training",
        ge=1,
    )
    pin_memory: bool = Field(
        default=True,
        description="Pin memory for faster GPU transfer",
    )
    shuffle: bool = Field(
        default=True,
        description="Shuffle data during training",
    )
    persistent_workers: bool | None = Field(
        default=None,
        description="Keep workers alive between epochs (auto-disabled if num_workers=0)",
    )

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class DatamoduleConfig(BaseModel):
    """Validates [DATAMODULE] section from linear.toml."""

    name: str = Field(
        default="InMemoryModule",
        description="DataModule class name",
    )
    dataloader: DataloaderConfig = Field(
        default_factory=DataloaderConfig,
        description="Dataloader configuration",
    )

    model_config = ConfigDict(
        extra="allow",  # Allow module-specific parameters
        frozen=True,
    )


class ModelConfigFile(BaseModel):
    """Complete linear.toml structure with validation.

    This model validates the top-level sections of a model configuration file.
    Sections like TRAINING, DATASET, and OPTUNA use extra="allow" to support
    dlkit's dynamic configuration structure.
    """

    SESSION: SessionConfig = Field(
        default_factory=SessionConfig,
        description="Session configuration",
    )
    MODEL: ModelConfig = Field(
        ...,
        description="Model architecture configuration",
    )
    PATHS: PathsConfig = Field(
        default_factory=PathsConfig,
        description="Path overrides",
    )
    MLFLOW: MLflowConfig = Field(
        default_factory=MLflowConfig,
        description="MLflow tracking configuration",
    )
    DATAMODULE: DatamoduleConfig = Field(
        default_factory=DatamoduleConfig,
        description="DataModule configuration",
    )
    # TRAINING, DATASET, OPTUNA remain as dict for dlkit compatibility
    # These are validated by dlkit's own schema and have dynamic structure
    TRAINING: dict[str, Any] | None = Field(
        default=None,
        description="Training configuration (dlkit-managed)",
    )
    DATASET: dict[str, Any] | None = Field(
        default=None,
        description="Dataset configuration (dlkit-managed)",
    )
    OPTUNA: dict[str, Any] | None = Field(
        default=None,
        description="Optuna hyperparameter tuning configuration",
    )

    model_config = ConfigDict(
        extra="allow",  # Allow additional top-level sections
        frozen=True,
    )
