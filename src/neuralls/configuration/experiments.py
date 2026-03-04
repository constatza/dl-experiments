"""Pydantic models for experiments.toml configuration.

Used by ``train_batch()`` to validate the experiments config before
training and comparison phases.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from neuralls.configuration.comparison import ComparisonsTrackingConfig


class MlflowClientTopologyConfig(BaseModel):
    """MLflow client connection settings.

    Attributes:
        tracking_uri: MLflow tracking server URI.
    """

    tracking_uri: str = Field(default="http://127.0.0.1:5000", min_length=1)
    model_config = ConfigDict(extra="forbid", frozen=True)


class MlflowTopologyConfig(BaseModel):
    """MLflow topology configuration.

    Attributes:
        client: Client connection settings.
    """

    client: MlflowClientTopologyConfig = Field(default_factory=MlflowClientTopologyConfig)
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentNamesConfig(BaseModel):
    """MLflow experiment names configuration.

    Attributes:
        training: Name for training experiments.
        comparison: Name for comparison experiments.
    """

    training: str = "neuralls-training"
    comparison: str = "neuralls-comparisons"
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentEntry(BaseModel):
    """Single experiment entry from the ``[[experiment]]`` TOML table.

    Attributes:
        id: Unique identifier for the experiment.
        dataset: Dataset config name (without .toml extension).
        model: Model config name (without .toml extension).
    """

    id: str
    dataset: str
    model: str

    model_config = ConfigDict(frozen=True)


class RunEntry(BaseModel):
    """Single run entry from the ``[[run]]`` TOML table.

    Uses direct relative config paths instead of short name references.

    Attributes:
        id: Unique identifier for the run.
        model: Relative path to model config (e.g. ``models/tiny-linear.toml``).
        data: Relative path to data config (e.g. ``datasets/tiny-data.toml``).
    """

    id: str
    model: str = Field(alias="model_config")
    data: str = Field(alias="data_config")

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class ExperimentsConfig(BaseModel):
    """Top-level experiments.toml configuration.

    Attributes:
        output_dir: Training output directory (optional; derived from mlflow if absent).
        comparisons: MLflow config for comparisons DB (optional; derived if absent).
        experiment: List of experiment entries (``[[experiment]]`` format).
        run: List of run entries (``[[run]]`` format with direct config paths).
        project_root: Optional project root path.
        mlflow: MLflow topology config (tracking URI, etc.).
        names: MLflow experiment names for training and comparison.
    """

    output_dir: Path | None = Field(default=None, description="Training output directory")
    comparisons: ComparisonsTrackingConfig | None = Field(
        default=None, description="MLflow config for the comparisons DB"
    )
    experiment: list[ExperimentEntry] = Field(default_factory=list)
    run: list[RunEntry] = Field(default_factory=list)
    project_root: Path | None = Field(default=None)
    mlflow: MlflowTopologyConfig = Field(default_factory=MlflowTopologyConfig)
    names: ExperimentNamesConfig = Field(default_factory=ExperimentNamesConfig)

    model_config = ConfigDict(extra="allow", frozen=True)
