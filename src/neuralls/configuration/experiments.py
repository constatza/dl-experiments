"""Pydantic models for experiments.toml configuration.

Used by ``train_batch()`` to validate the experiments config before
training and comparison phases.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from neuralls.configuration.comparison import ComparisonsTrackingConfig


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


class ExperimentsConfig(BaseModel):
    """Top-level experiments.toml configuration.

    Attributes:
        output_dir: Training output directory.
        comparisons: MLflow config for the comparisons DB (required).
        experiment: List of experiment entries.
        project_root: Optional project root path.
    """

    output_dir: Path = Field(..., description="Training output directory")
    comparisons: ComparisonsTrackingConfig = Field(
        ..., description="MLflow config for the comparisons DB (required)"
    )
    experiment: list[ExperimentEntry] = Field(default_factory=list)
    project_root: Path | None = Field(default=None)

    model_config = ConfigDict(extra="allow", frozen=True)
