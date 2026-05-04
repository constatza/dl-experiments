"""Domain models for the experiment configuration system.

These classes define the core data structures for experiments, strictly separating
inputs (Specifications) from outputs (Workspaces) and execution contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExperimentSpec(BaseModel):
    """Static definition of an experiment's inputs (Immutable).

    Attributes:
        experiment_id: Stable experiment identifier.
        experiment_display_name: Human-facing experiment label.
        model_config_path: Path to model config TOML.
        data_config_path: Path to data config TOML.
        checkpoint_path: Optional explicit checkpoint path.
    """

    experiment_id: str = Field(..., description="Stable experiment identifier")
    experiment_display_name: str = Field(..., description="Human-facing experiment label")
    dataset_registry_id: str | None = Field(default=None, description="Registry dataset identifier")
    dataset_display_name: str | None = Field(default=None, description="Human-facing dataset label")
    model_registry_id: str | None = Field(default=None, description="Registry model identifier")
    model_display_name: str | None = Field(default=None, description="Human-facing model label")
    model_config_path: Path = Field(..., description="Path to model config")
    data_config_path: Path = Field(..., description="Path to data config")
    checkpoint_path: Path | None = Field(
        default=None,
        description="Explicit checkpoint path (optional)",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class ExperimentWorkspace:
    """Resolved filesystem layout for experiment artifacts.

    Attributes:
        dataset_id: Dataset identifier (for organizing experiments).
        run_id: Model/run identifier (for organizing within dataset).
        root_dir: Experiment root directory (for this specific run).
        data_dir: Directory containing input data artifacts (manifest + arrays).
    """

    dataset_id: str
    run_id: str
    root_dir: Path
    data_dir: Path

    @property
    def checkpoint_dir(self) -> Path:
        """Checkpoint directory under root."""
        return self.root_dir / "checkpoints"

    @property
    def figures_dir(self) -> Path:
        """Figures directory under root."""
        return self.root_dir / "figures"

    @property
    def predictions_dir(self) -> Path:
        """Predictions directory under root."""
        return self.root_dir / "predictions"


@dataclass(frozen=True)
class RunnableExperiment:
    """Fully resolved experiment ready for execution.

    Combines the 'what' (Spec), the 'where' (Workspace), and the
    'how' (Settings).

    Attributes:
        spec: Experiment specification.
        workspace: Filesystem workspace.
        settings: DLKit workflow settings with injected paths.
    """

    spec: ExperimentSpec
    workspace: ExperimentWorkspace
    settings: Any  # DLKit workflow settings (avoid import)


@dataclass(frozen=True)
class ExperimentBatch:
    """Complete definition of a batch workload.

    Attributes:
        output_root: Master output root for all experiments.
        experiments: List of runnable experiments.
    """

    output_root: Path
    experiments: list[RunnableExperiment]
