"""Domain models for the configuration system.

These classes define the core data structures for experiments, strictly separating
inputs (Specifications) from outputs (Workspaces) and execution contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from dlkit import GeneralSettings


class ExperimentSpec(BaseModel):
    """Static definition of an experiment's inputs (Immutable)."""
    id: str = Field(..., description="Unique identifier from experiments.toml")
    model_config_path: Path = Field(..., description="Absolute path to model.toml")
    data_config_path: Path = Field(..., description="Absolute path to resolved data config")
    solver_config_path: Path = Field(..., description="Absolute path to solver.toml")
    
    class Config:
        frozen = True


@dataclass(frozen=True)
class ExperimentWorkspace:
    """Resolved filesystem layout for artifacts (Outputs)."""
    root_dir: Path          # The specific root for this experiment
    data_dir: Path          # Directory containing input data (normalized.npz)
    checkpoint_dir: Path    # root_dir / checkpoints
    figures_dir: Path       # root_dir / figures
    predictions_dir: Path   # root_dir / predictions
    run_id: str             # The Model ID (from config) used for MLflow tracking


@dataclass(frozen=True)
class RunnableExperiment:
    """Fully resolved experiment ready for execution.
    
    Combines the 'what' (Spec), the 'where' (Workspace), and the 
    'how' (Settings/Context).
    """
    spec: ExperimentSpec
    workspace: ExperimentWorkspace
    settings: GeneralSettings 


@dataclass(frozen=True)
class ExperimentBatch:
    """Complete definition of the requested workload."""
    global_output_dir: Path
    experiments: list[RunnableExperiment]
