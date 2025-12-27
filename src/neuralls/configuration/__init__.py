"""Configuration loading and domain models for neuralls experiments.

For loading experiments, import directly from submodules:
    from neuralls.configuration.loader import load_experiment, load_batch
"""

from neuralls.configuration.domain import (
    ExperimentBatch,
    ExperimentSpec,
    ExperimentWorkspace,
    RunnableExperiment,
)
from neuralls.configuration.paths import PathContext, build_path_context
from neuralls.configuration.services import WorkspaceFactory

__all__ = [
    # Domain models
    "ExperimentSpec",
    "ExperimentWorkspace",
    "RunnableExperiment",
    "ExperimentBatch",
    # Path resolution
    "PathContext",
    "build_path_context",
    # Services
    "WorkspaceFactory",
]
