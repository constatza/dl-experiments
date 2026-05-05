"""Legacy runtime topology models.

Runtime topology now lives in the top-level case config. This module remains as
a compatibility re-export surface for internal imports.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from neuralls.platform.config.models.experiments import (
    ExperimentNamesConfig,
    MlflowTopologyConfig,
)


class SystemConfig(BaseModel):
    """Runtime topology config (no filesystem output root)."""

    mlflow: MlflowTopologyConfig = Field(default_factory=MlflowTopologyConfig)
    names: ExperimentNamesConfig = Field(default_factory=ExperimentNamesConfig)

    model_config = ConfigDict(extra="forbid", frozen=True)
