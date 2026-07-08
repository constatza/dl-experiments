"""Job loading boundary for DLKit-backed assignment composition."""

from __future__ import annotations

from pathlib import Path

from neuralls.composition.assignments._job_types import AnyJobConfig
from neuralls.platform.config.dlkit_bridge import load_job_config
from neuralls.platform.config.settings import NeurallsSettings


def load_experiment_job(
    job_config_path: Path,
    settings: NeurallsSettings,
) -> AnyJobConfig:
    """Load one typed lower-case DLKit job for assignment composition."""
    return load_job_config(job_config_path, settings)
