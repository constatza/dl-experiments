"""DLKit job assembly for validated neuralls config models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.config import InferenceJobConfig, SearchJobConfig, TrainingJobConfig, load_job
from dlkit.infrastructure.config.core.patching import patch_model

from neuralls.platform.config.models.workspace import ExperimentWorkspace
from neuralls.platform.config.settings import NeurallsSettings

type ModelWorkflowSettings = TrainingJobConfig | SearchJobConfig
type AnyJobSettings = TrainingJobConfig | SearchJobConfig | InferenceJobConfig


def load_job_config(path: Path, settings: NeurallsSettings) -> AnyJobSettings:
    """Load and validate a DLKit job configuration file."""
    del settings
    return load_job(path)


def load_model_config(path: Path, settings: NeurallsSettings) -> AnyJobSettings:
    """Compatibility alias for the previous model-centric loader name."""
    return load_job_config(path, settings)


def build_settings(
    job_config_path: Path,
    workspace: ExperimentWorkspace,
    data_cfg: Any,
    settings: NeurallsSettings,
    output_override: Path | None = None,
    force_mlflow_enabled: bool = False,
    base_settings: ModelWorkflowSettings | None = None,
) -> ModelWorkflowSettings:
    """Build a DLKit training or search job with runtime paths injected."""
    del data_cfg, output_override
    dlkit_settings = (
        base_settings if base_settings is not None else load_job_config(job_config_path, settings)
    )
    if not isinstance(dlkit_settings, (TrainingJobConfig, SearchJobConfig)):
        raise TypeError(
            "Job configs must load as training or search workflows, "
            f"got {type(dlkit_settings).__name__}."
        )

    tracking_updates: dict[str, Any] = {}
    if getattr(dlkit_settings, "tracking", None) is None or force_mlflow_enabled:
        tracking_updates = {"tracking": {}}

    updates: dict[str, Any] = {
        "training": {
            "trainer": {
                "default_root_dir": str(workspace.root_dir),
            }
        },
        **tracking_updates,
    }
    return patch_model(dlkit_settings, updates)


def build_inference_settings(
    job_config_path: Path,
    workspace: ExperimentWorkspace,
    data_cfg: Any,
    settings: NeurallsSettings,
    output_override: Path | None = None,
    mlflow_run_name: str | None = None,
    mlflow_experiment_name: str | None = None,
    force_mlflow_enabled: bool = False,
) -> InferenceJobConfig:
    """Build DLKit inference settings with runtime paths injected."""
    del workspace, data_cfg, output_override, mlflow_run_name, mlflow_experiment_name
    loaded = load_job_config(job_config_path, settings)
    if isinstance(loaded, InferenceJobConfig):
        inference_settings = loaded
    else:
        inference_settings = load_job(job_config_path, run_type="predict")
        if not isinstance(inference_settings, InferenceJobConfig):
            raise TypeError(
                "Inference job configs must load as inference workflows, "
                f"got {type(inference_settings).__name__}."
            )

    if getattr(inference_settings, "tracking", None) is None or force_mlflow_enabled:
        return patch_model(inference_settings, {"tracking": {}})
    return inference_settings
