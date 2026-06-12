"""DLKit workflow assembly for validated neuralls config models."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import tomli_w
from dlkit.infrastructure.config.core.patching import patch_model
from dlkit.infrastructure.config.factories import load_settings
from dlkit.infrastructure.config.workflow_configs import (
    InferenceWorkflowConfig,
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)

from neuralls.platform.config.loaders import load_raw_toml
from neuralls.platform.config.model_mlflow import normalize_model_mlflow
from neuralls.platform.config.models.workspace import ExperimentWorkspace
from neuralls.platform.config.settings import NeurallsSettings

type ModelWorkflowSettings = TrainingWorkflowConfig | OptimizationWorkflowConfig


def _prepare_model_config(path: Path) -> dict[str, Any]:
    """Normalize one model config before DLKit workflow validation."""
    raw = load_raw_toml(path)
    return normalize_model_mlflow(raw, path)


def load_model_config(path: Path, settings: NeurallsSettings) -> ModelWorkflowSettings:
    """Load and validate a DLKit model configuration file."""
    del settings
    normalized = _prepare_model_config(path)
    with tempfile.TemporaryDirectory(prefix="neuralls_model_cfg_") as tmp_dir:
        tmp_path = Path(tmp_dir) / path.name
        with tmp_path.open("wb") as tmp:
            tomli_w.dump(normalized, tmp)
        loaded = load_settings(tmp_path)

    if not isinstance(loaded, (TrainingWorkflowConfig, OptimizationWorkflowConfig)):
        raise TypeError(
            "Model configs must load as training or optimization workflows, "
            f"got {type(loaded).__name__}."
        )
    return loaded


def build_settings(
    model_config_path: Path,
    workspace: ExperimentWorkspace,
    data_cfg: Any,
    settings: NeurallsSettings,
    output_override: Path | None = None,
    force_mlflow_enabled: bool = False,
    base_settings: ModelWorkflowSettings | None = None,
) -> ModelWorkflowSettings:
    """Build DLKit training or optimization settings with runtime paths injected."""
    del data_cfg, output_override
    dlkit_settings = (
        base_settings
        if base_settings is not None
        else load_model_config(model_config_path, settings)
    )
    if getattr(dlkit_settings, "MLFLOW", None) is None:
        dlkit_settings = patch_model(dlkit_settings, {"MLFLOW": {}})

    updates: dict[str, Any] = {
        "TRAINING": {
            "trainer": {
                "default_root_dir": workspace.root_dir,
            }
        },
    }
    if getattr(dlkit_settings, "MLFLOW", None) is not None or force_mlflow_enabled:
        updates["MLFLOW"] = {}
    return patch_model(dlkit_settings, updates)


def build_inference_settings(
    model_config_path: Path,
    workspace: ExperimentWorkspace,
    data_cfg: Any,
    settings: NeurallsSettings,
    output_override: Path | None = None,
    mlflow_run_name: str | None = None,
    mlflow_experiment_name: str | None = None,
    force_mlflow_enabled: bool = False,
) -> InferenceWorkflowConfig:
    """Build DLKit inference settings with runtime paths injected."""
    del (
        workspace,
        data_cfg,
        settings,
        output_override,
        mlflow_run_name,
        mlflow_experiment_name,
        force_mlflow_enabled,
    )

    inference_excluded = {"TRAINING", "MLFLOW", "OPTUNA"}
    toml_data = _prepare_model_config(model_config_path)
    inference_data = {k: v for k, v in toml_data.items() if k not in inference_excluded}
    session = dict(inference_data.get("SESSION") or {})
    session["workflow"] = "inference"
    inference_data["SESSION"] = session
    return InferenceWorkflowConfig.model_validate(inference_data)
