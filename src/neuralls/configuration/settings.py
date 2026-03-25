"""Settings construction for dlkit integration.

Injects resolved paths into dlkit's GeneralSettings, including MLflow configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.tools.config.core.patching import patch_model
from dlkit.tools.io.config import load_config
from dlkit.tools import io as dlkit_io

from neuralls.configuration.domain import ExperimentWorkspace
from neuralls.configuration.paths import PathContext
from neuralls.io.toml_loader import load_model_config


def build_settings(
    model_config_path: Path,
    workspace: ExperimentWorkspace,
    path_context: PathContext,
    force_mlflow_enabled: bool = False,
    base_settings: Any | None = None,
) -> Any:
    """Build dlkit settings with workspace and MLflow paths injected.

    This function:
    1. Loads base settings from model config (dlkit)
    2. Injects workspace root for training artifacts
    3. Enables MLflow when required by topology
    4. Injects processed data path

    Args:
        model_config_path: Path to model config TOML.
        workspace: Experiment workspace with data and run info.
        path_context: Resolved base paths (source of truth).
        base_settings: Optional pre-loaded settings instance to reuse.

    Returns:
        dlkit GeneralSettings with all paths configured.
    """
    # Load base settings from dlkit
    settings = base_settings if base_settings is not None else load_model_config(model_config_path)
    if getattr(settings, "MLFLOW", None) is None:
        settings = patch_model(settings, {"MLFLOW": {}})

    updates: dict[str, Any] = {
        "TRAINING": {
            "trainer": {
                "default_root_dir": workspace.root_dir,
            }
        },
        "PATHS": {
            "project_root": str(path_context.project_root),
            "processed_dir": str(path_context.processed_root),
            "output_dir": str(path_context.output_root),
        },
    }

    if getattr(settings, "MLFLOW", None) is not None or force_mlflow_enabled:
        updates["MLFLOW"] = {}

    settings = patch_model(settings, updates)

    return settings


def build_inference_settings(
    model_config_path: Path,
    workspace: ExperimentWorkspace,
    path_context: PathContext,
    mlflow_run_name: str | None = None,
    mlflow_experiment_name: str | None = None,
    force_mlflow_enabled: bool = False,
) -> Any:
    """Build dlkit inference settings with workspace and MLflow paths injected.

    This function:
    1. Loads base settings from model config (InferenceWorkflowConfig)
    2. Ensures SESSION.inference=True
    3. Injects MLflow tracking URI and artifact location (optional)
    4. Injects processed data path

    Note: Unlike training, inference does NOT require DATASET/DATAMODULE sections.
    Transforms are loaded from checkpoint metadata.

    Args:
        model_config_path: Path to model config TOML.
        workspace: Experiment workspace with data and run info.
        path_context: Resolved base paths (source of truth).
        mlflow_run_name: MLflow run name with timestamp (optional, defaults to workspace.run_id).

    Returns:
        dlkit InferenceWorkflowConfig with all paths configured.
    """
    # Load base inference settings from dlkit
    # Strip training-specific sections not accepted by InferenceWorkflowConfig
    _INFERENCE_EXCLUDED = {"TRAINING", "MLFLOW", "OPTUNA"}
    toml_data = load_config(model_config_path, raw=True)
    inference_data = {k: v for k, v in toml_data.items() if k not in _INFERENCE_EXCLUDED}
    from dlkit.tools.config.workflow_configs import InferenceWorkflowConfig

    settings = InferenceWorkflowConfig.model_validate(inference_data)
    sync_session_root_to_environment = getattr(
        dlkit_io.config,
        "_sync_session_root_to_environment",
    )
    sync_session_root_to_environment(settings)

    # Use provided MLflow run name or default to workspace.run_id
    run_name = mlflow_run_name if mlflow_run_name is not None else workspace.run_id

    # Ensure SESSION.inference=True
    if not settings.SESSION.inference:
        settings = patch_model(
            settings,
            {
                "SESSION": {
                    "inference": True,
                }
            },
        )

    # Inject optional sections (MLflow, paths) if they exist in config
    updates: dict[str, Any] = {
        "PATHS": {
            "project_root": str(path_context.project_root),
            "processed_dir": str(path_context.processed_root),
            "output_dir": str(path_context.output_root),
        },
    }

    # Only inject MLflow if enabled in config or explicitly forced.
    if getattr(settings, "MLFLOW", None) is not None or force_mlflow_enabled:
        updates["MLFLOW"] = {
            "experiment_name": mlflow_experiment_name or workspace.dataset_id,
            "run_name": run_name,
        }

    settings = patch_model(settings, updates)

    return settings
