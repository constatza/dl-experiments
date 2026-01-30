"""Settings construction for dlkit integration.

Injects resolved paths into dlkit's GeneralSettings, including MLflow configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.tools.io import load_settings
from dlkit.tools.config.core.updater import update_settings
from dlkit.tools.io.config import load_inference_config_eager

from neuralls.configuration.domain import ExperimentWorkspace
from neuralls.configuration.paths import PathContext


def build_settings(
    model_config_path: Path,
    workspace: ExperimentWorkspace,
    path_context: PathContext,
    mlflow_run_name: str | None = None,
) -> Any:
    """Build dlkit settings with workspace and MLflow paths injected.

    This function:
    1. Loads base settings from model config (dlkit)
    2. Injects workspace root for training artifacts
    3. Injects MLflow tracking URI and artifact location (from output_root)
    4. Injects processed data path

    Args:
        model_config_path: Path to model config TOML.
        workspace: Experiment workspace with data and run info.
        path_context: Resolved base paths (source of truth).
        mlflow_run_name: MLflow run name with timestamp (optional, defaults to workspace.run_id).

    Returns:
        dlkit GeneralSettings with all paths configured.
    """
    # Load base settings from dlkit
    settings = load_settings(str(model_config_path))

    # Use provided MLflow run name or default to workspace.run_id
    run_name = mlflow_run_name if mlflow_run_name is not None else workspace.run_id

    # Inject paths into settings
    settings = update_settings(
        settings,
        {
            "TRAINING": {
                "trainer": {
                    # Workspace root for checkpoints/logs
                    "default_root_dir": workspace.root_dir,
                }
            },
            "MLFLOW": {
                # MLflow tracking database (derived from output_root)
                "server": {
                    "backend_store_uri": path_context.mlflow_tracking_uri,
                    "artifacts_destination": path_context.mlflow_artifact_location,
                },
                "client": {
                    # Client tracking_uri comes from config (http:// URL)
                    # Only inject experiment/run names (run_name has timestamp for uniqueness)
                    "experiment_name": workspace.dataset_id,
                    "run_name": run_name,
                },
            },
            "PATHS": {
                "project_root": str(path_context.project_root),
                "processed_dir": str(path_context.processed_root),
                "output_dir": str(path_context.output_root),
            },
        },
    )

    return settings


def build_inference_settings(
    model_config_path: Path,
    workspace: ExperimentWorkspace,
    path_context: PathContext,
    mlflow_run_name: str | None = None,
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
    settings = load_inference_config_eager(str(model_config_path))

    # Use provided MLflow run name or default to workspace.run_id
    run_name = mlflow_run_name if mlflow_run_name is not None else workspace.run_id

    # Ensure SESSION.inference=True
    if not settings.SESSION.inference:
        settings = update_settings(
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

    # Only inject MLflow if enabled in config
    if settings.MLFLOW is not None:
        updates["MLFLOW"] = {
            # MLflow tracking database (derived from output_root)
            "server": {
                "backend_store_uri": path_context.mlflow_tracking_uri,
                "artifacts_destination": path_context.mlflow_artifact_location,
            },
            "client": {
                # Client tracking_uri comes from config (http:// URL)
                # Only inject experiment/run names (run_name has timestamp for uniqueness)
                "experiment_name": workspace.dataset_id,
                "run_name": run_name,
            },
        }

    settings = update_settings(settings, updates)

    return settings
