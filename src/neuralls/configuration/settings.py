"""Settings construction for dlkit integration.

Injects resolved paths into dlkit's GeneralSettings, including MLflow configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.tools.config import load_training_settings
from dlkit.tools.config.core.updater import update_settings

from neuralls.configuration.domain import ExperimentWorkspace
from neuralls.configuration.paths import PathContext


def build_settings(
    model_config_path: Path,
    workspace: ExperimentWorkspace,
    path_context: PathContext,
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

    Returns:
        dlkit GeneralSettings with all paths configured.
    """
    # Load base settings from dlkit
    settings = load_training_settings(str(model_config_path))

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
                    # Only inject experiment/run names
                    "experiment_name": workspace.dataset_id,
                    "run_name": workspace.run_id,
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
