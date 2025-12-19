"""Settings construction module for dlkit GeneralSettings.

This module handles the construction and injection of settings into
dlkit's GeneralSettings objects, accepting typed Pydantic models instead of dicts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.tools.config import load_training_settings
from dlkit.tools.config.core.updater import update_settings

from ..data_models import DataConfigFile
from ..domain import ExperimentWorkspace
from ..models import DatamoduleConfig, ModelConfigFile
from ...mlflow_utils import resolve_mlflow_paths


def apply_datamodule_overrides(
    settings: Any,
    datamodule_cfg: DatamoduleConfig,
) -> Any:
    """Apply datamodule settings with persistent_workers logic.

    Args:
        settings: dlkit GeneralSettings object.
        datamodule_cfg: Validated datamodule configuration.

    Returns:
        Updated settings object.
    """
    dataloader_cfg = datamodule_cfg.dataloader.model_dump(exclude_none=True)

    # Auto-disable persistent_workers if num_workers == 0
    if dataloader_cfg.get("num_workers", 0) <= 0:
        dataloader_cfg.setdefault("persistent_workers", False)

    return update_settings(
        settings,
        {"DATAMODULE": {"dataloader": dataloader_cfg}}
    )


def inject_workspace_paths(
    settings: Any,
    workspace: ExperimentWorkspace,
) -> Any:
    """Inject workspace paths into settings.

    Args:
        settings: dlkit GeneralSettings object.
        workspace: Experiment workspace with directory paths.

    Returns:
        Updated settings object.
    """
    return update_settings(
        settings,
        {
            "PATHS": {
                "results_dir": str(workspace.root_dir),
                "figures_dir": str(workspace.figures_dir),
            },
            "TRAINING": {
                "trainer": {
                    "default_root_dir": str(workspace.root_dir),
                }
            },
        },
    )




def inject_mlflow_paths(
    settings: Any,
    model_cfg: ModelConfigFile,
    project_root: Path,
    workspace_root: Path,
) -> Any:
    """Resolve and inject MLflow configuration.

    Args:
        settings: dlkit GeneralSettings object.
        model_cfg: Validated model configuration.
        project_root: Project root directory.
        workspace_root: Workspace root directory for artifacts.

    Returns:
        Updated settings object.
    """
    # MLflow configuration loaded from TOML - client connects to server at tracking_uri
    return settings


def build_settings(
    model_path: Path,
    model_cfg: ModelConfigFile,
    workspace: ExperimentWorkspace,
    project_root: Path,
    processed_root: Path,
) -> Any:
    """Orchestrate complete settings construction.

    This function loads base settings from dlkit and applies all configuration
    overrides in the correct order.

    Args:
        model_path: Path to model config file (for dlkit loader).
        model_cfg: Validated model configuration.
        workspace: Experiment workspace.
        project_root: Project root directory.
        processed_root: Processed data directory.

    Returns:
        Fully configured dlkit GeneralSettings object.
    """
    # Load base settings from dlkit
    settings = load_training_settings(str(model_path))

    # Apply overrides in sequence
    settings = apply_datamodule_overrides(settings, model_cfg.DATAMODULE)
    settings = inject_workspace_paths(settings, workspace)
    settings = inject_mlflow_paths(settings, model_cfg, project_root, workspace.root_dir)

    # Inject resolved roots
    settings = update_settings(
        settings,
        {
            "PATHS": {
                "project_root": str(project_root),
                "processed_dir": str(processed_root),
            }
        }
    )

    return settings
