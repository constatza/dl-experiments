"""TOML parsing and Pydantic validation module.

This module is responsible for loading TOML files and validating them
using Pydantic models, providing clear error messages on validation failures.
"""

from __future__ import annotations

import tempfile
import tomllib
from pathlib import Path
from typing import Any

import tomli_w
from dlkit.tools.config import TrainingWorkflowSettings
from dlkit.tools.config.core.patching import patch_model
from dlkit.tools.io import load_settings
from dlkit.tools.io.config import load_config
from dlkit.tools import io as dlkit_io

from neuralls.platform.config.models.comparison import ComparisonConfig, parse_comparison_config
from neuralls.platform.config.models.data_models import DataConfigFile
from neuralls.shared.workspace import ExperimentWorkspace
from neuralls.platform.config.mlflow import normalize_model_mlflow
from neuralls.platform.config.paths import PathContext


def load_model_config(path: Path) -> TrainingWorkflowSettings:
    """Load and validate linear.toml file.

    Args:
        path: Path to model configuration TOML file.

    Returns:
        Validated ModelConfigFile instance.

    Raises:
        ConfigLoadError: If file cannot be read or validation fails.
    """
    raw = load_raw_toml(path)
    normalized = normalize_model_mlflow(raw, path)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=True) as tmp:
        tomli_w.dump(normalized, tmp)
        tmp.flush()
        return load_settings(Path(tmp.name))


def load_data_config(path: Path) -> DataConfigFile:
    """Load and validate data config TOML file.

    Args:
        path: Path to data configuration TOML file.

    Returns:
        Validated DataConfigFile instance.

    Raises:
        ConfigLoadError: If file cannot be read or validation fails.
    """
    raw = load_raw_toml(path)
    return DataConfigFile(**raw)


def load_comparison_config(path: Path) -> ComparisonConfig:
    """Load and validate comparison TOML file.

    Args:
        path: Path to comparison configuration TOML file.

    Returns:
        Parsed ComparisonConfig dataclass.

    Raises:
        ConfigLoadError: If file cannot be read or validation fails.
    """
    raw = load_raw_toml(path)
    return parse_comparison_config(raw)


def load_raw_toml(path: Path) -> dict[str, Any]:
    """Load TOML file as raw dictionary without validation.

    This function is provided for legacy compatibility and special cases
    where Pydantic validation is not desired.

    Args:
        path: Path to TOML file.

    Returns:
        Raw dictionary from TOML file.

    Raises:
        ConfigLoadError: If file cannot be read or parsed.
    """
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise exc


def build_settings(
    model_config_path: Path,
    workspace: ExperimentWorkspace,
    path_context: PathContext,
    force_mlflow_enabled: bool = False,
    base_settings: Any | None = None,
) -> Any:
    """Build dlkit settings with workspace and MLflow paths injected.

    Args:
        model_config_path: Path to model config TOML.
        workspace: Experiment workspace with data and run info.
        path_context: Resolved base paths (source of truth).
        base_settings: Optional pre-loaded settings instance to reuse.

    Returns:
        dlkit GeneralSettings with all paths configured.
    """
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

    Args:
        model_config_path: Path to model config TOML.
        workspace: Experiment workspace with data and run info.
        path_context: Resolved base paths (source of truth).
        mlflow_run_name: MLflow run name with timestamp (optional).

    Returns:
        dlkit InferenceWorkflowConfig with all paths configured.
    """
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

    run_name = mlflow_run_name if mlflow_run_name is not None else workspace.run_id

    if not settings.SESSION.inference:
        settings = patch_model(
            settings,
            {
                "SESSION": {
                    "inference": True,
                }
            },
        )

    updates: dict[str, Any] = {
        "PATHS": {
            "project_root": str(path_context.project_root),
            "processed_dir": str(path_context.processed_root),
            "output_dir": str(path_context.output_root),
        },
    }

    if getattr(settings, "MLFLOW", None) is not None or force_mlflow_enabled:
        updates["MLFLOW"] = {
            "experiment_name": mlflow_experiment_name or workspace.dataset_id,
            "run_name": run_name,
        }

    settings = patch_model(settings, updates)

    return settings
