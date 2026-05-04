"""TOML parsing and workflow-config validation helpers."""

from __future__ import annotations

import tempfile
import tomllib
from pathlib import Path
from typing import Any, cast

import tomli_w
from dlkit.infrastructure.config.core.patching import patch_model
from dlkit.infrastructure.config.environment import sync_session_root_to_environment
from dlkit.infrastructure.config.factories import load_settings
from dlkit.infrastructure.config.workflow_configs import (
    InferenceWorkflowConfig,
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)

from neuralls.platform.config.models.comparison import ComparisonConfig, parse_comparison_config
from neuralls.platform.config.models.data_models import DataConfigFile
from neuralls.platform.config.models.workspace import ExperimentWorkspace
from neuralls.platform.config.mlflow import normalize_model_mlflow
from neuralls.platform.config.path_utils import (
    expand_nested_path_values,
    expand_path_placeholders,
    resolve_local_path,
    resolve_path_expression,
)
from neuralls.platform.config.paths import PathContext

type ModelWorkflowSettings = TrainingWorkflowConfig | OptimizationWorkflowConfig

_DATA_CONFIG_PATH_KEYS = frozenset(
    {
        "artifacts_destination",
        "case_path",
        "checkpoint_path",
        "config_path",
        "data_config_path",
        "data_dir",
        "matrix_file",
        "matrix_path",
        "model_config",
        "model_config_path",
        "output_dir",
        "path",
        "processed_dir",
        "project_root",
        "rhs_glob",
        "rhs_path",
        "rhs_pattern",
        "solutions_glob",
        "solutions_path",
        "tracking_uri",
    }
)

_COMPARISON_CONFIG_PATH_KEYS = frozenset(
    {
        "checkpoint_path",
        "config_path",
        "data_config_path",
        "matrix_path",
        "model_config_path",
        "rhs_path",
    }
)

_EXPERIMENTS_CONFIG_PATH_KEYS = frozenset(
    {
        "artifacts_destination",
        "checkpoint_path",
        "data_config",
        "model_config",
        "output_dir",
        "path",
        "project_root",
        "tracking_uri",
    }
)


def _normalize_data_config_paths(raw: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Expand env placeholders and anchor dataset config paths to the config directory."""
    normalized = expand_nested_path_values(raw, path_keys=_DATA_CONFIG_PATH_KEYS)
    config_dir = config_path.resolve().parent

    source = dict(normalized.get("source") or {})
    for key in ("case_path", "matrix_path", "rhs_path", "solutions_path"):
        value = source.get(key)
        if isinstance(value, str):
            source[key] = resolve_path_expression(value, base_dir=config_dir)
    matrix_file = source.get("matrix_file")
    if isinstance(matrix_file, str):
        source["matrix_file"] = expand_path_placeholders(matrix_file)
    rhs_pattern = source.get("rhs_pattern")
    if isinstance(rhs_pattern, str):
        source["rhs_pattern"] = expand_path_placeholders(rhs_pattern)
    normalized["source"] = source

    generation = dict(normalized.get("generation") or {})
    strategies = []
    for entry in generation.get("strategy", []):
        if not isinstance(entry, dict):
            strategies.append(entry)
            continue
        strategy = dict(entry)
        for key in ("solutions_glob", "rhs_glob"):
            value = strategy.get(key)
            if isinstance(value, str):
                strategy[key] = resolve_path_expression(value, base_dir=config_dir)
        strategies.append(strategy)
    if strategies:
        generation["strategy"] = strategies
    normalized["generation"] = generation

    output = dict(normalized.get("output") or {})
    data_dir = output.get("data_dir")
    if isinstance(data_dir, str):
        output["data_dir"] = resolve_local_path(data_dir, base_dir=config_dir)
    normalized["output"] = output

    test = dict(normalized.get("test") or {})
    for key in ("solutions_path", "rhs_path"):
        value = test.get(key)
        if isinstance(value, str):
            test[key] = resolve_path_expression(value, base_dir=config_dir)
    normalized["test"] = test
    return normalized


def _normalize_comparison_config_paths(raw: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Expand env placeholders and anchor comparison config paths to the config directory."""
    normalized = expand_nested_path_values(raw, path_keys=_COMPARISON_CONFIG_PATH_KEYS)
    config_dir = config_path.resolve().parent

    general = dict(normalized.get("general") or {})
    data = dict(general.get("data") or {})
    for key in ("matrix_path", "rhs_path"):
        value = data.get(key)
        if isinstance(value, str):
            data[key] = resolve_local_path(value, base_dir=config_dir)
    general["data"] = data
    normalized["general"] = general

    preconditioners = []
    for entry in normalized.get("preconditioners", []):
        if not isinstance(entry, dict):
            preconditioners.append(entry)
            continue
        spec = dict(entry)
        for key in ("checkpoint_path", "config_path", "data_config_path"):
            value = spec.get(key)
            if isinstance(value, str):
                spec[key] = resolve_local_path(value, base_dir=config_dir)
        preconditioners.append(spec)
    normalized["preconditioners"] = preconditioners
    return normalized


def _normalize_experiments_config_paths(raw: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Expand env placeholders and anchor master config paths to the config directory."""
    normalized = expand_nested_path_values(raw, path_keys=_EXPERIMENTS_CONFIG_PATH_KEYS)
    config_dir = config_path.resolve().parent
    if "experiment" in normalized:
        raise ValueError(
            "Unsupported '[[experiment]]' table. "
            "Use '[[experiments]]' entries in experiments config."
        )

    for key in ("output_dir", "project_root"):
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = resolve_local_path(value, base_dir=config_dir)

    mlflow = dict(normalized.get("mlflow") or {})
    for key in ("tracking_uri", "artifacts_destination"):
        value = mlflow.get(key)
        if isinstance(value, str):
            mlflow[key] = expand_path_placeholders(value)
    normalized["mlflow"] = mlflow

    for section_name in ("datasets", "models", "comparisons"):
        section = []
        for entry in normalized.get(section_name, []):
            if not isinstance(entry, dict):
                section.append(entry)
                continue
            item = dict(entry)
            value = item.get("path")
            if isinstance(value, str):
                item["path"] = resolve_local_path(value, base_dir=config_dir)
            section.append(item)
        normalized[section_name] = section

    experiments = []
    for entry in normalized.get("experiments", []):
        if not isinstance(entry, dict):
            experiments.append(entry)
            continue
        item = dict(entry)
        checkpoint_path = item.get("checkpoint_path")
        if isinstance(checkpoint_path, str):
            item["checkpoint_path"] = resolve_local_path(checkpoint_path, base_dir=config_dir)
        experiments.append(item)
    normalized["experiments"] = experiments

    runs = []
    for entry in normalized.get("run", []):
        if not isinstance(entry, dict):
            runs.append(entry)
            continue
        item = dict(entry)
        for key in ("model_config", "data_config"):
            value = item.get(key)
            if isinstance(value, str):
                item[key] = str(resolve_local_path(value, base_dir=config_dir))
        runs.append(item)
    normalized["run"] = runs
    return normalized


def _prepare_model_config(path: Path) -> dict[str, Any]:
    """Normalize one model config before DLKit workflow validation."""
    raw = load_raw_toml(path)
    return normalize_model_mlflow(raw, path)


def load_model_config(path: Path) -> ModelWorkflowSettings:
    """Load and validate a model configuration file."""
    normalized = _prepare_model_config(path)
    with tempfile.TemporaryDirectory(prefix="neuralls_model_cfg_") as tmp_dir:
        tmp_path = Path(tmp_dir) / path.name
        with tmp_path.open("wb") as tmp:
            tomli_w.dump(normalized, tmp)
        settings = load_settings(tmp_path)

    if not isinstance(settings, (TrainingWorkflowConfig, OptimizationWorkflowConfig)):
        raise TypeError(
            "Model configs must load as training or optimization workflows, "
            f"got {type(settings).__name__}."
        )
    return settings


def load_data_config(path: Path) -> DataConfigFile:
    """Load and validate data config TOML."""
    raw = load_raw_toml(path)
    raw = _normalize_data_config_paths(raw, path)
    return DataConfigFile(**raw)


def load_comparison_config(path: Path) -> ComparisonConfig:
    """Load and validate comparison TOML."""
    raw = load_raw_toml(path)
    raw = _normalize_comparison_config_paths(raw, path)
    return parse_comparison_config(raw)


def load_experiments_config(path: Path) -> dict[str, Any]:
    """Load master experiments TOML with path placeholders normalized."""
    raw = load_raw_toml(path)
    return _normalize_experiments_config_paths(raw, path)


def load_raw_toml(path: Path) -> dict[str, Any]:
    """Load TOML file as raw dictionary without validation."""
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
    base_settings: ModelWorkflowSettings | None = None,
) -> ModelWorkflowSettings:
    """Build DLKit training or optimization settings with runtime paths injected."""
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

    return patch_model(settings, updates)


def build_inference_settings(
    model_config_path: Path,
    workspace: ExperimentWorkspace,
    path_context: PathContext,
    mlflow_run_name: str | None = None,
    mlflow_experiment_name: str | None = None,
    force_mlflow_enabled: bool = False,
) -> InferenceWorkflowConfig:
    """Build DLKit inference settings with runtime paths injected.

    ``mlflow_run_name``, ``mlflow_experiment_name``, and
    ``force_mlflow_enabled`` are accepted for call-site compatibility but are
    intentionally ignored because DLKit's current inference workflow config
    does not carry an ``MLFLOW`` section.
    """
    del mlflow_run_name, mlflow_experiment_name, force_mlflow_enabled

    _INFERENCE_EXCLUDED = {"TRAINING", "MLFLOW", "OPTUNA"}
    toml_data = _prepare_model_config(model_config_path)
    inference_data = {k: v for k, v in toml_data.items() if k not in _INFERENCE_EXCLUDED}
    session = dict(inference_data.get("SESSION") or {})
    session["workflow"] = "inference"
    inference_data["SESSION"] = session
    settings = InferenceWorkflowConfig.model_validate(inference_data)
    sync_session_root_to_environment(cast(Any, settings))

    updates: dict[str, Any] = {
        "PATHS": {
            "project_root": str(path_context.project_root),
            "processed_dir": str(path_context.processed_root),
            "output_dir": str(path_context.output_root),
        },
    }

    return patch_model(settings, updates)
