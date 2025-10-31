"""Common utilities for graph-cg scripts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, cast

from dynaconf import Dynaconf

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import numpy as np

from dlkit import GeneralSettings
from dlkit.tools.config import load_training_settings
from dlkit.tools.config.core.updater import update_settings
from dlkit.tools.io import load_array

from .paths import (
    ProjectRoots,
    FlowPaths,
    DataPaths,
    TrainingPaths,
    PredictionPaths,
    ComparisonPaths,
    FlowContext,
    parse_flow_keys,
)
from .validation import validate_file_exists

# Default path constants - single source of truth
DEFAULT_PROJECT_ROOT = "/data/projects/graph-cg"
DEFAULT_PROCESSED_DIR = f"{DEFAULT_PROJECT_ROOT}/data/processed"
DEFAULT_RESULTS_DIR = f"{DEFAULT_PROJECT_ROOT}/data/output"
DEFAULT_FIGURES_DIR = f"{DEFAULT_PROJECT_ROOT}/data/figures"


def sanitize_identifier(value: str, default: str = "run") -> str:
    """Convert arbitrary identifier text into a filesystem-friendly slug."""

    cleaned = (value or "").strip()
    if not cleaned:
        return default

    cleaned = cleaned.replace("/", "-").replace("\\", "-")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    cleaned = cleaned.strip("_-.")

    return cleaned or default


def derive_model_identifier(
    settings: GeneralSettings,
    context: FlowContext,
    config_path: str | Path,
) -> str:
    """Derive a stable model identifier for artifact naming.

    Preference order:
        1. ``MODEL.name`` from the config settings
        2. ``SESSION.name`` if present and not the DLKit default
        3. Context ``run_id``
        4. Config filename stem
    """

    config_path = Path(config_path)

    candidates: list[str | None] = []
    if getattr(settings, "MODEL", None) and getattr(settings.MODEL, "name", None):
        candidates.append(str(settings.MODEL.name))

    if getattr(settings, "SESSION", None):
        candidates.append(getattr(settings.SESSION, "name", None))

    candidates.append(getattr(context, "run_id", None))
    candidates.append(config_path.stem)

    for candidate in candidates:
        if isinstance(candidate, str):
            normalized = candidate.strip()
            if normalized and not normalized.lower().startswith("dlkit-session"):
                return sanitize_identifier(normalized, default="model")

    return sanitize_identifier(config_path.stem, default="model")


def load_flow_keys(
    config_path: str | Path,
    data_config_path: str | Path | None = None,
) -> tuple[str, str]:
    """Load ``(flow_id, dataset_id)`` from a config or paired data config file."""

    if data_config_path is not None:
        data_config_path = validate_file_exists(data_config_path, "Data config file")
        with open(data_config_path, "rb") as fh:
            raw = tomllib.load(fh)
        return parse_flow_keys(raw, config_path=data_config_path)

    config_path = validate_file_exists(config_path, "Config file")
    with open(config_path, "rb") as fh:
        raw = tomllib.load(fh)
    return parse_flow_keys(raw, config_path=config_path)


def _load_raw_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "rb") as fh:
        return tomllib.load(fh)


def _build_flow_context(
    config_path: Path,
    raw_config: Mapping[str, Any],
    data_config: Mapping[str, Any] | None = None,
    data_config_path: Path | None = None,
) -> FlowContext:
    """Construct flow context using model config plus optional data config."""

    flow_source: Mapping[str, Any]
    source_path: Path
    if data_config is not None:
        flow_source = data_config
        source_path = data_config_path if data_config_path is not None else config_path
    else:
        flow_source = raw_config
        source_path = config_path

    try:
        flow_id, dataset_id = parse_flow_keys(flow_source, config_path=source_path)
    except ValueError:
        if data_config is not None:
            raise ValueError(
                "Data config is missing required [flow] section"
            )

        stem = config_path.stem
        flow_id = str(raw_config.get("flow_id", stem))
        dataset_id = str(raw_config.get("dataset", stem))

    paths_section = dict(raw_config.get("PATHS") or {})
    if data_config is not None:
        output_section = data_config.get("output") or {}
        if isinstance(output_section, Mapping):
            processed_dir = output_section.get("processed_dir")
            if processed_dir:
                paths_section.setdefault("processed_dir", processed_dir)
            results_dir = output_section.get("results_dir")
            if results_dir:
                paths_section.setdefault("results_dir", results_dir)
            figures_dir = output_section.get("figures_dir")
            if figures_dir:
                paths_section.setdefault("figures_dir", figures_dir)

    project_root = paths_section.get("project_root")
    processed_root = paths_section.get("processed_dir")
    output_root = paths_section.get("results_dir")
    figures_root = paths_section.get("figures_dir")

    roots = ProjectRoots.from_overrides(
        project_root=project_root,
        processed_root=processed_root,
        output_root=output_root,
        figures_root=figures_root,
    )

    session_section = raw_config.get("SESSION") or {}
    run_id = session_section.get("name")
    if not isinstance(run_id, str) or not run_id:
        # If SESSION.name not set, derive from config file name (e.g., "ffnn" from "ffnn.toml")
        if config_path:
            run_id = Path(config_path).stem
        else:
            run_id = dataset_id

    flow_paths = FlowPaths(flow_id=flow_id, roots=roots)
    data_paths = DataPaths(flow=flow_paths, dataset_id=dataset_id)
    training_paths = TrainingPaths(data=data_paths, run_id=run_id)
    prediction_paths = PredictionPaths(training=training_paths)
    comparison_paths = ComparisonPaths(flow=flow_paths, dataset_id=dataset_id)

    # Extract test paths from data_config if available
    test_rhs = None
    test_matrix = None
    test_solutions_path = None
    if data_config is not None:
        test_section = data_config.get("test") or {}
        if isinstance(test_section, Mapping):
            if "rhs" in test_section:
                test_rhs = Path(test_section["rhs"])
            if "matrix" in test_section:
                test_matrix = Path(test_section["matrix"])
            if "solutions_path" in test_section:
                test_solutions_path = str(test_section["solutions_path"])

    return FlowContext(
        flow=flow_paths,
        data=data_paths,
        training=training_paths,
        prediction=prediction_paths,
        comparison=comparison_paths,
        run_id=run_id,
        test_rhs=test_rhs,
        test_matrix=test_matrix,
        test_solutions_path=test_solutions_path,
    )


def _apply_data_config_overrides(
    settings: Dynaconf, data_config: Mapping[str, Any]
) -> None:
    """Inject relevant data-config metadata into lazy settings."""

    generation_section = data_config.get("generation") or {}
    if not isinstance(generation_section, Mapping):
        generation_section = {}

    extras_cfg = settings.get("EXTRAS") or {}
    if not isinstance(extras_cfg, dict):
        extras_cfg = {}

    data_gen_cfg = extras_cfg.get("data_generation") or {}
    if not isinstance(data_gen_cfg, dict):
        data_gen_cfg = {}

    for key in (
        "normalize",
        "num_samples",
        "mix",
        "krylov_iters",
        "seed",
        "shuffle",
    ):
        if key in generation_section:
            data_gen_cfg[key] = generation_section[key]

    if data_gen_cfg:
        extras_cfg["data_generation"] = data_gen_cfg
        settings.set("EXTRAS", extras_cfg)


def _remove_flow_metadata(settings: Dynaconf) -> None:
    """Remove raw FLOW metadata keys to avoid Pydantic validation errors.

    Args:
        settings: Dynaconf settings object to modify
    """
    for key in ("FLOW", "flow"):
        try:
            settings.unset(key)
        except Exception:
            pass


def _inject_path_settings(settings: Dynaconf, context: FlowContext) -> None:
    """Inject derived paths from context into settings.

    Args:
        settings: Dynaconf settings object to modify
        context: Flow context containing derived paths
    """
    settings.set("PATHS.processed_dir", str(context.flow.processed_root))
    settings.set("PATHS.results_dir", str(context.flow.output_root))
    settings.set("PATHS.figures_dir", str(context.flow.figures_root))
    settings.set("PATHS.matrix_path", str(context.data.matrix_file))
    settings.set("PATHS.rhs_path", str(context.data.mother_rhs_file))


def _update_dataset_features(
    features: Any, context: FlowContext
) -> list[dict[str, Any]]:
    """Update feature configurations with context paths.

    Args:
        features: Existing features configuration
        context: Flow context with feature paths

    Returns:
        Updated features list
    """
    if not isinstance(features, (list, tuple)) or not features:
        return [{"name": "rhs", "path": str(context.data.features_file)}]

    updated_features = []
    for idx, feat in enumerate(features):
        if isinstance(feat, dict):
            new_feat = dict(feat)
            if idx == 0:
                new_feat["path"] = str(context.data.features_file)
            updated_features.append(new_feat)
        else:
            updated_features.append(feat)

    return updated_features


def _update_dataset_targets(
    targets: Any, context: FlowContext
) -> list[dict[str, Any]]:
    """Update target configurations with context paths.

    Args:
        targets: Existing targets configuration
        context: Flow context with target paths

    Returns:
        Updated targets list
    """
    if not isinstance(targets, (list, tuple)) or not targets:
        return [{"name": "sol", "path": str(context.data.targets_file)}]

    updated_targets = []
    for idx, targ in enumerate(targets):
        if isinstance(targ, dict):
            new_targ = dict(targ)
            if idx == 0:
                new_targ["path"] = str(context.data.targets_file)
            updated_targets.append(new_targ)
        else:
            updated_targets.append(targ)

    return updated_targets


def _is_graph_dataset(dataset_cfg: dict[str, Any]) -> bool:
    """Check if dataset is a graph-based dataset.

    Args:
        dataset_cfg: Dataset configuration dictionary

    Returns:
        True if dataset uses GraphDataset (requires x, edge_index, y parameters)
    """
    return dataset_cfg.get("name") == "GraphDataset"


def _configure_graph_dataset(dataset_cfg: dict[str, Any], context: FlowContext) -> dict[str, Any]:
    """Configure GraphDataset with runtime-injected paths.

    GraphDataset uses PyTorch Geometric and expects:
    - x: Node features file (RHS samples)
    - edge_index: Adjacency matrix file (system matrix A)
    - y: Targets file (solution samples)

    Args:
        dataset_cfg: Existing dataset configuration
        context: Flow context with dataset paths

    Returns:
        Updated dataset configuration with injected paths
    """
    dataset_cfg["root_dir"] = str(context.data.base_dir)
    dataset_cfg["x"] = str(context.data.features_file)  # rhs-samples.npy
    dataset_cfg["edge_index"] = str(context.data.matrix_file)  # matrix.npy
    dataset_cfg["y"] = str(context.data.targets_file)  # sol-samples.npy
    return dataset_cfg


def _configure_dataset_section(settings: Dynaconf, context: FlowContext) -> None:
    """Configure DATASET section with context-derived paths.

    Uses strategy pattern to handle different dataset types:
    - GraphDataset: Direct x/edge_index/y parameters
    - Standard datasets: features/targets arrays

    Args:
        settings: Dynaconf settings object to modify
        context: Flow context with dataset paths
    """
    dataset_cfg = settings.get("DATASET") or {}
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}

    dataset_cfg["name"] = dataset_cfg.get("name") or "FlexibleDataset"
    dataset_cfg["root_dir"] = str(context.data.base_dir)

    # Strategy: configure based on dataset type
    if _is_graph_dataset(dataset_cfg):
        dataset_cfg = _configure_graph_dataset(dataset_cfg, context)
    else:
        # Standard dataset: features/targets arrays
        features = dataset_cfg.get("features")
        dataset_cfg["features"] = _update_dataset_features(features, context)

        targets = dataset_cfg.get("targets")
        dataset_cfg["targets"] = _update_dataset_targets(targets, context)

    settings.set("DATASET", dataset_cfg)


def _configure_training_paths(settings: Dynaconf, context: FlowContext) -> None:
    """Configure training paths in settings.

    Args:
        settings: Dynaconf settings object to modify
        context: Flow context with training paths
    """
    settings.set("TRAINING.trainer.default_root_dir", str(context.training.base_dir))


def _configure_session(settings: Dynaconf, context: FlowContext) -> None:
    """Configure SESSION section with training paths.

    Note: SESSION.name is intentionally not set to allow dlkit proper
    MLflow experiment name resolution.

    Args:
        settings: Dynaconf settings object to modify
        context: Flow context with session info
    """
    session_cfg = settings.get("SESSION") or {}
    if not isinstance(session_cfg, dict):
        session_cfg = {}

    session_cfg.setdefault("root_dir", str(context.training.base_dir))
    settings.set("SESSION", session_cfg)


def _configure_mlflow_experiment(settings: Dynaconf, context: FlowContext) -> None:
    """Configure MLflow experiment name from dataset context.

    Args:
        settings: Dynaconf settings object to modify
        context: Flow context with dataset ID
    """
    mlflow_cfg = settings.get("MLFLOW") or {}
    if not isinstance(mlflow_cfg, dict):
        return

    client_cfg = mlflow_cfg.get("client") or {}
    if not isinstance(client_cfg, dict):
        client_cfg = {}

    client_cfg["experiment_name"] = context.data.dataset_id
    mlflow_cfg["client"] = client_cfg
    settings.set("MLFLOW", mlflow_cfg)


def _apply_flow_context_to_lazy(settings: Dynaconf, context: FlowContext) -> None:
    """Inject derived paths into Dynaconf LazySettings.

    Orchestrates configuration of all settings sections with context-derived paths.

    Args:
        settings: Dynaconf settings object to modify
        context: Flow context containing all derived paths
    """
    _remove_flow_metadata(settings)
    _inject_path_settings(settings, context)
    _configure_dataset_section(settings, context)
    _configure_training_paths(settings, context)
    _configure_session(settings, context)
    _configure_mlflow_experiment(settings, context)


def _apply_training_overrides(
    settings: GeneralSettings,
    config_path: Path,
    raw_config: Mapping[str, Any],
) -> GeneralSettings:
    """Reapply training values ensuring callback configuration is preserved."""

    training_section: Mapping[str, Any] | None = None

    try:
        loaded_settings = load_training_settings(str(config_path))
        training_settings = getattr(loaded_settings, "TRAINING", None)
        if training_settings is not None:
            dumped = training_settings.model_dump(mode="python")
            callbacks = (
                tuple(dumped.get("trainer", {}).get("callbacks", ()))
                if isinstance(dumped, Mapping)
                else ()
            )
            if callbacks:
                training_section = dumped
    except Exception:
        training_section = None

    if training_section is None:
        raw_training = raw_config.get("TRAINING")
        if isinstance(raw_training, Mapping) and raw_training:
            training_section = raw_training

    if training_section is None:
        return settings

    current_training = settings.TRAINING
    current_trainer = getattr(current_training, "trainer", None) if current_training else None
    preserved_root = getattr(current_trainer, "default_root_dir", None)

    updated = update_settings(settings, {"TRAINING": dict(training_section)})

    if preserved_root:
        updated = update_settings(
            updated,
            {"TRAINING": {"trainer": {"default_root_dir": preserved_root}}},
        )

    return cast(GeneralSettings, updated)


def get_data_generation_params(
    config_path: str | Path,
    data_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Get data generation parameters from config using direct TOML reading.

    This is the single source of truth for data generation settings including
    normalization. Both data generation and solver scripts should use this function.

    Args:
        config_path: Path to config file

    Returns:
        Dictionary with 'num_samples' and 'normalize' keys (or empty dict if not found)
    """
    target_path = Path(data_config_path or config_path)
    if not target_path.exists():
        return {}

    try:
        with open(target_path, "rb") as f:
            config = tomllib.load(f)

        if data_config_path is not None:
            data_gen_cfg = config.get("generation", {})
        else:
            extras = config.get("EXTRAS", {})
            data_gen_cfg = extras.get("data_generation", {})

        if not isinstance(data_gen_cfg, dict):
            return {}

        result = {}

        raw_value = data_gen_cfg.get("num_samples") or data_gen_cfg.get("total_samples")
        if raw_value is not None:
            value = int(raw_value)
            if value < 1:
                raise ValueError(f"Configured num_samples must be positive, got {value}")
            result["num_samples"] = value

        normalize_value = data_gen_cfg.get("normalize")
        if normalize_value is not None:
            result["normalize"] = bool(normalize_value)

        return result

    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Invalid parameters in data generation config"
        ) from exc
    except Exception:
        return {}


def _load_validated_configs(
    config_path: str | Path,
    data_config_path: str | Path | None,
) -> tuple[Path, dict[str, Any], dict[str, Any] | None, Path | None]:
    """Load and validate configuration files.

    Args:
        config_path: Path to main config file
        data_config_path: Optional path to data config file

    Returns:
        Tuple of (validated_config_path, raw_config, data_config, validated_data_config_path)
    """
    validated_path = validate_file_exists(config_path, "Config file")
    validated_path = Path(validated_path)

    data_config = None
    validated_data_path = None
    if data_config_path is not None:
        validated_data_path = Path(validate_file_exists(data_config_path, "Data config file"))
        data_config = _load_raw_config(validated_data_path)

    raw_config = _load_raw_config(validated_path)
    return validated_path, raw_config, data_config, validated_data_path


def _create_lazy_settings_with_overrides(
    config_path: Path,
    data_config: dict[str, Any] | None,
    context: FlowContext,
) -> Dynaconf:
    """Create Dynaconf lazy settings with all overrides applied.

    Args:
        config_path: Path to config file
        data_config: Optional data config overrides
        context: Flow context with derived paths

    Returns:
        Configured Dynaconf settings
    """
    lazy_settings = Dynaconf(settings_files=[str(config_path)], load_dotenv=False)

    if data_config is not None:
        _apply_data_config_overrides(lazy_settings, data_config)

    _apply_flow_context_to_lazy(lazy_settings, context)
    return lazy_settings


def _ensure_training_directories(context: FlowContext) -> None:
    """Ensure training directories exist.

    Args:
        context: Flow context with training paths
    """
    context.training.base_dir.mkdir(parents=True, exist_ok=True)
    context.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def _extract_session_name(settings: GeneralSettings, fallback: str) -> str:
    """Extract session name from settings or use fallback.

    Args:
        settings: General settings with optional SESSION.name
        fallback: Fallback session name

    Returns:
        Extracted or fallback session name
    """
    if not settings.SESSION:
        return fallback

    name = getattr(settings.SESSION, "name", None)
    return name if name else fallback


def _build_settings_from_context(
    config_path: Path,
    raw_config: dict[str, Any],
    data_config: dict[str, Any] | None,
    context: FlowContext,
) -> GeneralSettings:
    """Build GeneralSettings from context and configs.

    Args:
        config_path: Path to config file
        raw_config: Raw config dictionary
        data_config: Optional data config overrides
        context: Flow context

    Returns:
        Configured GeneralSettings
    """
    _ensure_training_directories(context)
    lazy_settings = _create_lazy_settings_with_overrides(config_path, data_config, context)
    settings = GeneralSettings.dynaconf_to_settings(lazy_settings)
    return _apply_training_overrides(settings, config_path, raw_config)


def load_config_with_context(
    config_path: str | Path,
    data_config_path: str | Path | None = None,
) -> tuple[GeneralSettings, FlowContext]:
    """Load configuration and derive centralized path context.

    Orchestrates config loading, context building, and settings creation.
    When data_config_path is provided, derives dataset information from
    data config instead of model config for data-agnostic model files.

    Args:
        config_path: Path to main config file
        data_config_path: Optional path to data config file

    Returns:
        Tuple of (settings, context)
    """
    # Load and validate configs
    config_path, raw_config, data_config, validated_data_path = _load_validated_configs(
        config_path, data_config_path
    )

    # Build initial context
    context = _build_flow_context(config_path, raw_config, data_config, validated_data_path)

    # Build settings from context
    settings = _build_settings_from_context(
        config_path, raw_config, data_config, context
    )

    # Reconcile session name if needed
    session_name = _extract_session_name(settings, context.run_id)
    if session_name != context.run_id:
        context = context.with_run_id(session_name)
        settings = _build_settings_from_context(
            config_path, raw_config, data_config, context
        )

    return settings, context


def load_config(
    config_path: str | Path,
    data_config_path: str | Path | None = None,
) -> GeneralSettings:
    """Load configuration from file using centralized path context."""

    settings, _ = load_config_with_context(config_path, data_config_path)
    return settings


def get_paths_from_config(settings: GeneralSettings, context: FlowContext) -> dict[str, Any]:
    """Build a dictionary of resolved paths from settings and context."""

    run_id = (
        settings.SESSION.name
        if settings.SESSION and getattr(settings.SESSION, "name", None)
        else context.run_id
    )
    context = context.with_run_id(run_id)

    matrix_path = None
    rhs_path = None
    results_dir = None
    figures_dir = None

    if settings.PATHS:
        matrix_path = getattr(settings.PATHS, "matrix_path", None)
        rhs_path = getattr(settings.PATHS, "rhs_path", None)
        results_dir = getattr(settings.PATHS, "results_dir", None)
        figures_dir = getattr(settings.PATHS, "figures_dir", None)

    dataset_features = None
    dataset_targets = None
    if settings.DATASET:
        if settings.DATASET.features:
            dataset_features = settings.DATASET.features[0].path
        if settings.DATASET.targets:
            dataset_targets = settings.DATASET.targets[0].path

    checkpoint_path = None
    if settings.MODEL and getattr(settings.MODEL, "checkpoint", None):
        checkpoint_path = settings.MODEL.checkpoint
    elif settings.PATHS and getattr(settings.PATHS, "checkpoint_path", None):
        checkpoint_path = settings.PATHS.checkpoint_path

    training_dir = None
    checkpoint_dir = None
    if settings.TRAINING and settings.TRAINING.trainer:
        trainer = settings.TRAINING.trainer
        if getattr(trainer, "default_root_dir", None):
            training_dir = Path(trainer.default_root_dir)
        callbacks = getattr(trainer, "callbacks", None) or []
        for cb in callbacks:
            if getattr(cb, "dirpath", None):
                checkpoint_dir = Path(cb.dirpath)
                break

    return {
        "flow_id": context.flow.flow_id,
        "dataset_id": context.data.dataset_id,
        "run_id": context.run_id,
        "processed_root": context.flow.processed_root,
        "dataset_dir": context.data.base_dir,
        "matrix_path": Path(matrix_path) if matrix_path else context.data.matrix_file,
        "rhs_path": Path(rhs_path) if rhs_path else context.data.mother_rhs_file,
        "features_path": Path(dataset_features) if dataset_features else context.data.features_file,
        "targets_path": Path(dataset_targets) if dataset_targets else context.data.targets_file,
        "training_dir": training_dir or context.training.base_dir,
        "checkpoint_dir": checkpoint_dir or context.training.checkpoint_dir,
        "prediction_dir": context.prediction.base_dir,
        "results_dir": Path(results_dir) if results_dir else context.comparison.base_dir,
        "figures_dir": Path(figures_dir) if figures_dir else context.comparison.figures_dir,
        "comparison_reports_dir": context.comparison.report_dir,
        "checkpoint_path": Path(checkpoint_path) if checkpoint_path else None,
    }


def get_solver_params(settings: GeneralSettings) -> dict[str, Any]:
    """Extract solver parameters from config.

    NOTE: normalize_system comes from [EXTRAS.data_generation].normalize - single source of truth!
    Both data generation and solver must use the same core normalize_system() function from math_utils.

    Args:
        settings: Loaded GeneralSettings

    Returns:
        Dictionary with solver parameters
    """
    extras = settings.EXTRAS
    solver_cfg = {}
    data_gen_cfg = {}

    if extras is not None:
        extras_dict = extras.model_dump()
        raw_solver = extras_dict.get("solver", {})
        if isinstance(raw_solver, dict):
            solver_cfg = raw_solver
        raw_data_gen = extras_dict.get("data_generation", {})
        if isinstance(raw_data_gen, dict):
            data_gen_cfg = raw_data_gen

    # Extract solver-specific params with defaults
    tolerance = solver_cfg.get("tolerance", 1e-8)
    max_iterations = solver_cfg.get("max_iterations", 30)
    stopping_criterion = solver_cfg.get("stopping_criterion", "tolerance")

    # Get normalize from data_generation section - SINGLE SOURCE OF TRUTH
    normalize_system = data_gen_cfg.get("normalize", True)

    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError):
        tolerance = 1e-8

    try:
        max_iterations = int(max_iterations)
    except (TypeError, ValueError):
        max_iterations = 30

    if isinstance(normalize_system, str):
        normalize_system = normalize_system.lower() in {"1", "true", "yes", "on"}

    return {
        "tolerance": tolerance,
        "max_iterations": max_iterations,
        "normalize_system": bool(normalize_system),
        "stopping_criterion": str(stopping_criterion),
    }


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, creating it if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_matrix_file(matrix_path: str | Path) -> np.ndarray:
    """Load matrix from file (supports .npy and text formats).

    Args:
        matrix_path: Path to matrix file

    Returns:
        Matrix array as float64
    """
    validated_path = validate_file_exists(matrix_path, "Matrix file")

    if Path(validated_path).suffix == ".npy":
        A = load_array(validated_path).numpy()
        return np.asarray(A, dtype=np.float64)

    return np.loadtxt(validated_path, dtype=np.float64)


def _load_rhs_file(rhs_path: str | Path) -> np.ndarray:
    """Load RHS vector from file (supports .npy and text formats).

    Args:
        rhs_path: Path to RHS file

    Returns:
        RHS vector as float64, reshaped to 1D
    """
    validated_path = validate_file_exists(rhs_path, "RHS file")

    if Path(validated_path).suffix == ".npy":
        b = load_array(validated_path).numpy()
        b = np.asarray(b, dtype=np.float64)
    else:
        b = np.loadtxt(validated_path, dtype=np.float64)

    # Ensure 1D shape
    if b.ndim > 1:
        b = b.reshape(-1)

    return b


def _ensure_rhs_matches_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ensure RHS dimension matches matrix, creating fallback if needed.

    Args:
        A: System matrix
        b: RHS vector

    Returns:
        RHS vector with correct dimension
    """
    if b.shape[0] == A.shape[0]:
        return b

    print(
        f"RHS length {b.shape[0]} doesn't match matrix size {A.shape[0]}. "
        "Setting RHS to ones scaled by Frobenius norm."
    )
    return np.ones(A.shape[0], dtype=np.float64) * np.linalg.norm(A, "fro")


def load_system_data(
    matrix_path: str | Path, rhs_path: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    """Load matrix and RHS data.

    Orchestrates loading and validation of system components.

    Args:
        matrix_path: Path to matrix file
        rhs_path: Path to RHS file

    Returns:
        Tuple of (matrix, rhs) arrays
    """
    A = _load_matrix_file(matrix_path)
    b = _load_rhs_file(rhs_path)
    b = _ensure_rhs_matches_matrix(A, b)
    return A, b


def save_training_data(
    features: np.ndarray,
    targets: np.ndarray,
    features_path: str | Path,
    targets_path: str | Path,
) -> None:
    """Save training data arrays.

    Args:
        features: Feature array
        targets: Target array
        features_path: Output path for features
        targets_path: Output path for targets
    """
    features_path = Path(features_path)
    targets_path = Path(targets_path)

    ensure_dir(features_path.parent)
    ensure_dir(targets_path.parent)

    # Ensure float64 dtype for all saved data
    np.save(features_path, features.astype(np.float64, copy=False))
    np.save(targets_path, targets.astype(np.float64, copy=False))


# Normalization metadata functions removed - all normalization now happens at data generation time
# The 'normalized' boolean in metadata.json is the single source of truth


def get_latest_checkpoint(
    checkpoint_dir: str | Path, pattern: str = "*.ckpt"
) -> Path | None:
    """Find the most recent checkpoint file.

    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: File pattern to match

    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob(pattern))
    if not checkpoints:
        return None

    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def load_case_data(data_dir: str | Path) -> dict[str, np.ndarray | dict[str, Any]]:
    """Load all data from a case directory.

    Args:
        data_dir: Path to case directory (e.g., collect-504-norm)

    Returns:
        Dictionary with keys: matrix, rhs_samples, sol_samples, rhs_mother, metadata, normalization

    Example:
        >>> data = load_case_data("/data/projects/graph-cg/data/processed/collect-504-norm")
        >>> A = data["matrix"]
        >>> R = data["rhs_samples"]
        >>> X = data["sol_samples"]
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    result = {}

    # Load arrays
    for name in ["matrix", "rhs-samples", "sol-samples", "rhs-mother"]:
        path = data_dir / f"{name}.npy"
        if path.exists():
            result[name.replace("-", "_")] = np.load(path).astype(np.float64, copy=False)

    # Load metadata
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            result["metadata"] = json.load(f)

    # Load normalization metadata
    norm_path = data_dir / "normalization.json"
    if norm_path.exists():
        with norm_path.open("r", encoding="utf-8") as f:
            result["normalization"] = json.load(f)

    return result


def list_available_cases(processed_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """List all available data cases in processed directory.

    Args:
        processed_dir: Path to processed data directory (defaults to DEFAULT_PROCESSED_DIR)

    Returns:
        List of dictionaries with case information (name, type, dimension, etc.)

    Example:
        >>> cases = list_available_cases()
        >>> for case in cases:
        ...     print(f"{case['name']}: {case['dimension']}-dim, {case['source_type']}")
    """
    if processed_dir is None:
        processed_dir = Path(DEFAULT_PROCESSED_DIR)
    else:
        processed_dir = Path(processed_dir)

    if not processed_dir.exists():
        return []

    cases = []
    for case_dir in processed_dir.iterdir():
        if not case_dir.is_dir():
            continue

        # Check if it has metadata
        metadata_path = case_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

            case_info = {
                "name": case_dir.name,
                "path": str(case_dir),
                "source_type": metadata.get("source_type", "unknown"),
                "dimension": metadata.get("dimension"),
                "num_samples": metadata.get("num_samples"),
                "normalized": metadata.get("normalized", False),
            }

            cases.append(case_info)
        except (json.JSONDecodeError, OSError):
            continue

    return sorted(cases, key=lambda x: (x["source_type"], x["dimension"] or 0))


def parse_data_dir_name(dir_name: str) -> dict[str, Any]:
    """Parse data directory name to extract parameters.

    Args:
        dir_name: Directory name (e.g., "collect-504-norm", "generate-90-krylov50-norm")

    Returns:
        Dictionary with parsed parameters

    Examples:
        >>> parse_data_dir_name("collect-504-norm")
        {'source': 'collect', 'dimension': 504, 'normalized': True, 'krylov_percent': None}
        >>> parse_data_dir_name("generate-90-krylov50-norm")
        {'source': 'generate', 'dimension': 90, 'normalized': True, 'krylov_percent': 50}
    """
    parts = dir_name.split("-")

    if len(parts) < 3:
        return {"source": "unknown", "dimension": None, "normalized": False, "krylov_percent": None}

    result = {
        "source": parts[0],
        "dimension": None,
        "normalized": parts[-1] == "norm",
        "krylov_percent": None,
    }

    # Extract dimension (second part)
    try:
        result["dimension"] = int(parts[1])
    except (ValueError, IndexError):
        pass

    # Check for krylov tag (if present, it's the second-to-last or third part)
    for part in parts[2:-1]:
        if part.startswith("krylov"):
            try:
                result["krylov_percent"] = int(part.replace("krylov", ""))
            except ValueError:
                pass

    return result
