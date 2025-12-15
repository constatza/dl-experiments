"""Configuration loading and injection for training workflows.

This module unifies configuration management by:
- Loading model, data, and solver configs from TOML files
- Building FlowContext from extracted metadata
- Injecting context paths and solver parameters into GeneralSettings
- Creating training directories

All functions follow functional programming principles:
- Pure functions for data extraction and transformations
- I/O isolated in specific actions
- Clear separation between configuration and execution
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from dlkit import GeneralSettings
from dlkit.tools.config import load_training_settings
from dlkit.tools.config.core.updater import update_settings
from dlkit.tools.config.core.base_settings import BasicSettings

from ..constants import (
    DEFAULT_PROJECT_ROOT,
    EXP_MODEL_CONFIG_NAME,
    EXP_DATA_CONFIG_NAME,
    EXP_SOLVER_CONFIG_NAME,
)
from ..paths.core import (
    FlowContext,
    ProjectRoots,
    FlowPaths,
    DataPaths,
    TrainingPaths,
    PredictionPaths,
    ComparisonPaths,
    parse_flow_keys,
)
from ..validation import validate_file_exists


def _load_toml(path: Path) -> dict[str, Any]:
    """Load TOML file into dictionary.

    Pure function - reads and parses TOML without side effects on app state.

    Args:
        path: Path to TOML file.

    Returns:
        Parsed TOML content as dictionary.

    Raises:
        FileNotFoundError: If file doesn't exist.
        tomllib.TOMLDecodeError: If TOML is malformed.
    """
    with open(path, "rb") as f:
        return tomllib.load(f)


def _extract_project_roots(
    model_config: dict[str, Any],
    data_config: dict[str, Any] | None,
) -> ProjectRoots:
    """Extract project root paths from configs.

    Pure function - builds ProjectRoots from config data.

    Args:
        model_config: Model configuration dictionary.
        data_config: Optional data configuration dictionary.

    Returns:
        ProjectRoots with resolved paths or defaults.
    """
    paths_section = dict(model_config.get("PATHS") or {})

    # Merge output paths from data config if present
    if data_config is not None:
        output_section = data_config.get("output") or {}
        if isinstance(output_section, dict):
            if "processed_dir" in output_section:
                paths_section.setdefault("processed_dir", output_section["processed_dir"])
            if "results_dir" in output_section:
                paths_section.setdefault("results_dir", output_section["results_dir"])
            if "figures_dir" in output_section:
                paths_section.setdefault("figures_dir", output_section["figures_dir"])

    return ProjectRoots.from_overrides(
        project_root=paths_section.get("project_root"),
        processed_root=paths_section.get("processed_dir"),
        output_root=paths_section.get("results_dir"),
        figures_root=paths_section.get("figures_dir"),
    )


def _extract_flow_keys(
    model_config: dict[str, Any],
    data_config: dict[str, Any] | None,
    config_path: Path,
    data_config_path: Path | None,
) -> tuple[str, str]:
    """Extract flow_id and dataset_id from configs.

    Pure function - extracts identifiers using parse_flow_keys with fallbacks.

    Args:
        model_config: Model configuration dictionary.
        data_config: Optional data configuration dictionary.
        config_path: Path to model config file.
        data_config_path: Optional path to data config file.

    Returns:
        Tuple of (flow_id, dataset_id).
    """
    # Prefer data config for flow keys if available
    source_config = data_config or model_config
    source_path = data_config_path or config_path

    try:
        return parse_flow_keys(source_config, config_path=source_path)
    except ValueError:
        # Fallback to legacy keys
        stem = config_path.stem
        flow_id = str(model_config.get("flow_id", stem))
        dataset_id = str(model_config.get("dataset", stem))
        return flow_id, dataset_id


def _build_context(
    model_config: dict[str, Any],
    data_config: dict[str, Any] | None,
    config_path: Path,
    data_config_path: Path | None,
) -> FlowContext:
    """Build complete FlowContext from configurations.

    Pure function - orchestrates context construction from extracted data.

    Args:
        model_config: Model configuration dictionary.
        data_config: Optional data configuration dictionary.
        config_path: Path to model config file.
        data_config_path: Optional path to data config file.

    Returns:
        FlowContext with all path structures.
    """
    roots = _extract_project_roots(model_config, data_config)
    flow_id, dataset_id = _extract_flow_keys(
        model_config, data_config, config_path, data_config_path
    )

    # Extract run_id from SESSION.name or config filename
    session_section = model_config.get("SESSION") or {}
    run_id = session_section.get("name", config_path.stem)

    # Build path structures
    flow = FlowPaths(flow_id=flow_id, roots=roots)
    data = DataPaths(flow=flow, dataset_id=dataset_id)
    training = TrainingPaths(data=data, run_id=run_id)
    prediction = PredictionPaths(training=training)
    comparison = ComparisonPaths(flow=flow, dataset_id=dataset_id)

    # Extract test paths if present
    test_rhs = None
    test_matrix = None
    test_solutions_path = None
    if data_config is not None:
        test_section = data_config.get("test") or {}
        if isinstance(test_section, dict):
            if "rhs" in test_section:
                test_rhs = Path(test_section["rhs"])
            if "matrix" in test_section:
                test_matrix = Path(test_section["matrix"])
            if "solutions_path" in test_section:
                test_solutions_path = str(test_section["solutions_path"])

    return FlowContext(
        flow=flow,
        data=data,
        training=training,
        prediction=prediction,
        comparison=comparison,
        run_id=run_id,
        test_rhs=test_rhs,
        test_matrix=test_matrix,
        test_solutions_path=test_solutions_path,
    )


def load_data_context(
    data_config_path: str | Path | None,
) -> tuple[FlowContext | None, dict[str, Any] | None, Path | None]:
    """Load data config only and build a minimal FlowContext.

    Args:
        data_config_path: Path to data config TOML. If None, returns (None, None, None).

    Returns:
        Tuple of (FlowContext | None, data_config | None, resolved_path | None).
    """
    if data_config_path is None:
        return None, None, None

    data_config_path_obj = validate_file_exists(data_config_path, "Data config")
    data_config = _load_toml(data_config_path_obj)

    # Reuse project root extraction but with empty model config
    roots = _extract_project_roots({}, data_config)
    flow_id, dataset_id = parse_flow_keys(
        data_config, config_path=data_config_path_obj
    )
    run_id = data_config_path_obj.stem

    flow = FlowPaths(flow_id=flow_id, roots=roots)
    data = DataPaths(flow=flow, dataset_id=dataset_id)
    training = TrainingPaths(data=data, run_id=run_id)
    prediction = PredictionPaths(training=training)
    comparison = ComparisonPaths(flow=flow, dataset_id=dataset_id)

    test_rhs = None
    test_matrix = None
    test_solutions_path = None
    test_section = data_config.get("test") or {}
    if isinstance(test_section, dict):
        if "rhs" in test_section:
            test_rhs = Path(test_section["rhs"])
        if "matrix" in test_section:
            test_matrix = Path(test_section["matrix"])
        if "solutions_path" in test_section:
            test_solutions_path = str(test_section["solutions_path"])

    context = FlowContext(
        flow=flow,
        data=data,
        training=training,
        prediction=prediction,
        comparison=comparison,
        run_id=run_id,
        test_rhs=test_rhs,
        test_matrix=test_matrix,
        test_solutions_path=test_solutions_path,
    )
    return context, data_config, data_config_path_obj


def _inject_context_paths(
    settings: BasicSettings,
    context: FlowContext,
) -> BasicSettings:
    """Inject flow context paths into settings.

    Pure function - uses update_settings to inject path information.

    Args:
        settings: Settings object to update.
        context: FlowContext with computed paths.

    Returns:
        Updated settings with injected paths.
    """
    return update_settings(
        settings,
        {
            "PATHS": {
                "project_root": str(context.flow.roots.project_root),
                "processed_dir": str(context.flow.roots.processed_root),
                "results_dir": str(context.flow.roots.output_root),
                "figures_dir": str(context.flow.roots.figures_root),
            },
            "TRAINING": {
                "trainer": {
                    "default_root_dir": str(context.training.base_dir),
                }
            },
        },
    )


def _inject_solver_params(
    settings: BasicSettings,
    solver_config: dict[str, Any],
) -> BasicSettings:
    """Inject solver parameters into settings.EXTRAS."""
    return update_settings(
        settings,
        {
            "EXTRAS": {
                # Store full solver config under solver_config to decouple from model configs
                "solver_config": solver_config,
            }
        },
    )


def _get_default_solver_config_path() -> Path:
    """Get path to default solver configuration.

    Pure function - returns default solver config location.

    Returns:
        Path to configs/experiments/default/solver.toml.
    """
    return DEFAULT_PROJECT_ROOT / "configs" / "experiments" / "default" / "solver.toml"


def _ensure_training_dirs(context: FlowContext) -> None:
    """Create training directories if they don't exist.

    Action function - performs I/O to create directories.

    Args:
        context: FlowContext with training path information.
    """
    context.training.base_dir.mkdir(parents=True, exist_ok=True)
    context.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def load_config(
    config_path: str | Path,
    data_config_path: str | Path | None = None,
    solver_config_path: str | Path | None = None,
) -> tuple[GeneralSettings, FlowContext]:
    """Load and integrate model, data, and solver configurations.

    Orchestrates the full configuration loading pipeline:
    1. Validate and load model config TOML
    2. Validate and load data config TOML (if provided)
    3. Load solver config (use default if not provided)
    4. Build FlowContext from configs
    5. Load settings via dlkit's load_training_settings
    6. Inject solver params and context paths
    7. Ensure training directories exist
    8. Return integrated (settings, context)

    Args:
        config_path: Path to model config TOML file.
        data_config_path: Optional path to data config TOML file.
        solver_config_path: Optional path to solver config TOML file.
            Defaults to solver-configs/default.toml if not provided.

    Returns:
        Tuple of (BasicSettings, FlowContext) with all integrations applied.

    Raises:
        FileNotFoundError: If any required config file doesn't exist.
        ValueError: If configurations are malformed.
        tomllib.TOMLDecodeError: If TOML parsing fails.

    Examples:
        >>> settings, context = load_config(
        ...     "configs/ffnn.toml",
        ...     data_config_path="data-configs/collect-504.toml",
        ... )
        >>> print(context.training.base_dir)
        /data/projects/graph-cg/data/output/...
    """
    # Validate and load model config
    config_path = validate_file_exists(config_path, "Model config")
    model_config = _load_toml(config_path)

    # Validate and load data config if provided
    data_config = None
    data_config_path_obj = None
    if data_config_path is not None:
        data_config_path_obj = validate_file_exists(data_config_path, "Data config")
        data_config = _load_toml(data_config_path_obj)

    # Load solver config or use default
    if solver_config_path is None:
        solver_config_path_obj = _get_default_solver_config_path()
    else:
        solver_config_path_obj = validate_file_exists(
            solver_config_path, "Solver config"
        )
    solver_config = _load_toml(solver_config_path_obj)

    # Build FlowContext from configs
    context = _build_context(
        model_config, data_config, config_path, data_config_path_obj
    )

    # Ensure training directories exist BEFORE injecting paths
    _ensure_training_dirs(context)

    # Load settings using dlkit
    settings = load_training_settings(str(config_path))

    # Inject solver parameters
    settings = _inject_solver_params(settings, solver_config)

    # Inject context paths
    settings = _inject_context_paths(settings, context)

    return settings, context  # type: ignore[return-value]


def _resolve_experiment_config_path(
    experiment_name: str,
    config_filename: str,
    experiment_dir: Path,
    default_dir: Path,
    default_filename: str | None = None,
) -> Path:
    """
    Resolve a config file path using the experiment -> default fallback logic.
    """
    exp_path = experiment_dir / config_filename
    if exp_path.exists():
        return exp_path

    # Use specified default filename if provided, otherwise use standard config filename
    target_default_name = default_filename if default_filename else config_filename
    default_path = default_dir / target_default_name
    
    if default_path.exists():
        return default_path

    raise FileNotFoundError(
        f"Config file '{config_filename}' not found for experiment '{experiment_name}' "
        f"in '{exp_path.parent}'. Fallback '{target_default_name}' also not found in '{default_dir}'"
    )


def load_experiments(
    master_config_path: Path,
) -> list[tuple[str, GeneralSettings, FlowContext, Path, Path, Path]]:
    """
    Load all active experiment configurations based on the master config file.

    Args:
        master_config_path: Path to the TOML file listing experiments to run.

    Returns:
        A list of tuples, where each tuple contains:
        (experiment_name, settings, context, model_path, data_gen_path, solver_path).
    """
    if not master_config_path.exists():
        raise FileNotFoundError(f"Master config not found: {master_config_path}")

    config_dir = master_config_path.parent

    with open(master_config_path, "rb") as f:
        master_config_data = tomllib.load(f)
    
    # Resolve project root
    project_root_str = master_config_data.get("project_root", "..")
    project_root = (config_dir / project_root_str).resolve()

    # Resolve defaults directory
    defaults_dir_str = master_config_data.get("defaults_dir", "experiments/default")
    defaults_dir = (config_dir / defaults_dir_str).resolve()

    # Get experiments dictionary
    experiments_map = master_config_data.get("experiments", {})
    if not experiments_map:
        # Fallback for legacy "run" list, assuming standard structure if not defined
        run_list = master_config_data.get("run", [])
        experiments_map = {name: f"experiments/{name}" for name in run_list}

    resolved_experiments: list[tuple[str, GeneralSettings, FlowContext, Path, Path, Path]] = []
    for exp_name, exp_path_str in experiments_map.items():
        exp_dir = (config_dir / exp_path_str).resolve()

        # Resolve paths for all three config files
        # We search in the specific experiment dir, then fallback to defaults_dir
        model_path = _resolve_experiment_config_path(
            exp_name, EXP_MODEL_CONFIG_NAME, exp_dir, defaults_dir, default_filename="ffnn.toml"
        )
        data_pointer_path = _resolve_experiment_config_path(
            exp_name, EXP_DATA_CONFIG_NAME, exp_dir, defaults_dir
        )
        solver_path = _resolve_experiment_config_path(
            exp_name, EXP_SOLVER_CONFIG_NAME, exp_dir, defaults_dir
        )

        # The data.toml file is a pointer to the actual data generation config.
        with open(data_pointer_path, "rb") as f:
            data_pointer_config = tomllib.load(f)
        
        data_gen_path_str = data_pointer_config.get("dataconfig")
        if not data_gen_path_str:
            raise ValueError(f"The data.toml for experiment '{exp_name}' is missing the 'dataconfig' key.")
        
        # The path in dataconfig is relative to the project root.
        real_data_gen_path = project_root / data_gen_path_str

        # Use the existing load_config function to get the settings and context
        settings, context = load_config(
            config_path=model_path,
            data_config_path=real_data_gen_path,
            solver_config_path=solver_path,
        )
        
        resolved_experiments.append((exp_name, settings, context, model_path, real_data_gen_path, solver_path))

    return resolved_experiments
