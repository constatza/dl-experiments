"""Configuration loading and orchestration.

This module provides the main entry points for loading experiment configurations.
It orchestrates the new loader modules (toml_loader, path_resolver, settings_builder)
to construct strictly typed experiment definitions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dlkit.tools.config.core.updater import update_settings

logger = logging.getLogger(__name__)

from ..constants import (
    DEFAULT_FIGURES_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROCESSED_DATA_DIR,
    DEFAULT_PROJECT_ROOT,
    EXP_DATA_CONFIG_NAME,
    EXP_MODEL_CONFIG_NAME,
    EXP_SOLVER_CONFIG_NAME,
)
from ..validation import validate_file_exists
from .domain import (
    ExperimentBatch,
    ExperimentSpec,
    ExperimentWorkspace,
    RunnableExperiment,
)
from .loaders import (
    build_path_context,
    build_settings,
    load_data_config,
    load_model_config,
    load_raw_toml,
    load_solver_config,
    resolve_config_path,
    resolve_output_root,
    resolve_processed_root,
    resolve_project_root,
)
from .services import WorkspaceFactory

# Import paths module for FlowContext (needed by some scripts)
from neuralls.paths.core import (
    ComparisonPaths,
    DataPaths,
    FlowContext,
    FlowPaths,
    PredictionPaths,
    ProjectRoots,
    TrainingPaths,
)


def build_flow_context(
    spec: ExperimentSpec,
    workspace: ExperimentWorkspace,
    data_id: str,
    project_root: Path,
    *,
    processed_root: Path | None = None,
    output_root: Path | None = None,
    figures_root: Path | None = None,
) -> FlowContext:
    """Build a FlowContext using a single set of resolved roots.

    This function is kept for backward compatibility with scripts that use it.
    """
    roots = ProjectRoots.from_overrides(
        project_root=project_root,
        processed_root=processed_root or DEFAULT_PROCESSED_DATA_DIR,
        output_root=output_root or DEFAULT_OUTPUT_DIR,
        figures_root=figures_root or DEFAULT_FIGURES_DIR,
    )
    flow_paths = FlowPaths(flow_id=spec.id, roots=roots)
    data_paths = DataPaths(flow=flow_paths, dataset_id=data_id)

    # Load data config to check for test solutions
    data_config = load_data_config(spec.data_config_path)
    test_solutions_path = data_config.test.solutions_path

    training_paths = TrainingPaths(data=data_paths, run_id=spec.id)
    prediction_paths = PredictionPaths(training=training_paths)
    comparison_paths = ComparisonPaths(flow=flow_paths, dataset_id=data_id)

    return FlowContext(
        flow=flow_paths,
        data=data_paths,
        training=training_paths,
        prediction=prediction_paths,
        comparison=comparison_paths,
        run_id=spec.id,
        test_solutions_path=test_solutions_path,
    )


def load_data_context(
    data_config_path: Path | str | None,
    model_config_path: Path | str | None = None,
    solver_config_path: Path | str | None = None,
    output_root: Path | str | None = None,
) -> tuple[Any | None, dict[str, Any] | None, Path | None]:
    """Load data-specific context, optionally using a dummy model config.

    This function is kept for backward compatibility with data processing scripts.
    """
    if data_config_path is None:
        return None, None, None

    data_path_resolved = validate_file_exists(data_config_path, "Data config")

    # If no model config is provided, use a dummy one
    if model_config_path is None:
        dummy_model_path = DEFAULT_PROJECT_ROOT / "configs" / "experiments" / "default" / "dummy_model.toml"
        # Ensure the dummy model exists, create if not
        if not dummy_model_path.exists():
            # Fallback for unexpected state
            with open(dummy_model_path, "w") as f:
                f.write("[MODEL]\nname = \"dummy_model\"\nmodule_path = \"dlkit.nn.ffnn\"")
        model_path_to_use = dummy_model_path
    else:
        model_path_to_use = validate_file_exists(model_config_path, "Model config")

    try:
        # Load configs using new typed loaders
        model_cfg = load_model_config(model_path_to_use)
        data_cfg = load_data_config(data_path_resolved)

        # Resolve paths
        project_root = resolve_project_root(model_cfg)
        processed_root = resolve_processed_root(data_cfg, project_root)
        global_output_dir = resolve_output_root(output_root, project_root)

        # Load full experiment for context building
        experiment = load_experiment(
            model_path_to_use,
            data_path_resolved,
            solver_config_path,
            output_root,
        )

        # Return data config as dict for backward compatibility
        data_config_content = load_raw_toml(data_path_resolved)

        context = build_flow_context(
            experiment.spec,
            experiment.workspace,
            experiment.spec.data_config_path.stem,
            project_root,
            processed_root=processed_root,
            output_root=global_output_dir,
            figures_root=DEFAULT_FIGURES_DIR,
        )

        return context, data_config_content, data_path_resolved
    except ValueError as e:
        # If load_experiment fails, return None
        if "data_config_path is required" in str(e):
            return None, None, None
        raise e


def load_experiment(
    model_config_path: Path | str,
    data_config_path: Path | str | None = None,
    solver_config_path: Path | str | None = None,
    output_root: Path | str | None = None,
) -> RunnableExperiment:
    """Load a single experiment configuration (CLI entry point).

    Args:
        model_config_path: Path to model.toml file.
        data_config_path: Path to data config file (required).
        solver_config_path: Path to solver.toml file (optional, uses default if not provided).
        output_root: Override for output directory (optional).

    Returns:
        RunnableExperiment with validated configs and constructed workspace.

    Raises:
        ValueError: If data_config_path is not provided.
        ConfigLoadError: If config files are invalid or cannot be loaded.
    """
    # 1. Validate and load model config
    model_path = validate_file_exists(model_config_path, "Model config")
    model_cfg = load_model_config(model_path)

    # 2. Validate and load data config
    if not data_config_path:
        raise ValueError("data_config_path is required for loading a single experiment.")
    data_path = validate_file_exists(data_config_path, "Data config")
    data_cfg = load_data_config(data_path)

    # 3. Resolve or use default solver config
    if solver_config_path:
        solver_path = validate_file_exists(solver_config_path, "Solver config")
    else:
        solver_path = DEFAULT_PROJECT_ROOT / "configs" / "experiments" / "default" / "solver.toml"
    solver_cfg = load_solver_config(solver_path)

    # 4. Resolve paths using typed functions
    path_ctx = build_path_context(model_cfg, data_cfg, output_root)

    # 5. Extract identifiers from validated configs
    data_id = data_path.stem
    model_name = model_cfg.SESSION.name or model_cfg.MODEL.name

    if not model_name:
        raise ValueError(
            "Model name missing. Please set [SESSION].name or [MODEL].name in the model config."
        )

    # 6. Build domain objects
    spec = ExperimentSpec(
        id=model_name,
        model_config_path=model_path,
        data_config_path=data_path,
        solver_config_path=solver_path,
    )

    workspace_factory = WorkspaceFactory(path_ctx.output_root, path_ctx.processed_root)
    workspace = workspace_factory.create(data_id, model_name)

    workspace.root_dir.mkdir(parents=True, exist_ok=True)
    workspace.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 7. Build settings using typed configuration
    settings = build_settings(
        model_path,
        model_cfg,
        solver_cfg,
        workspace,
        path_ctx.project_root,
        path_ctx.processed_root,
    )

    return RunnableExperiment(
        spec=spec,
        workspace=workspace,
        settings=settings,
    )


def load_batch(master_config_path: Path) -> ExperimentBatch:
    """Load all experiments defined in the master configuration file.

    Supports new format with [[experiment]] entries containing:
    - id: Experiment identifier
    - dataset: Dataset ID (references configs/datasets/{dataset}.toml)
    - model: Model ID (references configs/models/{model}.toml)
    - solver: Solver ID (references configs/solvers/{solver}.toml, defaults to "default")
    - checkpoint_path: Optional explicit checkpoint path

    Args:
        master_config_path: Path to experiments.toml.

    Returns:
        A strictly typed ExperimentBatch containing all runnable experiments.

    Raises:
        FileNotFoundError: If master config or required experiment configs not found.
        ConfigLoadError: If config validation fails.
    """
    if not master_config_path.exists():
        raise FileNotFoundError(f"Master config not found: {master_config_path}")

    config_dir = master_config_path.parent
    master_config = load_raw_toml(master_config_path)

    # 1. Resolve project root
    project_root_str = master_config.get("project_root", "..")
    project_root = (config_dir / project_root_str).resolve()

    # 2. Resolve global output directory
    output_dir_str = master_config.get("output_dir")
    global_output_dir = resolve_output_root(None, project_root, output_dir_str)

    # 3. Load experiment entries (new format: [[experiment]] array)
    experiments_list = master_config.get("experiment", [])
    if not experiments_list:
        raise ValueError(
            "No experiments defined in experiments.toml. "
            "Expected [[experiment]] entries with id, dataset, model, solver fields."
        )

    resolved_experiments = []

    for exp_entry in experiments_list:
        # A. Extract experiment definition
        exp_id = exp_entry.get("id")
        dataset_id = exp_entry.get("dataset")
        model_id = exp_entry.get("model")
        solver_id = exp_entry.get("solver", "default")
        checkpoint_path_str = exp_entry.get("checkpoint_path")

        if not exp_id or not dataset_id or not model_id:
            raise ValueError(
                f"Experiment entry missing required fields. "
                f"Got: {exp_entry}. Required: id, dataset, model."
            )

        # B. Resolve config paths from shared directories
        data_path = (config_dir / "datasets" / f"{dataset_id}.toml").resolve()
        model_path = (config_dir / "models" / f"{model_id}.toml").resolve()
        solver_path = (config_dir / "solvers" / f"{solver_id}.toml").resolve()

        # Validate paths exist
        if not data_path.exists():
            raise FileNotFoundError(
                f"Experiment '{exp_id}': Dataset config not found: {data_path}"
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Experiment '{exp_id}': Model config not found: {model_path}"
            )
        if not solver_path.exists():
            raise FileNotFoundError(
                f"Experiment '{exp_id}': Solver config not found: {solver_path}"
            )

        # C. Load and validate configs
        model_cfg = load_model_config(model_path)
        data_cfg = load_data_config(data_path)
        solver_cfg = load_solver_config(solver_path)

        # D. Extract identifiers
        data_id = data_path.stem
        model_name = model_cfg.SESSION.name or model_cfg.MODEL.name

        if not model_name:
            raise ValueError(
                f"Experiment '{exp_id}': Could not determine model name from {model_path}. "
                "Ensure [MODEL].name or [SESSION].name is set."
            )

        # E. Resolve checkpoint path if provided
        checkpoint_path: Path | None = None
        if checkpoint_path_str:
            checkpoint_path = Path(checkpoint_path_str).resolve()
            if not checkpoint_path.exists():
                logger.warning(
                    f"Experiment '{exp_id}': Checkpoint not found: {checkpoint_path}. "
                    "Will use latest from checkpoint_dir at runtime."
                )

        # F. Build domain objects
        spec = ExperimentSpec(
            id=exp_id,
            model_config_path=model_path,
            data_config_path=data_path,
            solver_config_path=solver_path,
            checkpoint_path=checkpoint_path,
        )

        processed_root = resolve_processed_root(data_cfg, project_root)
        workspace_factory = WorkspaceFactory(global_output_dir, processed_root)
        workspace = workspace_factory.create(data_id, model_name)

        # Ensure workspace directories exist
        workspace.root_dir.mkdir(parents=True, exist_ok=True)
        workspace.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # F. Build settings
        settings = build_settings(
            model_path,
            model_cfg,
            solver_cfg,
            workspace,
            project_root,
            processed_root,
        )

        # Update with workspace-specific processed_dir
        settings = update_settings(
            settings,
            {
                "PATHS": {
                    "processed_dir": str(workspace.data_dir.parent),
                }
            }
        )

        resolved_experiments.append(
            RunnableExperiment(
                spec=spec,
                workspace=workspace,
                settings=settings,
            )
        )

    return ExperimentBatch(
        global_output_dir=global_output_dir,
        experiments=resolved_experiments
    )
