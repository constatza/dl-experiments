"""Configuration loading and orchestration.

This module is responsible for parsing configuration files and constructing
strictly typed experiment definitions using the domain models.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from dlkit.tools.config import load_training_settings
from dlkit.tools.config.core.updater import update_settings

from ..constants import (
    DEFAULT_PROJECT_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROCESSED_DATA_DIR,
    DEFAULT_FIGURES_DIR,
    EXP_MODEL_CONFIG_NAME,
    EXP_DATA_CONFIG_NAME,
    EXP_SOLVER_CONFIG_NAME,
    ConfigKeys,
)
from ..validation import validate_file_exists

from .domain import (
    ExperimentSpec,
    ExperimentWorkspace,
    RunnableExperiment,
    ExperimentBatch,
)
from .services import WorkspaceFactory


def _load_toml(path: Path) -> dict[str, Any]:
    """Load TOML file into dictionary."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def _apply_datamodule_overrides(settings: Any, model_config: Mapping[str, Any]) -> Any:
    """Re-apply DATAMODULE settings, ensuring nested dataloader options are preserved."""
    datamodule_cfg = dict(model_config.get("DATAMODULE") or {})
    dataloader_cfg = datamodule_cfg.get("dataloader")
    if isinstance(dataloader_cfg, Mapping):
        dataloader_updates = dict(dataloader_cfg)
        num_workers = dataloader_updates.get("num_workers")
        if isinstance(num_workers, int) and num_workers <= 0:
            dataloader_updates.setdefault("persistent_workers", False)
        datamodule_cfg["dataloader"] = dataloader_updates

    if datamodule_cfg:
        settings = update_settings(settings, {"DATAMODULE": datamodule_cfg})
    return settings


def _resolve_config_path(
    experiment_name: str,
    config_filename: str,
    experiment_dir: Path,
    default_dir: Path,
    default_filename: str | None = None,
) -> Path:
    """Resolve a config file path using experiment -> default fallback."""
    exp_path = experiment_dir / config_filename
    if exp_path.exists():
        return exp_path

    target_default_name = default_filename if default_filename else config_filename
    default_path = default_dir / target_default_name
    
    if default_path.exists():
        return default_path

    raise FileNotFoundError(
        f"Config file '{config_filename}' not found for experiment '{experiment_name}' "
        f"in '{exp_path.parent}'. Fallback '{target_default_name}' also not found in '{default_dir}'"
    )





def _inject_paths_and_params(
    settings: Any, 
    workspace: ExperimentWorkspace, 
    solver_config: dict[str, Any],
    project_root: Path,
) -> Any:
    """Inject workspace paths, solver params, and MLflow config into settings."""
    # Inject Paths
    settings = update_settings(
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
    
    # Inject Solver Params
    settings = update_settings(
        settings,
        {
            "EXTRAS": {
                "solver_config": solver_config,
            }
        },
    )

    # Inject/Resolve MLflow Configuration
    # We respect the value in model.toml if present, otherwise default to project/data/mlruns
    
    # Check if MLFLOW section exists
    mlf_cfg = settings.MLFLOW
    if mlf_cfg is not None:
        server_cfg = mlf_cfg.server if hasattr(mlf_cfg, "server") else None
        existing_uri = server_cfg.backend_store_uri if server_cfg else None

        resolved_uri = existing_uri

        if not existing_uri:
            # Default to project-local mlruns
            mlruns_db_path = project_root / "data" / "mlruns" / "mlflow.db"
            resolved_uri = f"sqlite:///{mlruns_db_path.as_posix()}"
        elif isinstance(existing_uri, str) and existing_uri.startswith("sqlite:///"):
            # It's a URI, check if the path part is relative
            path_part = existing_uri.replace("sqlite:///", "")
            # If it looks like a relative path (not starting with /), anchor it to project root
            if not path_part.startswith("/"):
                full_path = project_root / path_part
                resolved_uri = f"sqlite:///{full_path.as_posix()}"

        settings = update_settings(
            settings,
            {
                "MLFLOW": {
                    "server": {
                        "backend_store_uri": resolved_uri
                    }
                }
            }
        )

    return settings


from src.paths.core import (
    ProjectRoots,
    FlowPaths,
    DataPaths,
    TrainingPaths,
    PredictionPaths,
    ComparisonPaths,
    FlowContext,
)


def _coerce_project_root(model_config: Mapping[str, Any]) -> Path:
    override = (model_config.get("PATHS") or {}).get("project_root")
    if isinstance(override, str) and override:
        return Path(override).resolve()
    return DEFAULT_PROJECT_ROOT


def _coerce_processed_root(
    data_config: Mapping[str, Any],
    project_root: Path,
) -> Path:
    processed_dir = (data_config.get("output") or {}).get(ConfigKeys.PROCESSED_DIR)
    if isinstance(processed_dir, str) and processed_dir:
        processed_path = Path(processed_dir)
        if not processed_path.is_absolute():
            processed_path = (project_root / processed_path).resolve()
        return processed_path
    return DEFAULT_PROCESSED_DATA_DIR


def _resolve_output_root(
    output_root: Path | str | None,
    project_root: Path,
    fallback: str | None = None,
) -> Path:
    if output_root:
        return Path(output_root).expanduser().resolve()
    if isinstance(fallback, str) and fallback:
        candidate = Path(fallback)
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        return candidate
    return DEFAULT_OUTPUT_DIR


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
    """Build a FlowContext using a single set of resolved roots."""
    roots = ProjectRoots.from_overrides(
        project_root=project_root,
        processed_root=processed_root or DEFAULT_PROCESSED_DATA_DIR,
        output_root=output_root or DEFAULT_OUTPUT_DIR,
        figures_root=figures_root or DEFAULT_FIGURES_DIR,
    )
    flow_paths = FlowPaths(flow_id=spec.id, roots=roots)
    data_paths = DataPaths(flow=flow_paths, dataset_id=data_id)

    data_config_raw = _load_toml(spec.data_config_path)
    test_solutions_path = data_config_raw.get("test", {}).get("solutions_path")

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
    """Load data-specific context, optionally using a dummy model config."""
    if data_config_path is None:
        return None, None, None

    data_path_resolved = validate_file_exists(data_config_path, "Data config")

    # If no model config is provided, use a dummy one
    if model_config_path is None:
        dummy_model_path = DEFAULT_PROJECT_ROOT / "configs" / "experiments" / "default" / "dummy_model.toml"
        # Ensure the dummy model exists, create if not (should have been created by agent already)
        if not dummy_model_path.exists():
            # Fallback for unexpected state, should not happen if previous step worked
            with open(dummy_model_path, "w") as f:
                f.write("[MODEL]\nname = \"dummy_model\"")
        model_path_to_use = dummy_model_path
    else:
        model_path_to_use = validate_file_exists(model_config_path, "Model config")

    try:
        model_config_raw = _load_toml(model_path_to_use)
        project_root = _coerce_project_root(model_config_raw)
        data_config_raw = _load_toml(data_path_resolved)
        processed_root = _coerce_processed_root(data_config_raw, project_root)
        global_output_dir = _resolve_output_root(output_root, project_root)

        experiment = load_experiment(
            model_path_to_use,
            data_path_resolved,
            solver_config_path,
            output_root,
        )
        data_config_content = data_config_raw

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
        # If load_experiment fails for some reason, return None for context
        # This handles cases where data_config_path is invalid even if not None
        if "data_config_path is required" in str(e):
            return None, None, None
        raise e


def load_experiment(
    model_config_path: Path | str,
    data_config_path: Path | str | None = None,
    solver_config_path: Path | str | None = None,
    output_root: Path | str | None = None,
) -> RunnableExperiment:
    """Load a single experiment configuration (CLI entry point)."""
    model_path = validate_file_exists(model_config_path, "Model config")

    model_config_raw = _load_toml(model_path)
    project_root = _coerce_project_root(model_config_raw)
    global_output_dir = _resolve_output_root(output_root, project_root)

    # 2. Resolve Data Config
    if data_config_path:
        data_gen_path = validate_file_exists(data_config_path, "Data config")
    else:
        # Try to find data config via pointer in experiment dir? 
        # Or just fail? CLI usually requires explicit paths or has defaults.
        # Let's fallback to checking if model config has a pointer? No, standard is explicit.
        raise ValueError("data_config_path is required for loading a single experiment.")

    # Load data config to find processed_dir
    data_config_raw = _load_toml(data_gen_path)
    processed_root = _coerce_processed_root(data_config_raw, project_root)

    # 3. Resolve Solver Config
    if solver_config_path:
        solver_path = validate_file_exists(solver_config_path, "Solver config")
    else:
        # Default solver config
        solver_path = DEFAULT_PROJECT_ROOT / "configs" / "experiments" / "default" / "solver.toml"
        if not solver_path.exists():
             # Fallback to absolute default if in a different env?
             pass 

    # 4. Extract Identifiers
    data_id = data_gen_path.stem
    model_name = (
        (model_config_raw.get("SESSION") or {}).get("name") or 
        (model_config_raw.get("MODEL") or {}).get("name")
    )

    if not model_name:
        raise ValueError(
            "Model name missing. Please set [SESSION].name or [MODEL].name in the model config."
        )

    # 5. Build Domain Objects
    # For CLI runs, we might not have a clean "experiment ID" from experiments.toml.
    # We use model_name as the ID.
    spec = ExperimentSpec(
        id=model_name,
        model_config_path=model_path,
        data_config_path=data_gen_path,
        solver_config_path=solver_path,
    )

    # 3. Initialize Services
    workspace_factory = WorkspaceFactory(global_output_dir, processed_root)
    workspace = workspace_factory.create(data_id, model_name)
    
    workspace.root_dir.mkdir(parents=True, exist_ok=True)
    workspace.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 6. Load Settings
    settings = load_training_settings(str(model_path))
    settings = _apply_datamodule_overrides(settings, model_config_raw)
    solver_config = _load_toml(solver_path)
    settings = _inject_paths_and_params(settings, workspace, solver_config, project_root)

    settings = update_settings(
        settings,
        {
            "PATHS": {
                "project_root": str(project_root),
                "processed_dir": str(processed_root),
            }
        }
    )

    return RunnableExperiment(
        spec=spec,
        workspace=workspace,
        settings=settings,
    )


def load_batch(master_config_path: Path) -> ExperimentBatch:
    """
    Load all experiments defined in the master configuration file.

    Args:
        master_config_path: Path to experiments.toml.

    Returns:
        A strictly typed ExperimentBatch containing all runnable experiments.
    """
    if not master_config_path.exists():
        raise FileNotFoundError(f"Master config not found: {master_config_path}")

    config_dir = master_config_path.parent
    master_config = _load_toml(master_config_path)

    # 1. Resolve Project Root
    project_root_str = master_config.get("project_root", "..")
    project_root = (config_dir / project_root_str).resolve()

    # 2. Resolve Global Output Directory (predictions/artifacts)
    output_dir_str = master_config.get("output_dir")
    global_output_dir = _resolve_output_root(None, project_root, output_dir_str)

    # 4. Resolve Experiment List
    experiments_map = master_config.get("experiments", {})
    if not experiments_map:
        run_list = master_config.get("run", [])
        experiments_map = {name: f"experiments/{name}" for name in run_list}

    defaults_dir = (config_dir / master_config.get("defaults_dir", "experiments/default")).resolve()

    resolved_experiments = []

    for exp_id, exp_path_rel in experiments_map.items():
        exp_dir = (config_dir / exp_path_rel).resolve()

        # A. Resolve Config Paths
        model_path = _resolve_config_path(
            exp_id, EXP_MODEL_CONFIG_NAME, exp_dir, defaults_dir, default_filename="ffnn.toml"
        )
        data_pointer_path = _resolve_config_path(
            exp_id, EXP_DATA_CONFIG_NAME, exp_dir, defaults_dir
        )
        solver_path = _resolve_config_path(
            exp_id, EXP_SOLVER_CONFIG_NAME, exp_dir, defaults_dir
        )

        # B. Resolve Data Config (Dereference pointer)
        data_pointer = _load_toml(data_pointer_path)
        data_gen_path_str = data_pointer.get("dataconfig")
        if not data_gen_path_str:
            raise ValueError(f"Experiment '{exp_id}': data.toml missing 'dataconfig' key.")
        data_gen_path = (project_root / data_gen_path_str).resolve()

        # C. Extract Identifiers
        # Data ID comes from the filename of the data generation config
        data_id = data_gen_path.stem
        
        # Model Name comes STRICTLY from model.toml [MODEL] or [SESSION]
        model_config_raw = _load_toml(model_path)
        model_name = (
            (model_config_raw.get("SESSION") or {}).get("name") or 
            (model_config_raw.get("MODEL") or {}).get("name")
        )
        
        if not model_name:
            raise ValueError(
                f"Experiment '{exp_id}': Could not determine model name from {model_path}. "
                "Ensure [MODEL].name or [SESSION].name is set."
            )

        # D. Build Domain Objects
        spec = ExperimentSpec(
            id=exp_id,
            model_config_path=model_path,
            data_config_path=data_gen_path,
            solver_config_path=solver_path,
        )

        data_config_raw = _load_toml(data_gen_path)
        processed_root = _coerce_processed_root(data_config_raw, project_root)
        workspace_factory = WorkspaceFactory(global_output_dir, processed_root)
        workspace = workspace_factory.create(data_id, model_name)
        
        # Ensure workspace directories exist
        workspace.root_dir.mkdir(parents=True, exist_ok=True)
        workspace.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # E. Load and Configure Settings
        settings = load_training_settings(str(model_path))
        settings = _apply_datamodule_overrides(settings, model_config_raw)
        solver_config = _load_toml(solver_path)
        settings = _inject_paths_and_params(settings, workspace, solver_config, project_root)

        # Update settings with processed_dir
        settings = update_settings(
            settings,
            {
                "PATHS": {
                    "project_root": str(project_root),
                    "processed_dir": str(workspace.data_dir.parent), # data_dir is specific dataset dir, processed_dir is root
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
