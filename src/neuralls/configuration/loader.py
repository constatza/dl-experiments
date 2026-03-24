"""Configuration loading and orchestration.

This module provides the main entry points for loading experiment configurations.
All path resolution is delegated to the paths module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from neuralls.configuration.domain import (
    ExperimentBatch,
    ExperimentSpec,
    RunnableExperiment,
)
from neuralls.configuration.experiments import ExperimentsConfig, resolve_display_name
from neuralls.configuration.master_registry import (
    list_experiment_bindings,
    load_validated_master_config,
)
from neuralls.configuration.paths import build_path_context
from neuralls.configuration.services import WorkspaceFactory
from neuralls.configuration.settings import build_inference_settings, build_settings
from neuralls.configuration.mlflow_normalization import (
    build_mlflow_environment,
    derive_output_root_from_tracking_uri,
    derive_sqlite_artifacts_destination,
    scoped_mlflow_environment,
)
from neuralls.io.toml_loader import load_data_config, load_model_config, load_raw_toml


@dataclass(frozen=True)
class MlflowTopology:
    """Runtime MLflow topology for a training or inference workflow."""

    env: dict[str, str]
    experiment_name: str | None = None
    force_enabled: bool = False


def _resolve_relative_path(path: Path | None, base_dir: Path) -> Path | None:
    """Resolve an optional path relative to a config directory."""
    if path is None:
        return None
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _load_experiments_config(
    experiments_config_path: Path | None,
) -> ExperimentsConfig | None:
    """Load experiments topology when provided."""
    if experiments_config_path is None:
        return None
    cfg, _ = load_validated_master_config(Path(experiments_config_path))
    return cfg


def _resolve_output_override(
    *,
    output_root: Path | None,
    experiments_cfg: ExperimentsConfig | None,
    experiments_config_path: Path | None,
) -> Path | None:
    """Resolve output root from explicit override, experiments config, or defaults."""
    if output_root is not None:
        return output_root.resolve()
    if experiments_cfg is None or experiments_config_path is None:
        return None

    config_dir = Path(experiments_config_path).resolve().parent
    resolved_output = _resolve_relative_path(experiments_cfg.output_dir, config_dir)
    if resolved_output is not None:
        return resolved_output
    return derive_output_root_from_tracking_uri(
        experiments_cfg.mlflow.tracking_uri,
        config_path=Path(experiments_config_path),
    )


def _build_default_mlflow_topology(path_ctx_output_root: Path) -> MlflowTopology:
    """Build default MLflow env rooted under the output directory."""
    env = build_mlflow_environment(
        tracking_uri=f"sqlite:///{(path_ctx_output_root / 'mlruns' / 'mlflow.db').as_posix()}",
        artifacts_destination=str((path_ctx_output_root / "mlartifacts").resolve()),
    )
    return MlflowTopology(env=env, force_enabled=True)


def _build_experiments_mlflow_topology(
    experiments_cfg: ExperimentsConfig,
    experiments_config_path: Path,
) -> MlflowTopology:
    """Build MLflow env from experiments.toml."""
    env = build_mlflow_environment(
        tracking_uri=experiments_cfg.mlflow.tracking_uri,
        artifacts_destination=experiments_cfg.mlflow.artifacts_destination,
        config_path=experiments_config_path,
    )
    return MlflowTopology(
        env=env,
        experiment_name=experiments_cfg.names.training,
        force_enabled=True,
    )


def load_experiment(
    model_config_path: Path,
    data_config_path: Path,
    output_root: Path | None = None,
    mode: str = "training",
    experiments_config_path: Path | None = None,
    experiment_id: str | None = None,
    experiment_display_name: str | None = None,
    dataset_registry_id: str | None = None,
    dataset_display_name: str | None = None,
    model_registry_id: str | None = None,
    model_display_name: str | None = None,
) -> RunnableExperiment:
    """Load a single experiment configuration.

    This is the MAIN entry point for loading experiments.

    Args:
        model_config_path: Path to model config TOML.
        data_config_path: Path to data config TOML.
        output_root: Override for master output directory (optional).
        mode: Workflow mode - "training" or "inference" (default: "training").
              Training mode loads TrainingWorkflowConfig (requires DATASET/DATAMODULE).
              Inference mode loads InferenceWorkflowConfig (DATASET/DATAMODULE optional).
        experiments_config_path: Optional path to experiments TOML for MLflow topology
              injection. When provided, overrides MLFLOW settings in model config.

    Returns:
        RunnableExperiment with validated configs and workspace.

    Raises:
        ValueError: If configs are invalid or mode is invalid.
        FileNotFoundError: If config files don't exist.
    """
    # Validate mode
    if mode not in ("training", "inference"):
        raise ValueError(
            f"Invalid mode: {mode!r}. Expected 'training' or 'inference'."
        )
    experiments_cfg = _load_experiments_config(experiments_config_path)

    # 1. Load and validate configs (using existing loaders)
    data_cfg = load_data_config(data_config_path)

    # 2. Resolve base paths (SINGLE SOURCE OF TRUTH)
    resolved_output_root = _resolve_output_override(
        output_root=output_root,
        experiments_cfg=experiments_cfg,
        experiments_config_path=Path(experiments_config_path) if experiments_config_path else None,
    )
    project_override = None
    if experiments_cfg is not None and experiments_config_path is not None:
        project_override = _resolve_relative_path(
            experiments_cfg.project_root,
            Path(experiments_config_path).resolve().parent,
        )
    path_ctx = build_path_context(
        data_cfg,
        output_override=resolved_output_root,
        project_override=project_override,
    )
    mlflow_topology = (
        _build_experiments_mlflow_topology(
            experiments_cfg,
            Path(experiments_config_path),
        )
        if experiments_cfg is not None and experiments_config_path is not None
        else _build_default_mlflow_topology(path_ctx.output_root)
    )

    with scoped_mlflow_environment(mlflow_topology.env):
        model_cfg = load_model_config(model_config_path)

    # 3. Extract identifiers
    if dataset_registry_id is None:
        raise ValueError(
            "dataset_registry_id is required. Pass it from experiments.toml via load_batch()."
        )
    dataset_id = dataset_registry_id
    session = getattr(model_cfg, "SESSION", None)
    session_name = getattr(session, "name", None)
    model_name = getattr(getattr(model_cfg, "MODEL", None), "name", None)
    # Treat dlkit's default "dlkit-session" as unset, prefer MODEL.name for clarity
    if session_name and session_name != "dlkit-session":
        base_name_candidate = session_name
    else:
        base_name_candidate = model_name

    if not isinstance(base_name_candidate, str) or not base_name_candidate:
        raise ValueError(
            "Model name missing. Set [SESSION].name or [MODEL].name in model config."
        )
    base_name = base_name_candidate

    workspace_run_id = base_name

    # 4. Build experiment spec (use base_name for logical ID)
    resolved_experiment_id = experiment_id or base_name
    resolved_experiment_display_name = resolve_display_name(
        resolved_experiment_id,
        experiment_display_name,
    )
    spec = ExperimentSpec(
        experiment_id=resolved_experiment_id,
        experiment_display_name=resolved_experiment_display_name,
        dataset_registry_id=dataset_registry_id,
        dataset_display_name=dataset_display_name,
        model_registry_id=model_registry_id,
        model_display_name=model_display_name,
        model_config_path=model_config_path,
        data_config_path=data_config_path,
    )

    # 5. Create workspace
    factory = WorkspaceFactory(path_ctx.output_root, path_ctx.processed_root)
    workspace = factory.create(dataset_id, workspace_run_id)

    # 6. Build settings (mode-specific: training or inference)
    if mode == "inference":
        # Inference: Use InferenceWorkflowConfig (DATASET/DATAMODULE optional)
        # Transforms loaded from checkpoint metadata
        settings = build_inference_settings(
            model_config_path=model_config_path,
            workspace=workspace,
            path_context=path_ctx,
            mlflow_experiment_name=mlflow_topology.experiment_name,
            force_mlflow_enabled=mlflow_topology.force_enabled,
        )
        logger.debug(f"Loaded inference settings (DATASET/DATAMODULE optional)")
    else:
        # Training: Use TrainingWorkflowConfig (DATASET/DATAMODULE required)
        settings = build_settings(
            model_config_path=model_config_path,
            workspace=workspace,
            path_context=path_ctx,
            force_mlflow_enabled=mlflow_topology.force_enabled,
            base_settings=model_cfg,
        )
        logger.debug(f"Loaded training settings (DATASET/DATAMODULE required)")

    return RunnableExperiment(
        spec=spec,
        workspace=workspace,
        settings=settings,
    )


def load_batch(master_config_path: Path) -> ExperimentBatch:
    """Load all experiments from master config file.

    Supports format with [[experiment]] entries containing:
    - id: Experiment identifier
    - dataset: Dataset registry id (references configs/datasets/{dataset}.toml)
    - model: Model registry id (references configs/models/{model}.toml)
    - checkpoint_path: Optional explicit checkpoint path

    Args:
        master_config_path: Path to experiments.toml.

    Returns:
        ExperimentBatch with all runnable experiments.

    Raises:
        FileNotFoundError: If master config or experiment configs not found.
        ValueError: If config validation fails.
    """
    if not master_config_path.exists():
        raise FileNotFoundError(f"Master config not found: {master_config_path}")

    cfg, config_dir = load_validated_master_config(master_config_path)

    # Resolve global output root
    output_root = cfg.output_dir
    bindings = list_experiment_bindings(cfg, config_dir)
    if not bindings:
        raise ValueError(
            "No experiments defined. Expected [[experiments]] entries with "
            "id, dataset, model fields."
        )

    resolved_experiments = []

    for binding in bindings:
        experiment_id = binding.experiment_id
        data_path = binding.data_config_path
        model_path = binding.model_config_path
        checkpoint_path = binding.checkpoint_path

        # Validate paths exist
        if not data_path.exists():
            raise FileNotFoundError(
                f"Experiment '{experiment_id}': Dataset config not found: {data_path}"
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Experiment '{experiment_id}': Model config not found: {model_path}"
            )

        # Load experiment using main entry point
        experiment = load_experiment(
            model_path,
            data_path,
            output_root=output_root,
            experiment_id=experiment_id,
            experiment_display_name=binding.experiment_display_name,
            dataset_registry_id=binding.dataset_registry_id,
            dataset_display_name=binding.dataset_display_name,
            model_registry_id=binding.model_registry_id,
            model_display_name=binding.model_display_name,
        )

        if checkpoint_path is not None:
            if not checkpoint_path.exists():
                logger.warning(
                    f"Experiment '{experiment_id}': Checkpoint not found: {checkpoint_path}"
                )
            experiment = RunnableExperiment(
                spec=ExperimentSpec(
                    experiment_id=experiment.spec.experiment_id,
                    experiment_display_name=experiment.spec.experiment_display_name,
                    dataset_registry_id=experiment.spec.dataset_registry_id,
                    dataset_display_name=experiment.spec.dataset_display_name,
                    model_registry_id=experiment.spec.model_registry_id,
                    model_display_name=experiment.spec.model_display_name,
                    model_config_path=experiment.spec.model_config_path,
                    data_config_path=experiment.spec.data_config_path,
                    checkpoint_path=checkpoint_path,
                ),
                workspace=experiment.workspace,
                settings=experiment.settings,
            )

        resolved_experiments.append(experiment)

    # Determine final output root
    final_output_root = output_root
    if final_output_root is None:
        # Use first experiment's path context
        first_data_cfg = load_data_config(resolved_experiments[0].spec.data_config_path)
        path_ctx = build_path_context(first_data_cfg)
        final_output_root = path_ctx.output_root

    return ExperimentBatch(
        output_root=final_output_root,
        experiments=resolved_experiments,
    )
