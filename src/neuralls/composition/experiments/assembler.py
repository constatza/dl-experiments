"""Case-config loading and experiment assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from neuralls.platform.config.models.workspace import (
    ExperimentBatch,
    ExperimentSpec,
    RunnableExperiment,
)
from neuralls.platform.config.models.experiments import CaseConfig, resolve_display_name
from neuralls.platform.config.registry import list_experiment_bindings
from neuralls.platform.config.resolution import (
    derive_output_root_from_tracking_uri,
    resolve_case_config_path,
    resolve_path_context,
)
from neuralls.platform.config.settings import (
    NeurallsSettings,
    require_settings,
)
from neuralls.platform.storage.workspaces import WorkspaceFactory
from neuralls.platform.config.dlkit_bridge import (
    build_inference_settings,
    build_settings,
    load_model_config,
)
from neuralls.platform.config.loaders import (
    load_case_config,
    load_data_config,
)
from neuralls.platform.tracking.environment import scoped_mlflow_environment
from neuralls.platform.tracking.mlflow import build_workflow_environment


@dataclass(frozen=True)
class MlflowTopology:
    """Runtime MLflow topology for a training or inference workflow."""

    env: dict[str, str]
    experiment_name: str | None = None
    force_enabled: bool = False


def load_validated_case_config(
    config_path: Path,
    neuralls_settings: NeurallsSettings | None = None,
) -> tuple[CaseConfig, Path]:
    """Load one case config and validate all registry-backed references."""
    neuralls_settings = require_settings(
        neuralls_settings,
        case_config_path=config_path,
    )
    cfg = load_case_config(config_path, neuralls_settings)
    config_dir = config_path.resolve().parent
    return cfg, config_dir


def _load_case_config(
    case_config_path: Path | None,
    neuralls_settings: NeurallsSettings | None = None,
) -> CaseConfig | None:
    """Load case topology when provided explicitly or via environment."""
    resolved_case_config = resolve_case_config_path(case_config_path)
    if resolved_case_config is None:
        return None
    cfg, _ = load_validated_case_config(resolved_case_config, neuralls_settings)
    return cfg


def _resolve_output_override(
    *,
    output_root: Path | None,
    case_cfg: CaseConfig | None,
    case_config_path: Path | None,
    neuralls_settings: NeurallsSettings,
) -> Path | None:
    """Resolve output root from explicit override or case config."""
    if output_root is not None:
        return output_root.resolve()
    if case_cfg is None:
        return neuralls_settings.output_dir
    if case_config_path is None:
        return neuralls_settings.output_dir

    tracking_uri = case_cfg.mlflow.tracking_uri
    if tracking_uri is None:
        return neuralls_settings.output_dir
    return derive_output_root_from_tracking_uri(tracking_uri, config_path=case_config_path)


def _build_default_mlflow_topology(path_ctx_output_root: Path) -> MlflowTopology:
    """Build default MLflow env rooted under the output directory."""
    runtime = build_workflow_environment(
        tracking_uri=None,
        artifact_location=None,
        default_output_root=path_ctx_output_root,
    )
    return MlflowTopology(env=runtime.env, force_enabled=True)


def _build_case_mlflow_topology(
    case_cfg: CaseConfig,
    case_config_path: Path,
    neuralls_settings: NeurallsSettings,
) -> MlflowTopology:
    """Build MLflow env from case config or derive it from output_dir."""
    runtime = build_workflow_environment(
        tracking_uri=case_cfg.mlflow.tracking_uri,
        artifact_location=case_cfg.mlflow.artifacts_destination,
        default_output_root=neuralls_settings.output_dir,
        config_path=case_config_path,
    )
    return MlflowTopology(
        env=runtime.env,
        experiment_name=case_cfg.names.training,
        force_enabled=True,
    )


def load_experiment(
    model_config_path: Path,
    data_config_path: Path,
    neuralls_settings: NeurallsSettings | None = None,
    output_root: Path | None = None,
    mode: str = "training",
    case_config_path: Path | None = None,
    experiment_id: str | None = None,
    experiment_display_name: str | None = None,
    dataset_registry_id: str | None = None,
    dataset_display_name: str | None = None,
    model_registry_id: str | None = None,
    model_display_name: str | None = None,
) -> RunnableExperiment:
    """Load a single experiment configuration.

    Args:
        model_config_path: Path to model config TOML.
        data_config_path: Path to data config TOML.
        output_root: Override for case output directory (optional).
        mode: Workflow mode - "training" or "inference" (default: "training").
        case_config_path: Optional path to a case TOML for settings and MLflow topology.

    Returns:
        RunnableExperiment with validated configs and workspace.

    Raises:
        ValueError: If configs are invalid or mode is invalid.
        FileNotFoundError: If config files don't exist.
    """
    resolved_case_config_path = resolve_case_config_path(case_config_path)
    neuralls_settings = require_settings(
        neuralls_settings,
        case_config_path=resolved_case_config_path,
    )
    if mode not in ("training", "inference"):
        raise ValueError(f"Invalid mode: {mode!r}. Expected 'training' or 'inference'.")
    case_cfg = _load_case_config(resolved_case_config_path, neuralls_settings)

    data_cfg = load_data_config(data_config_path, neuralls_settings)

    resolved_output_root = _resolve_output_override(
        output_root=output_root,
        case_cfg=case_cfg,
        case_config_path=resolved_case_config_path,
        neuralls_settings=neuralls_settings,
    )
    path_ctx = resolve_path_context(
        processed_root=neuralls_settings.processed_dir,
        output_root=neuralls_settings.output_dir,
        data_dir_override=data_cfg.output.data_dir,
        output_override=resolved_output_root,
    )
    mlflow_topology = (
        _build_case_mlflow_topology(
            case_cfg,
            resolved_case_config_path,
            neuralls_settings,
        )
        if case_cfg is not None and resolved_case_config_path is not None
        else _build_default_mlflow_topology(path_ctx.output_root)
    )

    with scoped_mlflow_environment(mlflow_topology.env):
        model_cfg = load_model_config(model_config_path, neuralls_settings)

    if dataset_registry_id is None:
        raise ValueError(
            "dataset_registry_id is required. Pass it from the case config via load_batch()."
        )
    dataset_id = dataset_registry_id
    session = getattr(model_cfg, "SESSION", None)
    session_name = getattr(session, "name", None)
    model_name = getattr(getattr(model_cfg, "MODEL", None), "name", None)
    if session_name and session_name != "dlkit-session":
        base_name_candidate = session_name
    else:
        base_name_candidate = model_name

    if not isinstance(base_name_candidate, str) or not base_name_candidate:
        raise ValueError("Model name missing. Set [SESSION].name or [MODEL].name in model config.")
    base_name = base_name_candidate

    workspace_run_id = base_name

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

    factory = WorkspaceFactory(path_ctx.output_root, path_ctx.processed_root)
    workspace = factory.create(dataset_id, workspace_run_id)

    if mode == "inference":
        settings = build_inference_settings(
            model_config_path=model_config_path,
            workspace=workspace,
            data_cfg=data_cfg,
            settings=neuralls_settings,
            output_override=resolved_output_root,
            mlflow_experiment_name=mlflow_topology.experiment_name,
            force_mlflow_enabled=mlflow_topology.force_enabled,
        )
        logger.debug("Loaded inference settings (DATASET/DATAMODULE optional)")
    else:
        settings = build_settings(
            model_config_path=model_config_path,
            workspace=workspace,
            data_cfg=data_cfg,
            settings=neuralls_settings,
            output_override=resolved_output_root,
            force_mlflow_enabled=mlflow_topology.force_enabled,
            base_settings=model_cfg,
        )
        logger.debug("Loaded training settings (DATASET/DATAMODULE required)")

    return RunnableExperiment(
        spec=spec,
        workspace=workspace,
        settings=settings,
    )


def load_batch(
    case_config_path: Path,
    neuralls_settings: NeurallsSettings | None = None,
) -> ExperimentBatch:
    """Load all experiments from one case config file.

    Args:
        case_config_path: Path to case TOML.

    Returns:
        ExperimentBatch with all runnable experiments.

    Raises:
        FileNotFoundError: If case config or experiment configs are not found.
        ValueError: If config validation fails.
    """
    neuralls_settings = require_settings(
        neuralls_settings,
        case_config_path=case_config_path,
    )
    if not case_config_path.exists():
        raise FileNotFoundError(f"Case config not found: {case_config_path}")

    cfg, config_dir = load_validated_case_config(case_config_path, neuralls_settings)

    output_root = neuralls_settings.output_dir
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

        if not data_path.exists():
            raise FileNotFoundError(
                f"Experiment '{experiment_id}': Dataset config not found: {data_path}"
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Experiment '{experiment_id}': Model config not found: {model_path}"
            )

        experiment = load_experiment(
            model_path,
            data_path,
            neuralls_settings,
            output_root=output_root,
            case_config_path=case_config_path,
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

    final_output_root = output_root
    if final_output_root is None:
        first_data_cfg = load_data_config(
            resolved_experiments[0].spec.data_config_path,
            neuralls_settings,
        )
        path_ctx = resolve_path_context(
            processed_root=neuralls_settings.processed_dir,
            output_root=neuralls_settings.output_dir,
            data_dir_override=first_data_cfg.output.data_dir,
        )
        final_output_root = path_ctx.output_root

    return ExperimentBatch(
        output_root=final_output_root,
        experiments=resolved_experiments,
    )


load_validated_master_config = load_validated_case_config
