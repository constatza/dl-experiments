"""Configuration loading and orchestration.

This module provides the main entry points for loading experiment configurations.
All path resolution is delegated to the paths module.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from neuralls.configuration.domain import (
    ExperimentBatch,
    ExperimentSpec,
    RunnableExperiment,
)
from neuralls.configuration.paths import build_path_context
from neuralls.configuration.services import WorkspaceFactory
from neuralls.configuration.settings import build_settings
from neuralls.io.toml_loader import load_data_config, load_model_config, load_raw_toml


def load_experiment(
    model_config_path: Path,
    data_config_path: Path,
    output_root: Path | None = None,
) -> RunnableExperiment:
    """Load a single experiment configuration.

    This is the MAIN entry point for loading experiments.

    Args:
        model_config_path: Path to model config TOML.
        data_config_path: Path to data config TOML.
        output_root: Override for master output directory (optional).

    Returns:
        RunnableExperiment with validated configs and workspace.

    Raises:
        ValueError: If configs are invalid.
        FileNotFoundError: If config files don't exist.
    """
    # 1. Load and validate configs (using existing loaders)
    model_cfg = load_model_config(model_config_path)
    data_cfg = load_data_config(data_config_path)

    # 2. Resolve base paths (SINGLE SOURCE OF TRUTH)
    path_ctx = build_path_context(
        data_cfg,
        output_override=output_root,
    )

    # 3. Extract identifiers
    dataset_id = data_config_path.stem
    session_name = model_cfg.SESSION.name
    # Treat dlkit's default "dlkit-session" as unset, prefer MODEL.name for clarity
    if session_name and session_name != "dlkit-session":
        run_id = session_name
    else:
        run_id = model_cfg.MODEL.name

    if not run_id:
        raise ValueError(
            "Model name missing. Set [SESSION].name or [MODEL].name in model config."
        )

    # 4. Build experiment spec
    spec = ExperimentSpec(
        id=run_id,
        model_config_path=model_config_path,
        data_config_path=data_config_path,
    )

    # 5. Create workspace (directory structure)
    factory = WorkspaceFactory(path_ctx.output_root, path_ctx.processed_root)
    workspace = factory.create(dataset_id, run_id)

    # 6. Build settings with injected paths (including MLflow)
    settings = build_settings(
        model_config_path=model_config_path,
        workspace=workspace,
        path_context=path_ctx,
    )

    return RunnableExperiment(
        spec=spec,
        workspace=workspace,
        settings=settings,
    )


def load_batch(master_config_path: Path) -> ExperimentBatch:
    """Load all experiments from master config file.

    Supports format with [[experiment]] entries containing:
    - id: Experiment identifier
    - dataset: Dataset ID (references configs/datasets/{dataset}.toml)
    - model: Model ID (references configs/models/{model}.toml)
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

    config_dir = master_config_path.parent
    master_config = load_raw_toml(master_config_path)

    # Resolve global output root
    output_root_str = master_config.get("output_dir")
    output_root = Path(output_root_str) if output_root_str else None

    # Load experiment entries
    experiments_list = master_config.get("experiment", [])
    if not experiments_list:
        raise ValueError(
            "No experiments defined. Expected [[experiment]] entries with "
            "id, dataset, model fields."
        )

    resolved_experiments = []

    for exp_entry in experiments_list:
        # Extract experiment definition
        exp_id = exp_entry.get("id")
        dataset_id = exp_entry.get("dataset")
        model_id = exp_entry.get("model")
        checkpoint_path_str = exp_entry.get("checkpoint_path")

        if not all([exp_id, dataset_id, model_id]):
            raise ValueError(
                f"Experiment entry missing required fields. Got: {exp_entry}. "
                "Required: id, dataset, model."
            )

        # Resolve config paths
        data_path = (config_dir / "datasets" / f"{dataset_id}.toml").resolve()
        model_path = (config_dir / "models" / f"{model_id}.toml").resolve()

        # Validate paths exist
        if not data_path.exists():
            raise FileNotFoundError(
                f"Experiment '{exp_id}': Dataset config not found: {data_path}"
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Experiment '{exp_id}': Model config not found: {model_path}"
            )

        # Load experiment using main entry point
        experiment = load_experiment(
            model_path,
            data_path,
            output_root=output_root,
        )

        # Optionally override checkpoint path
        if checkpoint_path_str:
            checkpoint_path = Path(checkpoint_path_str).resolve()
            if not checkpoint_path.exists():
                logger.warning(
                    f"Experiment '{exp_id}': Checkpoint not found: {checkpoint_path}"
                )
            # Update spec with explicit checkpoint
            experiment = RunnableExperiment(
                spec=ExperimentSpec(
                    id=experiment.spec.id,
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
