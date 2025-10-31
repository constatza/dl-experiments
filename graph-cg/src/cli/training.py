"""Training helpers consumed by CLI scripts and workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.interfaces.api import execute
from dlkit.tools.config.data_entries import Feature, Target
from dlkit.tools.config.core.updater import update_settings

from ..common import (
    load_config_with_context,
    get_paths_from_config,
    get_latest_checkpoint,
)
from ..constants import DEFAULT_FIGURES_DIR
from ..experiment_manifest import update_manifest
from ..validation import validate_file_exists, validate_directory_writable
from ..pca_training import (
    fit_pca_from_solutions,
    save_pca_model,
    plot_variance_ratios,
)
from ..common import derive_model_identifier, sanitize_identifier


def train_model(
    *,
    config_path: str | Path,
    data_config_path: str | Path | None = None,
    features_path: str | Path | None = None,
    targets_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    accelerator: str | None = None,
    session_name: str | None = None,
    checkpoint_filename: str | None = None,
    manifest_metadata: dict[str, Any] | None = None,
) -> Path:
    """Train a DLKit model using resolved data+config context."""

    settings, context = load_config_with_context(config_path, data_config_path)

    if features_path is not None or targets_path is not None:
        dataset_cfg = settings.DATASET
        if dataset_cfg is None:
            raise ValueError("Config is missing [DATASET] section")

        features = dataset_cfg.features
        targets = dataset_cfg.targets

        if features_path is not None:
            validated = validate_file_exists(features_path, "Features file")
            if features and len(features) > 0:
                features = (Feature(name=features[0].name, path=str(validated)),)
            else:
                features = (Feature(name="x", path=str(validated)),)

        if targets_path is not None:
            validated = validate_file_exists(targets_path, "Targets file")
            if targets and len(targets) > 0:
                targets = (Target(name=targets[0].name, path=str(validated)),)
            else:
                targets = (Target(name="y", path=str(validated)),)

        dataset_cfg = dataset_cfg.model_copy(update={"features": features, "targets": targets})
        settings = settings.model_copy(update={"DATASET": dataset_cfg})

    if session_name is not None:
        session_cfg = settings.SESSION
        if session_cfg is None:
            raise ValueError("Config is missing [SESSION] section")
        session_cfg = session_cfg.model_copy(update={"name": session_name})
        settings = settings.model_copy(update={"SESSION": session_cfg})
        context = context.with_run_id(session_name)

    if output_dir is not None or accelerator is not None:
        training_cfg = settings.TRAINING
        if training_cfg is None:
            raise ValueError("Config is missing [TRAINING] section")

        trainer_cfg = training_cfg.trainer
        callbacks = list(trainer_cfg.callbacks or [])

        if output_dir is not None:
            validated_dir = validate_directory_writable(output_dir, "Output directory")
            trainer_cfg = trainer_cfg.model_copy(
                update={"default_root_dir": str(validated_dir)}
            )

            checkpoint_dir = Path(validated_dir) / "checkpoints"
            updated_callbacks = []
            for cb in callbacks:
                if getattr(cb, "name", None) == "ModelCheckpoint":
                    updates: dict[str, Any] = {"dirpath": str(checkpoint_dir)}
                    if checkpoint_filename is not None:
                        updates["filename"] = checkpoint_filename
                    cb = cb.model_copy(update=updates)
                updated_callbacks.append(cb)
            callbacks = updated_callbacks

        if accelerator is not None:
            trainer_cfg = trainer_cfg.model_copy(update={"accelerator": accelerator})

        trainer_cfg = trainer_cfg.model_copy(update={"callbacks": callbacks})
        training_cfg = training_cfg.model_copy(update={"trainer": trainer_cfg})
        settings = settings.model_copy(update={"TRAINING": training_cfg})

    if settings.MLFLOW:
        settings = update_settings(
            settings,
            {"MLFLOW": {"client": {"experiment_name": context.data.dataset_id}}},
        )

    execute(settings, run_name=context.run_id)

    paths = get_paths_from_config(settings, context)
    checkpoint_dir = Path(paths["checkpoint_dir"])
    checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        raise RuntimeError(f"No checkpoint found in {checkpoint_dir}")

    experiment_dir = checkpoint_dir.parent
    data_dir = Path(paths["dataset_dir"])

    if data_config_path is not None:
        update_manifest(
            experiment_dir,
            "data",
            {
                "config_path": str(data_config_path),
                "data_dir": str(data_dir),
            },
        )

    training_payload = {
        "config_path": str(config_path),
        "data_config_path": str(data_config_path) if data_config_path else None,
        "checkpoint_path": str(checkpoint_path.relative_to(experiment_dir)),
    }
    if manifest_metadata:
        training_payload.update(
            {k: v for k, v in manifest_metadata.items() if v is not None}
        )
    update_manifest(experiment_dir, "training", training_payload)

    return checkpoint_path


def train_pca_preconditioner(
    *,
    config_path: str | Path,
    n_components: int,
    data_config_path: str | Path | None = None,
    solution_samples_path: str | Path | None = None,
    output_path: str | Path | None = None,
    normalize: bool = True,
) -> tuple[Path, Path]:
    """Train PCA preconditioner on solution samples."""

    settings, context = load_config_with_context(config_path, data_config_path)
    paths = get_paths_from_config(settings, context)

    solution_samples = (
        Path(solution_samples_path)
        if solution_samples_path is not None
        else Path(paths.get("targets_path") or "")
    )
    if not solution_samples.exists():
        raise ValueError(
            "No targets path resolved from config; please pass solution_samples_path"
        )

    if output_path is None:
        base_output = Path(paths.get("results_dir", context.training.base_dir))
        output_path = base_output / "pca" / f"pca_{n_components}comp.pt"
    output_path = Path(output_path)

    pca, stats = fit_pca_from_solutions(solution_samples, n_components, normalize=normalize)
    save_pca_model(pca, stats, output_path)

    figures_dir = Path(paths.get("figures_dir", DEFAULT_FIGURES_DIR))
    figures_dir.mkdir(parents=True, exist_ok=True)
    dataset_value = paths.get("dataset_id") or "dataset"
    dataset_slug = sanitize_identifier(str(dataset_value))
    model_slug = sanitize_identifier(derive_model_identifier(settings, context, config_path))
    plot_filename = f"pca_variance_{dataset_slug}-{model_slug}_{n_components}comp.png"
    plot_path = figures_dir / plot_filename
    plot_variance_ratios(stats, plot_path)

    return output_path, plot_path
