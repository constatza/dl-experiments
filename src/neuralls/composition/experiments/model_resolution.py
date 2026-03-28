"""Resolve strict model references into concrete checkpoint paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from neuralls.platform.config.models.dataset_identity import normalize_registry_id
from dlkit.interfaces.api.functions.model_logged import (
    build_logged_model_uri,
    search_logged_models,
)
from dlkit.interfaces.api.functions.model_registry import (
    build_registered_model_uri,
    get_model_version,
    list_model_versions,
    search_registered_models,
)
from loguru import logger
from mlflow.tracking import MlflowClient

from neuralls.platform.config.models.preconditioner import (
    LoggedModelRefConfig,
    NeuralPreconditionerConfig,
    PreconditionerConfig,
    RegisteredModelRefConfig,
)

_DATASET_ALIAS_PLACEHOLDER = "@dataset"


@dataclass(frozen=True)
class ExperimentModelContext:
    """Per-experiment lookup context for comparison model resolution."""

    dataset_alias: str | None = None
    model_name: str | None = None


@dataclass(frozen=True)
class ModelResolution:
    """Immutable model resolution result."""

    model_uri: str
    run_id: str
    checkpoint_path: Path


@dataclass(frozen=True)
class PreconditionerResolutionResult:
    """Resolved specs plus any skipped-resolution warnings."""

    specs: list[PreconditionerConfig]
    warnings: tuple[str, ...] = ()


def _find_single_checkpoint(root: Path) -> Path:
    """Find a checkpoint under a local artifact directory."""
    checkpoints = sorted(root.glob("**/*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found under {root}")
    if len(checkpoints) > 1:
        logger.warning(
            "Multiple checkpoints found under {}. Using {}.",
            root,
            checkpoints[0],
        )
    return checkpoints[0]


def _download_checkpoint_for_run(
    *,
    client: MlflowClient,
    run_id: str,
    destination: Path,
    fallback_artifact_path: str,
) -> Path:
    """Download run artifacts and return a concrete checkpoint path."""
    primary_root: Path | None = None
    try:
        primary_root = Path(
            client.download_artifacts(
                run_id=run_id,
                path="checkpoints",
                dst_path=str(destination),
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "Could not download 'checkpoints' artifacts for run {}: {}",
            run_id,
            exc,
        )
    if primary_root is not None:
        try:
            return _find_single_checkpoint(primary_root)
        except FileNotFoundError:
            logger.debug(
                "No checkpoint under 'checkpoints' for run {}. Falling back to '{}'.",
                run_id,
                fallback_artifact_path,
            )

    try:
        fallback_root = Path(
            client.download_artifacts(
                run_id=run_id,
                path=fallback_artifact_path,
                dst_path=str(destination),
            )
        )
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"Could not download checkpoint artifacts for run '{run_id}' "
            f"from 'checkpoints' or '{fallback_artifact_path}'."
        ) from exc
    return _find_single_checkpoint(fallback_root)


def _resolve_registered_ref(
    *,
    ref: RegisteredModelRefConfig,
    tracking_uri: str,
    client: MlflowClient,
    destination: Path,
    dataset_alias: str | None,
    model_name: str | None,
) -> ModelResolution:
    """Resolve a registered model reference."""
    resolved_model_name = ref.name or model_name
    if resolved_model_name is None:
        raise ValueError(
            "Registered model_ref.name is required unless supplied by an experiment binding."
        )
    matches = search_registered_models(model_name=resolved_model_name, tracking_uri=tracking_uri)
    if not matches:
        raise ValueError(f"Registered model '{resolved_model_name}' not found")

    if ref.alias is not None:
        resolved_alias = _resolve_registered_alias(ref.alias, dataset_alias=dataset_alias)
        model_uri = build_registered_model_uri(resolved_model_name, alias=resolved_alias)
        version = client.get_model_version_by_alias(resolved_model_name, resolved_alias)
        run_id = version.run_id
    else:
        if ref.version is not None:
            resolved_version = ref.version
        else:
            versions = list_model_versions(resolved_model_name, tracking_uri=tracking_uri)
            if not versions:
                raise ValueError(f"Registered model '{resolved_model_name}' has no versions")
            resolved_version = max(versions)
        version = get_model_version(
            model_name=resolved_model_name,
            version=resolved_version,
            tracking_uri=tracking_uri,
        )
        model_uri = build_registered_model_uri(resolved_model_name, version=resolved_version)
        run_id = version.run_id
        if run_id is None:
            raise ValueError(
                f"Registered model '{resolved_model_name}' version {resolved_version} has no run_id."
            )
    if run_id is None:
        raise ValueError(
            f"Registered model '{resolved_model_name}' could not be resolved to an MLflow run."
        )

    checkpoint_path = _download_checkpoint_for_run(
        client=client,
        run_id=run_id,
        destination=destination,
        fallback_artifact_path="model",
    )
    return ModelResolution(
        model_uri=model_uri,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
    )


def _resolve_registered_alias(alias: str, dataset_alias: str | None) -> str:
    """Resolve explicit alias and @dataset placeholder to canonical alias."""
    stripped = alias.strip()
    if stripped == _DATASET_ALIAS_PLACEHOLDER:
        if dataset_alias is None:
            raise ValueError(
                "model_ref alias '@dataset' requires general.data.dataset_alias "
                "or a neural experiment binding."
            )
        return normalize_registry_id(dataset_alias)
    return normalize_registry_id(stripped)


def _resolve_logged_ref(
    *,
    ref: LoggedModelRefConfig,
    tracking_uri: str,
    client: MlflowClient,
    destination: Path,
) -> ModelResolution:
    """Resolve a logged-model reference."""
    if ref.run_id is not None:
        run_id = ref.run_id
        model_uri = build_logged_model_uri(run_id=run_id, artifact_path=ref.artifact_path)
    else:
        records = search_logged_models(
            model_name=ref.model_name,
            experiment_name=ref.experiment_name,
            experiment_id=ref.experiment_id,
            run_name=ref.run_name,
            tracking_uri=tracking_uri,
            artifact_path=ref.artifact_path,
            tags=ref.tags,
            max_results=100,
        )
        if not records:
            raise ValueError("No logged model matches the provided model_ref filters")
        latest = records[0]
        run_id = latest.run_id
        model_uri = latest.model_uri or build_logged_model_uri(
            run_id=run_id,
            artifact_path=ref.artifact_path,
        )

    checkpoint_path = _download_checkpoint_for_run(
        client=client,
        run_id=run_id,
        destination=destination,
        fallback_artifact_path=ref.artifact_path,
    )
    return ModelResolution(
        model_uri=model_uri,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
    )


def resolve_model_ref(
    *,
    spec: NeuralPreconditionerConfig,
    tracking_uri: str,
    destination: Path,
    dataset_alias: str | None = None,
    model_name: str | None = None,
) -> ModelResolution:
    """Resolve one neural preconditioner model reference."""
    ref = spec.model_ref
    client = MlflowClient(tracking_uri=tracking_uri)
    destination.mkdir(parents=True, exist_ok=True)

    if isinstance(ref, RegisteredModelRefConfig):
        return _resolve_registered_ref(
            ref=ref,
            tracking_uri=tracking_uri,
            client=client,
            destination=destination,
            dataset_alias=dataset_alias,
            model_name=model_name,
        )
    if isinstance(ref, LoggedModelRefConfig):
        return _resolve_logged_ref(
            ref=ref,
            tracking_uri=tracking_uri,
            client=client,
            destination=destination,
        )
    raise TypeError(f"Unsupported model_ref type: {type(ref)}")


def resolve_preconditioner_models(
    *,
    specs: list[PreconditionerConfig],
    tracking_uri: str,
    download_root: Path,
    dataset_alias: str | None = None,
    experiment_contexts: dict[str, ExperimentModelContext] | None = None,
) -> list[PreconditionerConfig]:
    """Resolve all neural preconditioners to concrete checkpoint paths."""
    return resolve_preconditioner_models_with_warnings(
        specs=specs,
        tracking_uri=tracking_uri,
        download_root=download_root,
        dataset_alias=dataset_alias,
        experiment_contexts=experiment_contexts,
    ).specs


def resolve_preconditioner_models_with_warnings(
    *,
    specs: list[PreconditionerConfig],
    tracking_uri: str,
    download_root: Path,
    dataset_alias: str | None = None,
    experiment_contexts: dict[str, ExperimentModelContext] | None = None,
    skip_unresolved: bool = False,
) -> PreconditionerResolutionResult:
    """Resolve neural preconditioners and optionally skip unresolved ones."""
    resolved: list[PreconditionerConfig] = []
    warnings: list[str] = []
    for spec in specs:
        if spec.type != "neural":
            resolved.append(spec)
            continue
        neural_spec = cast(NeuralPreconditionerConfig, spec)
        if neural_spec.checkpoint_path is not None:
            resolved.append(
                neural_spec.model_copy(
                    update={"resolved_checkpoint_path": neural_spec.checkpoint_path}
                )
            )
            continue
        if neural_spec.model_ref is None:
            raise ValueError(
                f"Neural solver '{neural_spec.name}' requires either checkpoint_path or model_ref."
            )
        context = (
            experiment_contexts.get(neural_spec.experiment)
            if experiment_contexts is not None and neural_spec.experiment is not None
            else None
        )
        try:
            resolution = resolve_model_ref(
                spec=neural_spec,
                tracking_uri=tracking_uri,
                destination=download_root / neural_spec.name.replace(" ", "_"),
                dataset_alias=context.dataset_alias if context is not None else dataset_alias,
                model_name=context.model_name if context is not None else None,
            )
        except (ValueError, FileNotFoundError, RuntimeError, OSError, KeyError) as exc:
            if not skip_unresolved:
                raise
            warning = f"Skipping neural preconditioner '{neural_spec.name}': {exc}"
            logger.warning(warning)
            warnings.append(warning)
            continue
        checkpoint_path = resolution.checkpoint_path
        resolved.append(
            neural_spec.model_copy(
                update={
                    "checkpoint_path": checkpoint_path,
                    "resolved_checkpoint_path": checkpoint_path,
                }
            )
        )
    return PreconditionerResolutionResult(
        specs=resolved,
        warnings=tuple(warnings),
    )
