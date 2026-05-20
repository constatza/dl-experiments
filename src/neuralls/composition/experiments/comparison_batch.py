"""Backend helpers for comparison workflows."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mlflow
from loguru import logger

from neuralls.platform.config.models.dataset_identity import resolve_dataset_identity
from neuralls.platform.config.models.comparison import ComparisonConfig
from neuralls.platform.config.models.experiments import (
    CaseConfig,
    ComparisonRegistryEntry,
    ExperimentEntry,
)
from neuralls.platform.config.registry import get_experiment_binding
from neuralls.composition.comparison.config_assembler import resolve_comparison_config
from neuralls.composition.experiments.assembler import load_validated_case_config
from neuralls.platform.config.resolution import build_mlflow_environment, build_sqlite_tracking_uri
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerConfig,
    PreconditionerType,
    RegisteredModelRefConfig,
)
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.storage.datasets import load_dataset_manifest, resolve_dataset_paths
from neuralls.platform.reporting.artifacts import (
    coerce_comparison_result_payload,
    write_comparison_artifacts,
)
from neuralls.platform.tracking.comparison_tracking import setup_comparison_tracking
from neuralls.platform.tracking.mlflow_client import log_comparison_artifacts_to_mlflow
from neuralls.composition.comparison.single_run import compare_preconditioners
from neuralls.composition.experiments.model_resolution import (
    ExperimentModelContext,
    resolve_preconditioner_models_with_warnings,
)
from neuralls.domain.solver.models.result import ComparisonResult
from neuralls.composition.tracking.run_specs import (
    build_comparison_run_spec,
    build_child_comparison_tags,
    ComparisonRunTags,
)
from neuralls.composition.comparison.models import ComparisonOutcome, ComparisonParams
from neuralls.shared.constants import (
    DATASET_MANIFEST_FILENAME,
    MATRIX_COO_DIRNAME,
    RHS_ARRAY_FILENAME,
)

_SUPPORTED_COMPARISON_FILE_SUFFIXES = {"", ".npy", ".txt"}


@dataclass(frozen=True)
class ComparisonTopology:
    """Shared MLflow topology for comparison execution."""

    tracking_uri: str
    artifact_location: str | None
    experiment_name: str
    model_store_tracking_uri: str


def _build_missing_input_error(path: Path) -> FileNotFoundError:
    """Build a user-facing missing-input error for a comparison dataset path."""
    return FileNotFoundError(f"Comparison input not found: {path}.")


def _validate_matrix_input(path: Path) -> None:
    """Validate one comparison matrix input without fully executing the comparison."""
    if not path.exists():
        raise _build_missing_input_error(path)
    if not path.is_dir():
        if path.suffix not in _SUPPORTED_COMPARISON_FILE_SUFFIXES:
            raise ValueError(
                f"Unsupported comparison matrix input format: {path}. "
                "Use a dataset directory, .npy file, or text matrix file."
            )
        return
    try:
        load_dataset_manifest(path)
    except FileNotFoundError, ValueError:
        manifest_path = path / DATASET_MANIFEST_FILENAME
        values_path = path / "values.npy"
        if not (manifest_path.exists() and values_path.exists()):
            raise ValueError(
                f"Comparison matrix dataset directory is not loadable: {path}. "
                f"Expected a dataset root with {DATASET_MANIFEST_FILENAME} or a sparse-pack "
                "matrix directory containing manifest.json and values.npy."
            ) from None
        return
    matrix_pack_dir = resolve_dataset_paths(path).matrix_pack_dir
    if not matrix_pack_dir.exists():
        raise ValueError(
            f"Comparison matrix dataset directory is missing {MATRIX_COO_DIRNAME}: {path}"
        )


def _validate_rhs_input(path: Path) -> None:
    """Validate one comparison RHS input without fully executing the comparison."""
    if not path.exists():
        raise _build_missing_input_error(path)
    if not path.is_dir():
        if path.suffix not in _SUPPORTED_COMPARISON_FILE_SUFFIXES:
            raise ValueError(
                f"Unsupported comparison RHS input format: {path}. "
                "Use a dataset directory, .npy file, or text vector file."
            )
        return
    load_dataset_manifest(path)
    rhs_path = resolve_dataset_paths(path).rhs_path
    if not rhs_path.exists():
        raise ValueError(
            f"Comparison RHS dataset directory is missing {RHS_ARRAY_FILENAME}: {path}"
        )


def _preflight_comparison_inputs(cfg: ComparisonConfig) -> None:
    """Validate comparison matrix/RHS inputs before opening tracking runs."""
    _validate_matrix_input(Path(cfg.general.data.matrix_path))
    _validate_rhs_input(Path(cfg.general.data.rhs_path))


def _validate_neural_preconditioner(spec: Any) -> None:
    """Validate strict neural preconditioner requirements for comparisons."""
    if not isinstance(spec, NeuralPreconditionerConfig):
        return
    if spec.model_ref is None:
        raise ValueError(f"Neural preconditioner '{spec.name}' must define model_ref.")
    if spec.checkpoint_path is not None:
        raise ValueError(
            f"Neural preconditioner '{spec.name}' cannot use checkpoint_path/experiment "
            "legacy resolution in comparison configs."
        )


def _resolve_preconditioner(
    spec: Any,
) -> Any:
    """Validate one preconditioner and any experiment-bound references."""
    _validate_neural_preconditioner(spec)
    return spec


def _resolve_neural_preconditioners(
    solver_specs: list[Any],
) -> list[PreconditionerConfig]:
    """Validate neural preconditioners and return unchanged specs."""
    return [_resolve_preconditioner(spec) for spec in solver_specs]


def _needs_model_resolution(specs: tuple[PreconditionerConfig, ...]) -> bool:
    """Return True when any neural preconditioner needs model_ref lookup."""
    for spec in specs:
        if not isinstance(spec, NeuralPreconditionerConfig):
            continue
        if spec.model_ref is not None:
            return True
    return False


def _referenced_experiment_ids(
    specs: tuple[PreconditionerConfig, ...],
) -> tuple[str, ...]:
    """Return experiment ids explicitly referenced by neural specs."""
    ids: list[str] = []
    for spec in specs:
        if not isinstance(spec, NeuralPreconditionerConfig):
            continue
        experiment_id = spec.experiment
        if experiment_id is None or experiment_id in ids:
            continue
        ids.append(experiment_id)
    return tuple(ids)


def _build_master_experiment_contexts(
    cfg_path: Path,
    experiment_ids: tuple[str, ...],
    settings: NeurallsSettings,
) -> dict[str, ExperimentModelContext]:
    """Resolve dataset/model identity for all case-config experiments."""
    master_cfg, config_dir = load_validated_case_config(cfg_path, settings)
    contexts: dict[str, ExperimentModelContext] = {}
    for experiment_id in experiment_ids:
        binding = get_experiment_binding(master_cfg, config_dir, experiment_id)
        data_cfg = load_data_config(binding.data_config_path, settings)
        dataset_id = resolve_dataset_identity(
            data_cfg=data_cfg,
            config_path=binding.data_config_path,
        ).name
        contexts[experiment_id] = ExperimentModelContext(
            dataset_alias=dataset_id,
            model_name=experiment_id,
        )
    return contexts


def _load_master_config(
    experiments_config_path: Path,
    settings: NeurallsSettings,
) -> tuple[CaseConfig, Path]:
    """Load the case config and its resolution root."""
    return load_validated_case_config(experiments_config_path, settings)


def _existing_experiment_ids(specs: tuple[PreconditionerConfig, ...]) -> set[str]:
    """Return experiment ids already claimed by explicit neural preconditioners."""
    return {
        spec.experiment
        for spec in specs
        if isinstance(spec, NeuralPreconditionerConfig) and spec.experiment is not None
    }


def neural_specs_from_experiments(
    entries: Sequence[ExperimentEntry],
    claimed_ids: set[str],
) -> list[NeuralPreconditionerConfig]:
    """Generate NeuralPreconditionerConfig stubs from experiment entries.

    Args:
        entries: Experiment entries to convert.
        claimed_ids: Experiment ids already covered by explicit preconditioners.

    Returns:
        List of auto-generated neural preconditioner configs.
    """
    return [
        NeuralPreconditionerConfig(
            name=entry.effective_display_name,
            type=PreconditionerType.NEURAL,
            experiment=entry.id,
            model_ref=RegisteredModelRefConfig(latest=True),
        )
        for entry in entries
        if entry.id not in claimed_ids
    ]


def _resolve_specs(
    cfg: ComparisonConfig,
    work_root: Path,
    experiments_config_path: Path,
    model_store_tracking_uri: str,
    settings: NeurallsSettings,
) -> tuple[list[PreconditionerConfig], tuple[str, ...]]:
    """Resolve model_ref preconditioners into concrete checkpoint paths."""
    specs = _resolve_neural_preconditioners(list(cfg.preconditioners))
    if not _needs_model_resolution(cfg.preconditioners):
        return specs, ()

    experiment_ids = _referenced_experiment_ids(cfg.preconditioners)
    experiment_contexts = None
    if experiment_ids:
        experiment_contexts = _build_master_experiment_contexts(
            experiments_config_path, experiment_ids, settings
        )
    resolution = resolve_preconditioner_models_with_warnings(
        specs=specs,
        tracking_uri=model_store_tracking_uri,
        download_root=work_root / "models",
        dataset_alias=cfg.general.data.dataset_alias,
        experiment_contexts=experiment_contexts,
        skip_unresolved=True,
    )
    return resolution.specs, resolution.warnings


def _resolve_comparison_topology(
    experiments_config_path: Path,
    settings: NeurallsSettings,
) -> ComparisonTopology:
    """Resolve comparison MLflow topology from the case config."""
    master_cfg, _ = _load_master_config(experiments_config_path, settings)
    if master_cfg.mlflow.tracking_uri is None:
        env = build_mlflow_environment(
            tracking_uri=build_sqlite_tracking_uri(settings.output_dir / "mlruns" / "mlflow.db"),
            artifacts_destination=str((settings.output_dir / "mlartifacts").resolve()),
        )
    else:
        env = build_mlflow_environment(
            tracking_uri=master_cfg.mlflow.tracking_uri,
            artifacts_destination=master_cfg.mlflow.artifacts_destination,
            config_path=experiments_config_path,
        )
    return ComparisonTopology(
        tracking_uri=env["MLFLOW_TRACKING_URI"],
        artifact_location=env.get("MLFLOW_ARTIFACT_URI"),
        experiment_name=master_cfg.names.comparison,
        model_store_tracking_uri=env["MLFLOW_TRACKING_URI"],
    )


def _log_child_residual_runs(
    result: ComparisonResult,
    comparison_id: str,
    parent_run_name: str,
) -> None:
    """Open a nested MLflow run per preconditioner and log its residual history.

    Args:
        result: Completed comparison result containing per-preconditioner histories.
        comparison_id: Registry id of the parent comparison entry.
        parent_run_name: Display name of the parent MLflow run.
    """
    for name, result_entry in result.results.items():
        child_tags = build_child_comparison_tags(
            preconditioner_name=name,
            comparison_id=comparison_id,
            parent_run_name=parent_run_name,
        )
        with mlflow.start_run(run_name=name, nested=True, tags=child_tags.as_mlflow_tags()):
            for step, residual in enumerate(result_entry.residual_history_rel):
                mlflow.log_metric("residual", residual, step=step)


def _log_comparison_params(
    comp_run_id: str,
    entry: ComparisonRegistryEntry,
) -> None:
    """Log final metadata params to the active comparison MLflow run.

    Args:
        comp_run_id: The run's own id, stored for cross-referencing.
        entry: Registry entry supplying id, display name, and method path.
    """
    config_label = entry.method.stem if entry.method is not None else entry.id
    mlflow.log_param("comparison_config", config_label)
    mlflow.log_param("comparison_id", entry.id)
    mlflow.log_param("comparison_display_name", entry.effective_display_name)
    mlflow.log_param("comp_run_id", comp_run_id)


def _run_comparison_body(
    cfg: ComparisonConfig,
    entry: ComparisonRegistryEntry,
    work_root: Path,
    topology: ComparisonTopology,
    experiments_config_path: Path,
    settings: NeurallsSettings,
) -> tuple[ComparisonResult, tuple[str, ...]]:
    """Resolve preconditioners, run comparison, and stage artifacts to disk.

    Args:
        cfg: Fully resolved comparison configuration.
        entry: Registry entry for display name and method config path.
        work_root: Temporary directory for staged outputs.
        topology: Resolved MLflow topology for model-store lookups.
        experiments_config_path: Case config path for model resolution context.
        settings: Runtime settings.

    Returns:
        Tuple of (comparison result, resolution warning strings).

    Raises:
        ValueError: When all preconditioners fail to resolve.
    """
    resolved_specs, warnings = _resolve_specs(
        cfg,
        work_root,
        experiments_config_path,
        topology.model_store_tracking_uri,
        settings,
    )
    if not resolved_specs:
        raise ValueError("No runnable preconditioners remain after model resolution.")
    raw_result = compare_preconditioners(
        general_params=cfg.general,
        preconditioner_configs=resolved_specs,
        output_root=work_root,
        display_name=entry.effective_display_name,
    )
    write_comparison_artifacts(
        result=coerce_comparison_result_payload(raw_result),
        work_root=work_root,
        comparison_config=entry.method,
    )
    return raw_result, warnings


def _execute_comparison_in_run(
    cfg: ComparisonConfig,
    entry: ComparisonRegistryEntry,
    topology: ComparisonTopology,
    run_name: str,
    comp_tags: ComparisonRunTags,
    experiments_config_path: Path,
    settings: NeurallsSettings,
) -> list[ComparisonOutcome]:
    """Open an MLflow run, execute the comparison, upload artifacts, and return outcomes.

    Args:
        cfg: Fully resolved comparison configuration.
        entry: Registry entry for display name, id, and method config path.
        topology: Resolved MLflow topology (tracking URI, artifact location).
        run_name: Display name for the MLflow run.
        comp_tags: Structured tags applied to the MLflow run.
        experiments_config_path: Case config path forwarded to body execution.
        settings: Runtime settings forwarded to body execution.

    Returns:
        Single-element list with a successful ComparisonOutcome.
    """
    with mlflow.start_run(run_name=run_name, tags=comp_tags.as_mlflow_tags()) as comp_run:
        comp_run_id = comp_run.info.run_id
        mlflow.log_param("artifact_uri", mlflow.get_artifact_uri())

        with tempfile.TemporaryDirectory() as _tmp:
            work_root = Path(_tmp)
            raw_result, warnings = _run_comparison_body(
                cfg=cfg,
                entry=entry,
                work_root=work_root,
                topology=topology,
                experiments_config_path=experiments_config_path,
                settings=settings,
            )
            if warnings:
                mlflow.log_param("skipped_preconditioners", str(len(warnings)))
            log_comparison_artifacts_to_mlflow(
                tracking_uri=topology.tracking_uri,
                run_id=comp_run_id,
                work_root=work_root,
            )
            if isinstance(raw_result, ComparisonResult):
                _log_child_residual_runs(raw_result, entry.id, run_name)

        _log_comparison_params(comp_run_id, entry)

    return [
        ComparisonOutcome(
            comparison_id=entry.id,
            comparison_display_name=entry.effective_display_name,
            success=True,
            payload=raw_result,
            warnings=warnings,
        )
    ]


def _run_comparison_from_config(
    cfg: ComparisonConfig,
    entry: ComparisonRegistryEntry,
    experiments_config_path: Path,
    settings: NeurallsSettings,
) -> list[ComparisonOutcome]:
    """Execute one resolved comparison and log results to MLflow.

    Args:
        cfg: Fully resolved ComparisonConfig with injected data paths and preconditioners.
        entry: Registry entry providing display name and method path for artifact logging.
        experiments_config_path: Path to the case config (used for topology resolution).
        settings: Resolved runtime settings.

    Returns:
        Single-element list with the outcome of the comparison run.
    """
    try:
        _preflight_comparison_inputs(cfg)
    except (FileNotFoundError, ValueError) as exc:
        return [
            ComparisonOutcome(
                comparison_id=entry.id,
                comparison_display_name=entry.effective_display_name,
                success=False,
                error=str(exc),
                warnings=(),
            )
        ]

    topology = _resolve_comparison_topology(experiments_config_path, settings)
    setup_comparison_tracking(
        tracking_uri=topology.tracking_uri,
        artifact_location=topology.artifact_location,
        experiment_name=topology.experiment_name,
    )
    run_name, comp_tags = build_comparison_run_spec(entry=entry)

    try:
        return _execute_comparison_in_run(
            cfg=cfg,
            entry=entry,
            topology=topology,
            run_name=run_name,
            comp_tags=comp_tags,
            experiments_config_path=experiments_config_path,
            settings=settings,
        )
    except (ValueError, RuntimeError, KeyError) as exc:
        logger.error(f"Comparison failed: {exc}")
        return [
            ComparisonOutcome(
                comparison_id=entry.id,
                comparison_display_name=entry.effective_display_name,
                success=False,
                error=str(exc),
                warnings=(),
            )
        ]


def run_comparison_batch(
    experiments_config_path: Path,
    params: ComparisonParams,
    settings: NeurallsSettings | None = None,
) -> list[ComparisonOutcome]:
    """Run all configured comparison entries from the case config.

    Args:
        experiments_config_path: Path to the case config TOML.
        params: Comparison execution parameters (currently unused, reserved for future use).
        settings: Optional pre-loaded runtime settings.

    Returns:
        List of comparison outcomes, one per [[comparisons]] entry.
    """
    _ = params
    settings = require_settings(settings, case_config_path=experiments_config_path)
    master_cfg, config_dir = _load_master_config(experiments_config_path, settings)
    if not master_cfg.comparisons:
        raise ValueError("Case config must define at least one [[comparisons]] entry.")

    outcomes: list[ComparisonOutcome] = []
    for entry in master_cfg.comparisons:
        experiment_entries: list[ExperimentEntry] = (
            [e for e in master_cfg.experiments if e.id in entry.experiments]
            if entry.experiments
            else list(master_cfg.experiments)
        )
        cfg = resolve_comparison_config(master_cfg, config_dir, entry, settings)
        claimed_ids = _existing_experiment_ids(cfg.preconditioners)
        auto_specs = neural_specs_from_experiments(experiment_entries, claimed_ids)
        if auto_specs:
            cfg = replace(cfg, preconditioners=cfg.preconditioners + tuple(auto_specs))
        outcomes.extend(_run_comparison_from_config(cfg, entry, experiments_config_path, settings))
    return outcomes
