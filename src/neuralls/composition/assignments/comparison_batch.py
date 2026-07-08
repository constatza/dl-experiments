"""Backend helpers for comparison workflows."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

from neuralls.platform.config.models.dataset_identity import resolve_dataset_identity
from neuralls.platform.config.models.comparison import ComparisonConfig
from neuralls.platform.config.models.experiments import (
    CaseConfig,
    ComparisonRegistryEntry,
    AssignmentEntry,
)
from neuralls.platform.config.registry import get_assignment_binding
from neuralls.composition.comparison.config_assembler import resolve_comparison_config
from neuralls.composition.assignments.assembler import load_validated_case_config
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerConfig,
    PreconditionerType,
    RegisteredModelRefConfig,
)
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.reporting.artifacts import (
    coerce_comparison_result_payload,
    write_comparison_artifacts,
)
from neuralls.platform.reporting.comparison_inputs import (
    ComparisonInputArtifacts,
    stage_comparison_inputs,
)
from neuralls.platform.storage.validation import (
    validate_comparison_inputs,
    validate_comparison_matrix_input,
    validate_comparison_rhs_input,
)
from neuralls.platform.config.models.comparison import (
    DatasetRhsSourceModel,
    RawLhsSourceModel,
    RawRhsSourceModel,
)
from neuralls.shared.types import ComparisonRhsSourceKind
from neuralls.platform.tracking.comparison_tracking import (
    log_comparison_artifact_uri,
    log_comparison_input_artifacts,
    log_comparison_result_metrics,
    log_comparison_run_params,
    log_skipped_preconditioners,
    setup_comparison_tracking,
)
from neuralls.platform.tracking.extra_features import fetch_extra_input_names_for_model
from neuralls.platform.tracking.mlflow import build_workflow_environment
from neuralls.platform.tracking.mlflow_client import log_comparison_artifacts_to_mlflow
from neuralls.composition.comparison.single_run import compare_preconditioners
from neuralls.composition.comparison._input_resolution import resolve_comparison_input
from neuralls.composition.assignments.model_resolution import (
    AssignmentModelContext,
    resolve_preconditioner_models_with_warnings,
)
from neuralls.platform.tracking.model_registry import build_registered_model_name
from neuralls.domain.solver.models.result import ComparisonResult
from neuralls.composition.tracking.run_specs import (
    build_comparison_run_spec,
    build_child_comparison_tags,
    build_session_run_spec,
    ComparisonRunTags,
)
from neuralls.composition.comparison.models import ComparisonOutcome, ComparisonParams


@dataclass(frozen=True)
class ComparisonTopology:
    """Shared MLflow topology for comparison execution."""

    tracking_uri: str
    artifact_location: str | None
    experiment_name: str
    model_store_tracking_uri: str


def _validate_neural_preconditioner(spec: Any) -> None:
    """Validate strict neural preconditioner requirements for comparisons."""
    if not isinstance(spec, NeuralPreconditionerConfig):
        return
    if spec.model_ref is None:
        raise ValueError(f"Neural preconditioner '{spec.name}' must define model_ref.")
    if spec.checkpoint_path is not None:
        raise ValueError(
            f"Neural preconditioner '{spec.name}' cannot use checkpoint_path/assignment "
            "legacy resolution in comparison configs."
        )


def _resolve_preconditioner(
    spec: Any,
) -> Any:
    """Validate one preconditioner and any assignment-bound references."""
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


def _referenced_assignment_ids(
    specs: tuple[PreconditionerConfig, ...],
) -> tuple[str, ...]:
    """Return assignment ids explicitly referenced by neural specs."""
    ids: list[str] = []
    for spec in specs:
        if not isinstance(spec, NeuralPreconditionerConfig):
            continue
        assignment_id = spec.assignment
        if assignment_id is None or assignment_id in ids:
            continue
        ids.append(assignment_id)
    return tuple(ids)


def _build_master_assignment_contexts(
    cfg_path: Path,
    assignment_ids: tuple[str, ...],
    settings: NeurallsSettings,
) -> dict[str, AssignmentModelContext]:
    """Resolve dataset/model identity for all case-config assignments."""
    master_cfg, config_dir = load_validated_case_config(cfg_path, settings)
    contexts: dict[str, AssignmentModelContext] = {}
    for assignment_id in assignment_ids:
        binding = get_assignment_binding(master_cfg, config_dir, assignment_id)
        data_cfg = load_data_config(binding.data_config_path, settings)
        dataset_id = resolve_dataset_identity(
            data_cfg=data_cfg,
            config_path=binding.data_config_path,
        ).name
        contexts[assignment_id] = AssignmentModelContext(
            dataset_alias=dataset_id,
            model_name=build_registered_model_name(assignment_id),
        )
    return contexts


def _load_master_config(
    case_config_path: Path,
    settings: NeurallsSettings,
) -> tuple[CaseConfig, Path]:
    """Load the case config and its resolution root."""
    return load_validated_case_config(case_config_path, settings)


def _existing_assignment_ids(specs: tuple[PreconditionerConfig, ...]) -> set[str]:
    """Return assignment ids already claimed by explicit neural preconditioners."""
    return {
        spec.assignment
        for spec in specs
        if isinstance(spec, NeuralPreconditionerConfig) and spec.assignment is not None
    }


def neural_specs_from_assignments(
    entries: Sequence[AssignmentEntry],
    claimed_ids: set[str],
    *,
    client: MlflowClient,
) -> list[NeuralPreconditionerConfig]:
    """Generate NeuralPreconditionerConfig stubs from assignment entries.

    For each entry, fetches the ``neuralls.extra_feature_names`` tag from the
    training run so that FiLM models receive their condition tensor during
    comparison.

    Args:
        entries: Assignment entries to convert.
        claimed_ids: Assignment ids already covered by explicit preconditioners.
        client: Configured MLflow client used to look up training run tags.

    Returns:
        List of auto-generated neural preconditioner configs with extra_input_names
        populated from the training run tag where available.
    """
    return [
        NeuralPreconditionerConfig(
            name=entry.effective_display_name,
            type=PreconditionerType.NEURAL,
            assignment=entry.id,
            model_ref=RegisteredModelRefConfig(latest=True),
            extra_input_names=fetch_extra_input_names_for_model(entry.id, client),
        )
        for entry in entries
        if entry.id not in claimed_ids
    ]


def _resolve_specs(
    cfg: ComparisonConfig,
    work_root: Path,
    case_config_path: Path,
    model_store_tracking_uri: str,
    settings: NeurallsSettings,
) -> tuple[list[PreconditionerConfig], tuple[str, ...]]:
    """Resolve model_ref preconditioners into concrete checkpoint paths."""
    specs = _resolve_neural_preconditioners(list(cfg.preconditioners))
    if not _needs_model_resolution(cfg.preconditioners):
        return specs, ()

    assignment_ids = _referenced_assignment_ids(cfg.preconditioners)
    assignment_contexts = None
    if assignment_ids:
        assignment_contexts = _build_master_assignment_contexts(
            case_config_path, assignment_ids, settings
        )
    resolution = resolve_preconditioner_models_with_warnings(
        specs=specs,
        tracking_uri=model_store_tracking_uri,
        download_root=work_root / "models",
        dataset_alias=cfg.general.data.dataset_alias,
        assignment_contexts=assignment_contexts,
        skip_unresolved=True,
    )
    return resolution.specs, resolution.warnings


def _resolve_comparison_topology(
    case_config_path: Path,
    settings: NeurallsSettings,
) -> ComparisonTopology:
    """Resolve comparison MLflow topology from the case config."""
    master_cfg, _ = _load_master_config(case_config_path, settings)
    runtime = build_workflow_environment(
        tracking_uri=master_cfg.mlflow.tracking_uri,
        artifact_location=master_cfg.mlflow.artifacts_destination,
        default_output_root=settings.output_dir,
        config_path=case_config_path,
    )
    return ComparisonTopology(
        tracking_uri=runtime.tracking_uri,
        artifact_location=runtime.artifact_uri,
        experiment_name=master_cfg.names.comparison,
        model_store_tracking_uri=runtime.tracking_uri,
    )


def _require_rhs_source_kind(cfg: ComparisonConfig) -> ComparisonRhsSourceKind:
    """Return the configured RHS source kind or fail with a focused error."""
    kind = cfg.general.data.rhs_source_kind
    if kind is None:
        raise ValueError("Comparison config must define rhs_source.")
    return kind


def _run_comparison_body(
    cfg: ComparisonConfig,
    entry: ComparisonRegistryEntry,
    work_root: Path,
    topology: ComparisonTopology,
    case_config_path: Path,
    settings: NeurallsSettings,
) -> tuple[ComparisonResult, tuple[str, ...]]:
    """Resolve preconditioners, run comparison, and stage artifacts to disk.

    Args:
        cfg: Fully resolved comparison configuration.
        entry: Registry entry for display name and method config path.
        work_root: Temporary directory for staged outputs.
        topology: Resolved MLflow topology for model-store lookups.
        case_config_path: Case config path for model resolution context.
        settings: Runtime settings.

    Returns:
        Tuple of (comparison result, resolution warning strings).

    Raises:
        ValueError: When all preconditioners fail to resolve.
    """
    resolved_specs, warnings = _resolve_specs(
        cfg,
        work_root,
        case_config_path,
        topology.model_store_tracking_uri,
        settings,
    )
    if not resolved_specs:
        raise ValueError("No runnable preconditioners remain after model resolution.")
    rhs_source_kind = _require_rhs_source_kind(cfg)
    resolved_input = resolve_comparison_input(
        matrix_path=Path(cfg.general.data.matrix_path),
        matrix_dataset_id=entry.matrix_dataset,
        matrix_index=cfg.general.data.matrix_index,
        require_non_residual_rhs=cfg.general.data.require_non_residual_rhs,
        seed=cfg.general.data.selection_seed,
        rhs_source_kind=rhs_source_kind,
        rhs_source_params=cfg.general.data.rhs_source_params,
    )
    staged_input_dir = stage_comparison_inputs(
        work_root,
        ComparisonInputArtifacts(
            matrix=resolved_input.matrix,
            rhs=resolved_input.rhs,
            matrix_dataset_id=resolved_input.matrix_dataset_id,
            matrix_index=resolved_input.matrix_index,
            rhs_source_type=resolved_input.rhs_source_type,
            lhs=resolved_input.lhs,
            rhs_dataset_id=resolved_input.rhs_dataset_id,
            rhs_sample_index=resolved_input.rhs_sample_index,
            rhs_kind=resolved_input.rhs_kind,
            rhs_source_kind=resolved_input.rhs_source_kind,
            rhs_source_params=resolved_input.rhs_source_params,
        ),
    )
    raw_result = compare_preconditioners(
        general_params=cfg.general,
        preconditioner_configs=resolved_specs,
        output_root=work_root,
        display_name=entry.effective_display_name,
        resolved_input=resolved_input,
    )
    write_comparison_artifacts(
        result=coerce_comparison_result_payload(raw_result),
        work_root=work_root,
        comparison_config=entry.method,
    )
    log_comparison_input_artifacts(staged_input_dir)
    return raw_result, warnings


def _execute_comparison_in_run(
    cfg: ComparisonConfig,
    entry: ComparisonRegistryEntry,
    topology: ComparisonTopology,
    run_name: str,
    comp_tags: ComparisonRunTags,
    case_config_path: Path,
    settings: NeurallsSettings,
) -> list[ComparisonOutcome]:
    """Open an MLflow run, execute the comparison, upload artifacts, and return outcomes.

    Args:
        cfg: Fully resolved comparison configuration.
        entry: Registry entry for display name, id, and method config path.
        topology: Resolved MLflow topology (tracking URI, artifact location).
        run_name: Display name for the MLflow run.
        comp_tags: Structured tags applied to the MLflow run.
        case_config_path: Case config path forwarded to body execution.
        settings: Runtime settings forwarded to body execution.

    Returns:
        Single-element list with a successful ComparisonOutcome.
    """
    with mlflow.start_run(
        run_name=run_name, nested=True, tags=comp_tags.as_mlflow_tags()
    ) as comp_run:
        comp_run_id = comp_run.info.run_id
        log_comparison_artifact_uri()

        with tempfile.TemporaryDirectory() as _tmp:
            work_root = Path(_tmp)
            raw_result, warnings = _run_comparison_body(
                cfg=cfg,
                entry=entry,
                work_root=work_root,
                topology=topology,
                case_config_path=case_config_path,
                settings=settings,
            )
            log_skipped_preconditioners(warnings)
            log_comparison_artifacts_to_mlflow(
                tracking_uri=topology.tracking_uri,
                run_id=comp_run_id,
                work_root=work_root,
            )
            if isinstance(raw_result, ComparisonResult):
                child_run_tags = {
                    name: build_child_comparison_tags(
                        preconditioner_name=name,
                        comparison_id=entry.id,
                        parent_run_name=run_name,
                    ).as_mlflow_tags()
                    for name in raw_result.results
                }
                log_comparison_result_metrics(raw_result, child_run_tags=child_run_tags)

        log_comparison_run_params(
            comp_run_id=comp_run_id,
            comparison_id=entry.id,
            comparison_display_name=entry.effective_display_name,
            comparison_config=entry.method.stem if entry.method is not None else entry.id,
        )

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
    topology: ComparisonTopology,
    case_config_path: Path,
    settings: NeurallsSettings,
) -> list[ComparisonOutcome]:
    """Execute one resolved comparison and log results to MLflow.

    Runs nested under the batch's session parent run (already active), so this
    becomes the "subrun" level between the case-wide parent and per-preconditioner
    leaf runs.

    Args:
        cfg: Fully resolved ComparisonConfig with injected data paths and preconditioners.
        entry: Registry entry providing display name and method path for artifact logging.
        topology: Resolved MLflow topology for the whole batch (tracking/experiment).
        case_config_path: Case config path forwarded to body execution.
        settings: Resolved runtime settings.

    Returns:
        Single-element list with the outcome of the comparison run.
    """
    try:
        _validate_comparison_sources(cfg)
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

    run_name, comp_tags = build_comparison_run_spec(entry=entry, include_timestamp=False)

    try:
        return _execute_comparison_in_run(
            cfg=cfg,
            entry=entry,
            topology=topology,
            run_name=run_name,
            comp_tags=comp_tags,
            case_config_path=case_config_path,
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


def _validate_comparison_sources(cfg: ComparisonConfig) -> None:
    """Validate only the concrete input artifacts required by this workflow mode."""
    matrix_path = Path(cfg.general.data.matrix_path)
    rhs_source_kind = _require_rhs_source_kind(cfg)
    rhs_source_params = cfg.general.data.rhs_source_params or {}
    match rhs_source_kind:
        case ComparisonRhsSourceKind.GAUSSIAN | ComparisonRhsSourceKind.SPARSE:
            validate_comparison_matrix_input(matrix_path)
        case ComparisonRhsSourceKind.DATASET:
            source = DatasetRhsSourceModel.model_validate(
                {"kind": rhs_source_kind, **rhs_source_params}
            )
            validate_comparison_rhs_input(source.path)
        case ComparisonRhsSourceKind.RAW_LHS:
            source = RawLhsSourceModel.model_validate(
                {"kind": rhs_source_kind, **rhs_source_params}
            )
            validate_comparison_inputs(matrix_path=matrix_path, rhs_path=source.path)
        case ComparisonRhsSourceKind.RAW_RHS:
            source = RawRhsSourceModel.model_validate(
                {"kind": rhs_source_kind, **rhs_source_params}
            )
            validate_comparison_inputs(matrix_path=matrix_path, rhs_path=source.path)
        case _:
            raise ValueError(f"Unsupported comparison RHS source: {rhs_source_kind!r}.")


def run_comparison_batch(
    case_config_path: Path,
    params: ComparisonParams,
    settings: NeurallsSettings | None = None,
) -> list[ComparisonOutcome]:
    """Run all configured comparison entries from the case config.

    Opens one MLflow session parent run for the whole batch — named
    ``"<case-name> | <timestamp>"``, mirroring training's session parent — so
    every comparison entry's own run nests underneath it as a "subrun", with
    each entry's per-preconditioner runs nested one level further.

    Args:
        case_config_path: Path to the case config TOML.
        params: Comparison execution parameters (currently unused, reserved for future use).
        settings: Optional pre-loaded runtime settings.

    Returns:
        List of comparison outcomes, one per [[comparisons]] entry.
    """
    _ = params
    settings = require_settings(settings, case_config_path=case_config_path)
    master_cfg, config_dir = _load_master_config(case_config_path, settings)
    if not master_cfg.comparisons:
        raise ValueError("Case config must define at least one [[comparisons]] entry.")

    topology = _resolve_comparison_topology(case_config_path, settings)
    mlflow_client = MlflowClient(tracking_uri=topology.model_store_tracking_uri)
    setup_comparison_tracking(
        tracking_uri=topology.tracking_uri,
        artifact_location=topology.artifact_location,
        experiment_name=topology.experiment_name,
    )
    session_run_name, session_tags = build_session_run_spec(
        case_config_path=case_config_path.resolve(),
        experiment_name=topology.experiment_name,
        phase="session_comparison",
    )

    outcomes: list[ComparisonOutcome] = []
    with mlflow.start_run(run_name=session_run_name, tags=session_tags.as_mlflow_tags()):
        for entry in master_cfg.comparisons:
            assignment_entries: list[AssignmentEntry] = (
                [e for e in master_cfg.assignments if e.id in entry.assignments]
                if entry.assignments
                else list(master_cfg.assignments)
            )
            cfg = resolve_comparison_config(master_cfg, config_dir, entry, settings)
            claimed_ids = _existing_assignment_ids(cfg.preconditioners)
            auto_specs = neural_specs_from_assignments(
                assignment_entries, claimed_ids, client=mlflow_client
            )
            if auto_specs:
                cfg = replace(cfg, preconditioners=cfg.preconditioners + tuple(auto_specs))
            outcomes.extend(
                _run_comparison_from_config(cfg, entry, topology, case_config_path, settings)
            )
    return outcomes
