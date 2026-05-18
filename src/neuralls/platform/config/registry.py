"""Case-config registry resolution helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from neuralls.platform.config.resolution import resolve_registry_path
from neuralls.platform.config.models.experiments import (
    CaseConfig,
    ComparisonDefaults,
    ComparisonRegistryEntry,
    ExperimentEntry,
    RegistryEntry,
)

if TYPE_CHECKING:
    from neuralls.platform.config.models.comparison import ComparisonConfig
    from neuralls.platform.config.models.preconditioner import PreconditionerConfig
    from neuralls.platform.config.settings import NeurallsSettings


@dataclass(frozen=True)
class ResolvedExperimentBinding:
    """Resolved experiment registry entry with concrete config paths."""

    experiment_id: str
    dataset_registry_id: str
    model_registry_id: str
    data_config_path: Path
    model_config_path: Path
    checkpoint_path: Path | None = None
    experiment_display_name: str = ""
    dataset_display_name: str = ""
    model_display_name: str = ""


def _lookup_registry_entry(
    entries: Sequence[RegistryEntry],
    registry_id: str,
) -> RegistryEntry | None:
    """Find one registry entry by id."""
    for entry in entries:
        if entry.id == registry_id:
            return entry
    return None


def _require_registry_entry(
    entries: Sequence[RegistryEntry],
    *,
    registry_id: str,
    registry_section: str,
    owner_kind: str | None = None,
    owner_id: str | None = None,
) -> RegistryEntry:
    """Return a registry entry or raise a focused validation error."""
    entry = _lookup_registry_entry(entries, registry_id)
    if entry is not None:
        return entry
    if owner_kind is not None and owner_id is not None:
        raise ValueError(
            f"{owner_kind} '{owner_id}' references {registry_section[:-1]} id "
            f"'{registry_id}', but [[{registry_section}]] does not define it."
        )
    raise ValueError(
        f"{registry_section[:-1].capitalize()} id '{registry_id}' is not defined "
        f"in [[{registry_section}]]."
    )


def _require_comparison_entry(
    entries: Sequence[ComparisonRegistryEntry],
    comparison_id: str,
) -> ComparisonRegistryEntry:
    """Return a comparison registry entry or raise a focused error."""
    for entry in entries:
        if entry.id == comparison_id:
            return entry
    raise ValueError(f"Comparison id '{comparison_id}' is not defined in [[comparisons]].")


def resolve_dataset_config_path(
    cfg: CaseConfig,
    config_dir: Path,
    dataset_id: str,
    *,
    experiment_id: str | None = None,
) -> Path:
    """Resolve a dataset config path from the case registry."""
    entry = _require_registry_entry(
        cfg.datasets,
        registry_id=dataset_id,
        registry_section="datasets",
        owner_kind="Experiment" if experiment_id is not None else None,
        owner_id=experiment_id,
    )
    return resolve_registry_path(config_dir, entry.path)


def resolve_model_config_path(
    cfg: CaseConfig,
    config_dir: Path,
    model_id: str,
    *,
    experiment_id: str | None = None,
) -> Path:
    """Resolve a model config path from the case registry."""
    entry = _require_registry_entry(
        cfg.models,
        registry_id=model_id,
        registry_section="models",
        owner_kind="Experiment" if experiment_id is not None else None,
        owner_id=experiment_id,
    )
    return resolve_registry_path(config_dir, entry.path)


def resolve_comparison_config(
    case_cfg: CaseConfig,
    config_dir: Path,
    entry: ComparisonRegistryEntry,
    settings: NeurallsSettings,
) -> ComparisonConfig:
    """Build a fully resolved ComparisonConfig from a case comparison entry.

    Merges ``[comparison_defaults]`` with an optional methodology override file,
    then injects the data paths resolved from the dataset registry.

    Args:
        case_cfg: Top-level case configuration.
        config_dir: Directory of the case config file (for resolving relative paths).
        entry: The specific ``[[comparisons]]`` entry to resolve.
        settings: ``NeurallsSettings`` for processed-dir expansion.

    Returns:
        Fully materialised ``ComparisonConfig`` with injected paths.

    Raises:
        ValueError: If required defaults or datasets are missing.
    """
    from neuralls.platform.config.loaders import load_data_config, load_raw_toml
    from neuralls.platform.config.context import ConfigContext
    from neuralls.platform.config.models.comparison import (
        parse_comparison_method_override,
        ComparisonConfig,
    )
    from neuralls.domain.solver.models.config import (
        ComparisonData,
        ComparisonGeneral,
        SolverParams,
    )

    defaults: ComparisonDefaults | None = case_cfg.comparison_defaults

    # --- Solver params: from defaults (required) ---
    if defaults is None:
        raise ValueError(
            "Case config must define [comparison_defaults] to use the [[comparisons]] registry."
        )

    # --- Optional method override ---
    preconditioners: list[PreconditionerConfig] = list(defaults.preconditioners)
    normalize_system = defaults.normalize_system
    params_override: dict[str, object] | None = None

    if entry.method is not None:
        method_path = resolve_registry_path(config_dir, entry.method)
        ctx = ConfigContext(config_path=method_path.resolve(), settings=settings)
        raw = load_raw_toml(method_path)
        override = parse_comparison_method_override(raw, ctx)
        if override.preconditioners:
            preconditioners = list(override.preconditioners)
        if "normalize_system" in override.general.data.model_fields_set:
            normalize_system = override.general.data.normalize_system
        if override.general.params is not None:
            params_override = override.general.params.model_dump()

    # --- Resolve paths from dataset registry ---
    matrix_dataset_entry = _require_registry_entry(
        case_cfg.datasets,
        registry_id=entry.matrix_dataset,
        registry_section="datasets",
    )
    rhs_dataset_entry = _require_registry_entry(
        case_cfg.datasets,
        registry_id=entry.rhs_dataset,
        registry_section="datasets",
    )
    matrix_data_cfg = load_data_config(
        resolve_registry_path(config_dir, matrix_dataset_entry.path), settings
    )
    rhs_data_cfg = load_data_config(
        resolve_registry_path(config_dir, rhs_dataset_entry.path), settings
    )
    matrix_path = (
        Path(matrix_data_cfg.output.data_dir) / matrix_data_cfg.id
        if matrix_data_cfg.output.data_dir is not None
        else settings.processed_dir / matrix_data_cfg.id
    )
    rhs_path = (
        Path(rhs_data_cfg.output.data_dir) / rhs_data_cfg.id
        if rhs_data_cfg.output.data_dir is not None
        else settings.processed_dir / rhs_data_cfg.id
    )

    # --- Build SolverParams ---
    base_params = {
        "rtol": defaults.rtol,
        "atol": defaults.atol,
        "max_iterations": defaults.max_iterations,
        "stopping_criterion": defaults.stopping_criterion,
        "m_max": defaults.m_max,
        "breakdown_tol": defaults.breakdown_tol,
    }
    if params_override:
        base_params.update(params_override)

    return ComparisonConfig(
        general=ComparisonGeneral(
            params=SolverParams(**base_params),
            data=ComparisonData(
                matrix_path=matrix_path,
                rhs_path=rhs_path,
                rhs_index=entry.rhs_index,
                matrix_index=entry.matrix_index,
                normalize_system=normalize_system,
            ),
        ),
        preconditioners=tuple(preconditioners),
    )


def resolve_experiment_binding(
    cfg: CaseConfig,
    config_dir: Path,
    entry: ExperimentEntry,
) -> ResolvedExperimentBinding:
    """Resolve a single experiment entry into concrete config paths."""
    dataset_entry = _require_registry_entry(
        cfg.datasets,
        registry_id=entry.dataset_id,
        registry_section="datasets",
        owner_kind="Experiment",
        owner_id=entry.id,
    )
    model_entry = _require_registry_entry(
        cfg.models,
        registry_id=entry.model_id,
        registry_section="models",
        owner_kind="Experiment",
        owner_id=entry.id,
    )
    return ResolvedExperimentBinding(
        experiment_id=entry.id,
        dataset_registry_id=entry.dataset_id,
        model_registry_id=entry.model_id,
        data_config_path=resolve_dataset_config_path(
            cfg,
            config_dir,
            entry.dataset_id,
            experiment_id=entry.id,
        ),
        model_config_path=resolve_model_config_path(
            cfg,
            config_dir,
            entry.model_id,
            experiment_id=entry.id,
        ),
        checkpoint_path=(
            resolve_registry_path(config_dir, entry.checkpoint_path)
            if entry.checkpoint_path is not None
            else None
        ),
        experiment_display_name=entry.effective_display_name,
        dataset_display_name=dataset_entry.effective_display_name,
        model_display_name=model_entry.effective_display_name,
    )


def get_experiment_binding(
    cfg: CaseConfig,
    config_dir: Path,
    experiment_id: str,
) -> ResolvedExperimentBinding:
    """Resolve one experiment by id."""
    for entry in cfg.experiments:
        if entry.id == experiment_id:
            return resolve_experiment_binding(cfg, config_dir, entry)
    raise KeyError(f"Experiment '{experiment_id}' not found in case config registry.")


def list_experiment_bindings(
    cfg: CaseConfig,
    config_dir: Path,
) -> list[ResolvedExperimentBinding]:
    """Resolve all experiment entries with concrete config paths."""
    return [resolve_experiment_binding(cfg, config_dir, entry) for entry in cfg.experiments]
