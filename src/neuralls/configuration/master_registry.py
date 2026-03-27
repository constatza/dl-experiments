"""Master registry resolution helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from neuralls.configuration.experiments import (
    ExperimentEntry,
    ExperimentsConfig,
    RegistryEntry,
)


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


def _resolve_registry_path(base_dir: Path, relative_or_absolute: Path) -> Path:
    """Resolve one registry path against the master config directory."""
    if relative_or_absolute.is_absolute():
        return relative_or_absolute.resolve()
    return (base_dir / relative_or_absolute).resolve()


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


def resolve_dataset_config_path(
    cfg: ExperimentsConfig,
    config_dir: Path,
    dataset_id: str,
    *,
    experiment_id: str | None = None,
) -> Path:
    """Resolve a dataset config path from the master registry."""
    entry = _require_registry_entry(
        cfg.datasets,
        registry_id=dataset_id,
        registry_section="datasets",
        owner_kind="Experiment" if experiment_id is not None else None,
        owner_id=experiment_id,
    )
    return _resolve_registry_path(config_dir, entry.path)


def resolve_model_config_path(
    cfg: ExperimentsConfig,
    config_dir: Path,
    model_id: str,
    *,
    experiment_id: str | None = None,
) -> Path:
    """Resolve a model config path from the master registry."""
    entry = _require_registry_entry(
        cfg.models,
        registry_id=model_id,
        registry_section="models",
        owner_kind="Experiment" if experiment_id is not None else None,
        owner_id=experiment_id,
    )
    return _resolve_registry_path(config_dir, entry.path)


def resolve_comparison_config_path(
    cfg: ExperimentsConfig,
    config_dir: Path,
    comparison_id: str,
) -> Path:
    """Resolve a comparison config path from the master registry."""
    entry = _require_registry_entry(
        cfg.comparisons,
        registry_id=comparison_id,
        registry_section="comparisons",
    )
    return _resolve_registry_path(config_dir, entry.path)


def resolve_experiment_binding(
    cfg: ExperimentsConfig,
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
            _resolve_registry_path(config_dir, entry.checkpoint_path)
            if entry.checkpoint_path is not None
            else None
        ),
        experiment_display_name=entry.effective_display_name,
        dataset_display_name=dataset_entry.effective_display_name,
        model_display_name=model_entry.effective_display_name,
    )


def get_experiment_binding(
    cfg: ExperimentsConfig,
    config_dir: Path,
    experiment_id: str,
) -> ResolvedExperimentBinding:
    """Resolve one experiment by id."""
    for entry in cfg.experiments:
        if entry.id == experiment_id:
            return resolve_experiment_binding(cfg, config_dir, entry)
    raise KeyError(f"Experiment '{experiment_id}' not found in master registry.")


def list_experiment_bindings(
    cfg: ExperimentsConfig,
    config_dir: Path,
) -> list[ResolvedExperimentBinding]:
    """Resolve all experiment entries with concrete config paths."""
    return [resolve_experiment_binding(cfg, config_dir, entry) for entry in cfg.experiments]
