"""DLKit dataset-entry adapters for resolved neuralls dataset specs."""

from __future__ import annotations

from typing import Any

from dlkit.common.geometry import FieldRole, GeometryKind
from dlkit.infrastructure.config.core.patching import patch_model
from dlkit.infrastructure.config.data_entries import (
    DataEntry,
    DataRole,
    NpyEntry,
    ValueEntry,
    ZarrEntry,
)

from neuralls.shared.types import EntryRole, ResolvedDatasetEntrySpec


def build_data_entries(specs: list[ResolvedDatasetEntrySpec]) -> list[DataEntry]:
    """Translate resolved neuralls dataset specs into concrete DLKit entries."""
    return [_build_data_entry(spec) for spec in specs]


def apply_placeholder_metadata(
    specs: list[ResolvedDatasetEntrySpec],
    config_entries: tuple[DataEntry, ...],
) -> list[ResolvedDatasetEntrySpec]:
    """Merge placeholder metadata from model TOML into resolved dataset specs."""
    patches_by_name: dict[str, dict[str, Any]] = {}
    for entry in config_entries:
        if entry.name is None:
            continue
        patch = _metadata_patch_from_entry(entry)
        if patch:
            patches_by_name[entry.name] = patch
    if not patches_by_name:
        return specs

    consumed: set[str] = set()
    merged: list[ResolvedDatasetEntrySpec] = []
    for spec in specs:
        if spec.name in patches_by_name:
            metadata = dict(spec.metadata)
            metadata.update(patches_by_name[spec.name])
            merged.append(
                ResolvedDatasetEntrySpec(
                    name=spec.name,
                    path=spec.path,
                    format=spec.format,
                    role=spec.role,
                    model_input=spec.model_input,
                    value=spec.value,
                    metadata=metadata,
                )
            )
            consumed.add(spec.name)
        else:
            merged.append(spec)

    unmatched = patches_by_name.keys() - consumed
    if unmatched:
        raise ValueError(
            "DATASET placeholder entries must match injected runtime entry names. "
            f"Unmatched entries: {sorted(unmatched)}"
        )
    return merged


def _build_data_entry(spec: ResolvedDatasetEntrySpec) -> DataEntry:
    entry = _base_entry_from_spec(spec)
    if not spec.metadata:
        return entry
    return patch_model(entry, spec.metadata, revalidate=False)


def _base_entry_from_spec(spec: ResolvedDatasetEntrySpec) -> DataEntry:
    data_role = _resolve_data_role(spec.role)
    if spec.value is not None:
        return ValueEntry(
            name=spec.name,
            value=spec.value,
            model_input=spec.model_input,
            data_role=data_role,
        )
    match spec.format:
        case "npy":
            return NpyEntry(
                name=spec.name,
                path=spec.path,
                model_input=spec.model_input,
                data_role=data_role,
            )
        case "zarr":
            return ZarrEntry(
                name=spec.name,
                path=spec.path,
                model_input=spec.model_input,
                data_role=data_role,
            )


def _resolve_data_role(role: EntryRole) -> DataRole:
    if role == "feature":
        return DataRole.FEATURE
    return DataRole.TARGET


def _metadata_patch_from_entry(entry: DataEntry) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    transforms = getattr(entry, "transforms", None)
    if transforms:
        patch["transforms"] = list(transforms)
    field_role = getattr(entry, "field_role", None)
    if field_role is not None and field_role != FieldRole.FEATURE:
        patch["field_role"] = field_role
    geometry_kind = getattr(entry, "geometry_kind", None)
    if geometry_kind is not None and geometry_kind != GeometryKind.TABULAR:
        patch["geometry_kind"] = geometry_kind
    return patch
