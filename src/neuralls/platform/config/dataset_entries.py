"""DLKit dataset-entry adapters for resolved neuralls dataset specs."""

from __future__ import annotations

from typing import Any

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
        case _:
            raise ValueError(f"Unsupported dataset format: {spec.format!r}")


def _resolve_data_role(role: EntryRole) -> DataRole:
    match role:
        case "feature":
            return DataRole.FEATURE
        case "target":
            return DataRole.TARGET
        case _:
            raise ValueError(f"Unknown entry role: {role!r}")


def _metadata_patch_from_entry(entry: DataEntry) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    if entry.transforms:
        patch["transforms"] = list(entry.transforms)
    if entry.dtype is not None:
        patch["dtype"] = entry.dtype
    if entry.loss_input is not None:
        patch["loss_input"] = entry.loss_input
    if entry.write:
        patch["write"] = True
    if not entry.model_input:
        patch["model_input"] = False
    return patch
