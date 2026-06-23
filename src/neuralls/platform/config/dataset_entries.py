"""DLKit dataset-entry adapters for resolved neuralls dataset paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.infrastructure.config.core.patching import patch_model
from dlkit.infrastructure.config.data_entries import (
    DataEntry,
    DataRole,
    NpyEntry,
    PathBasedEntry,
    ZarrEntry,
)

from neuralls.shared.types import EntryRole

# ponytail: add new format here + import its dlkit entry type — no other changes needed
_SUFFIX_TO_ENTRY: dict[str, type[PathBasedEntry]] = {
    ".npy": NpyEntry,
}


def entry_from_path(
    path: Path,
    *,
    name: str,
    model_input: bool,
    role: EntryRole,
) -> DataEntry:
    """Build a dlkit DataEntry from an artifact path.

    Format is derived from path suffix: .npy → NpyEntry, anything else → ZarrEntry.
    Delegates validation (zarr.json existence, .npy suffix) to dlkit.
    """
    data_role = DataRole.FEATURE if role == "feature" else DataRole.TARGET
    entry_type = _SUFFIX_TO_ENTRY.get(path.suffix, ZarrEntry)
    return entry_type(name=name, path=path, model_input=model_input, data_role=data_role)


def apply_placeholder_metadata(
    entries: list[DataEntry],
    config_entries: tuple[DataEntry, ...],
) -> list[DataEntry]:
    """Merge placeholder metadata from model TOML into resolved dataset entries."""
    patches_by_name: dict[str, dict[str, Any]] = {}
    for entry in config_entries:
        if entry.name is None:
            continue
        patch = _metadata_patch_from_entry(entry)
        if patch:
            patches_by_name[entry.name] = patch
    if not patches_by_name:
        return entries

    consumed: set[str] = set()
    merged: list[DataEntry] = []
    for entry in entries:
        if entry.name in patches_by_name:
            merged.append(patch_model(entry, patches_by_name[entry.name], revalidate=False))
            consumed.add(entry.name)
        else:
            merged.append(entry)

    unmatched = patches_by_name.keys() - consumed
    if unmatched:
        raise ValueError(
            "DATASET placeholder entries must match injected runtime entry names. "
            f"Unmatched entries: {sorted(unmatched)}"
        )
    return merged


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
