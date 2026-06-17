"""Dataset assembly helpers for the training workflow.

Translates TOML feature declarations and storage artifacts into DLKit DataEntry
objects, validates the runtime dataset contract, and merges TOML metadata into
file-backed entries.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dlkit.common.geometry import FieldRole, GeometryKind
from dlkit.infrastructure.config.core.patching import patch_model
from dlkit.infrastructure.config.data_entries import (
    DataEntry,
    DataRole,
    ValueEntry,
    ZarrEntry,
)
from dlkit.infrastructure.config.dataset_settings import DatasetSettings
from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)
from loguru import logger

from neuralls.composition.experiments.runtime_dataset_contract import RuntimeDatasetContract
from neuralls.platform.storage.datasets import load_dense_training_arrays
from neuralls.platform.storage.training_artifacts import TrainingArrays, load_training_arrays

type TrainingWorkflowSettings = TrainingWorkflowConfig | OptimizationWorkflowConfig


def _validate_dataset_section(settings: TrainingWorkflowSettings) -> None:
    """Validate that DATASET section exists in settings.

    Args:
        settings: DLKit workflow settings to validate.

    Raises:
        ValueError: If [DATASET] section is missing from config.
    """
    if settings.DATASET is None:
        raise ValueError("Config is missing [DATASET] section")


def _find_duplicate_entry_names(entries: tuple[DataEntry, ...]) -> set[str]:
    """Return duplicated non-null dataset entry names.

    Args:
        entries: Tuple of DataEntry objects to inspect.

    Returns:
        Set of names that appear more than once.
    """
    seen: set[str] = set()
    duplicates: set[str] = set()
    for entry in entries:
        if entry.name is None:
            continue
        if entry.name in seen:
            duplicates.add(entry.name)
        seen.add(entry.name)
    return duplicates


def _validate_runtime_dataset_contract(
    settings: TrainingWorkflowSettings,
    contract: RuntimeDatasetContract,
) -> None:
    """Validate the local runtime dataset-entry contract for training.

    This training bridge keeps storage artifact names separate from runtime
    entry names. Runtime placeholder entries and loss routing must resolve
    through the caller-supplied contract.

    Args:
        settings: DLKit workflow settings carrying the TOML feature declarations.
        contract: Runtime dataset contract defining canonical entry names.

    Raises:
        ValueError: If duplicate entries, unsupported target names, or
            mismatched loss_function.target_key are found.
    """
    _validate_dataset_section(settings)
    dataset = settings.DATASET or DatasetSettings()

    duplicate_features = _find_duplicate_entry_names(dataset.features)
    if duplicate_features:
        raise ValueError(
            f"Duplicate DATASET feature entry names are not allowed: {sorted(duplicate_features)}"
        )

    duplicate_targets = _find_duplicate_entry_names(dataset.targets)
    if duplicate_targets:
        raise ValueError(
            f"Duplicate DATASET target entry names are not allowed: {sorted(duplicate_targets)}"
        )

    target_names = {entry.name for entry in dataset.targets if entry.name is not None}
    unsupported_target_names = sorted(name for name in target_names if name != contract.target_name)
    if unsupported_target_names:
        raise ValueError(
            "DATASET target placeholders must use only the resolved supervised "
            f"target name '{contract.target_name}', got {unsupported_target_names}."
        )

    training_cfg = settings.TRAINING
    loss_function = getattr(training_cfg, "loss_function", None) if training_cfg else None
    target_key = getattr(loss_function, "target_key", None)
    if target_key is not None and target_key != contract.loss_target_key:
        raise ValueError(
            "TRAINING.loss_function.target_key must resolve to the runtime "
            f"supervised target '{contract.loss_target_key}', got '{target_key}'."
        )


def _merge_entry_metadata[T: DataEntry](
    entries: list[T],
    config_entries: tuple[DataEntry, ...],
) -> list[T]:
    """Inject transforms, field_role, and geometry_kind from TOML placeholders into file-backed entries.

    Propagates transforms and any non-default field_role / geometry_kind declared in
    [[DATASET.features]] or [[DATASET.targets]] into the corresponding file-backed entries.
    This makes the TOML the declarative source of truth for field geometry semantics
    (e.g. ``field_role = "target_coordinates"`` for DeepONet query inputs).

    Args:
        entries: File-backed Feature/Target objects built from disk artifacts.
        config_entries: Placeholder entries parsed from [[DATASET.features]] or
            [[DATASET.targets]] in the model TOML (name + metadata, no path).

    Returns:
        New list where each entry whose name matches a config placeholder has
        the placeholder's metadata applied via patch_model; all others unchanged.

    Raises:
        ValueError: If any config placeholder name does not match an injected entry.
    """
    patches_by_name: dict[str, dict[str, Any]] = {}
    for e in config_entries:
        if e.name is None:
            continue
        patch: dict[str, Any] = {}
        if e.transforms:
            patch["transforms"] = list(e.transforms)
        if e.field_role != FieldRole.FEATURE:
            patch["field_role"] = e.field_role
        if e.geometry_kind != GeometryKind.TABULAR:
            patch["geometry_kind"] = e.geometry_kind
        if patch:
            patches_by_name[e.name] = patch
    if not patches_by_name:
        return entries
    result: list[T] = []
    consumed: set[str] = set()
    for entry in entries:
        name = entry.name
        if name is not None and name in patches_by_name:
            entry = patch_model(entry, patches_by_name[name], revalidate=False)
            consumed.add(name)
        result.append(entry)
    unmatched = patches_by_name.keys() - consumed
    if unmatched:
        raise ValueError(
            "DATASET placeholder entries must match injected runtime entry names. "
            f"Unmatched entries: {sorted(unmatched)}"
        )
    return result


def _primary_feature_name_from_settings(
    settings: TrainingWorkflowSettings,
    contract: RuntimeDatasetContract,
) -> str:
    """Resolve the dispatch name for the primary (RHS/branch) input feature.

    The first declared [[DATASET.features]] entry that is not the matrix input
    determines the name used when forwarding the RHS tensor to the model. This
    allows DeepONet-style models that expect ``forward(u, y)`` to declare ``u``
    as their primary feature without the composition layer needing model-family
    knowledge.

    Args:
        settings: DLKit workflow settings carrying the TOML feature declarations.
        contract: Provides the matrix_input_name to skip and the fallback
            primary_input_name when no feature is declared.

    Returns:
        The dispatch name for the RHS tensor (e.g. ``"x"``, ``"u"``).
    """
    dataset = settings.DATASET
    if dataset is None:
        return contract.primary_input_name
    for entry in dataset.features:
        if entry.name is not None and entry.name != contract.matrix_input_name:
            return entry.name
    return contract.primary_input_name


def _extra_feature_names_from_settings(
    settings: TrainingWorkflowSettings,
    contract: RuntimeDatasetContract,
) -> list[str]:
    """Extract extra feature names declared in [[DATASET.features]] beyond the base entries.

    Preserves TOML declaration order — the i-th returned name maps to parameters_i.zarr.
    The primary (RHS/branch) feature and the matrix entry are excluded; everything else
    is an extra sourced from parameters_*.zarr.

    Args:
        settings: DLKit workflow settings; DATASET.features carries the TOML declarations.
        contract: Provides the matrix_input_name to exclude.

    Returns:
        Ordered list of extra feature names beyond the primary and matrix entries.
    """
    dataset = settings.DATASET
    if dataset is None:
        return []
    primary_name = _primary_feature_name_from_settings(settings, contract)
    base = {primary_name, contract.matrix_input_name}
    return [e.name for e in dataset.features if e.name is not None and e.name not in base]


def _create_feature_configs(
    arrays: TrainingArrays,
    rhs_data: np.ndarray,
    contract: RuntimeDatasetContract,
    declared_extra_names: list[str],
    primary_name: str,
) -> list[DataEntry]:
    """Create in-memory Feature configs from dataset artifacts.

    The i-th declared extra name maps to parameters_i.zarr by position.

    Args:
        arrays: Training data artifact paths (used for matrix and parameters paths).
        rhs_data: RHS array loaded from rhs.zarr.
        contract: Runtime dataset entry name contract.
        declared_extra_names: Ordered extra feature names declared in [[DATASET.features]]
            beyond the base and matrix entries (TOML declaration order preserved).
        primary_name: TOML-declared name for the primary (RHS/branch) input feature.
            Used as the DataEntry name so DLKit dispatches it to the correct
            forward() parameter (e.g. ``"u"`` for DeepONet).

    Returns:
        List of feature entries: base entries (rhs, matrix) plus any extras
        mapped by index to parameters_i.zarr.

    Raises:
        ValueError: If more extras are declared than parameters_*.zarr files available.
    """
    if len(declared_extra_names) > len(arrays.parameters_zarr):
        raise ValueError(
            f"Model declares {len(declared_extra_names)} extra features but dataset has "
            f"{len(arrays.parameters_zarr)} parameters_*.zarr files."
        )
    base: list[DataEntry] = [
        ValueEntry(name=primary_name, value=rhs_data),
        ZarrEntry(name=contract.matrix_input_name, path=arrays.matrix_zarr, model_input=False),
    ]
    extras: list[DataEntry] = [
        ZarrEntry(
            name=name,
            path=arrays.parameters_zarr[i],
            model_input=True,
            field_role=FieldRole.FEATURE,
            geometry_kind=GeometryKind.TABULAR,
        )
        for i, name in enumerate(declared_extra_names)
    ]
    return [*base, *extras]


def _create_target_configs(
    solutions_data: np.ndarray,
    contract: RuntimeDatasetContract,
) -> list[DataEntry]:
    """Create in-memory Target configs from dataset artifacts.

    Args:
        solutions_data: Solutions array loaded from solutions.zarr.
        contract: Runtime dataset entry name contract.

    Returns:
        The canonical supervised target entry.
    """
    return [ValueEntry(name=contract.target_name, value=solutions_data, data_role=DataRole.TARGET)]


def _load_and_prepare_data(
    settings: TrainingWorkflowSettings,
    workspace: Any,
    contract: RuntimeDatasetContract,
) -> tuple[TrainingArrays, list[DataEntry], list[DataEntry]]:
    """Load training data and create Feature/Target configurations.

    Args:
        settings: DLKit training or optimization workflow settings.
        workspace: Experiment workspace (provides data_dir path).
        contract: Runtime dataset contract defining canonical entry names.

    Returns:
        Tuple of (arrays, features, targets) where:
            - arrays: Resolved data artifact paths
            - features: List of in-memory / path-dropped feature configs for DLKit
            - targets: List of in-memory target configs for DLKit
    """
    arrays = load_training_arrays(workspace.data_dir)
    logger.info(
        "Loading training arrays ({} samples) from {}", arrays.sample_count, workspace.data_dir
    )
    rhs_data, solutions_data = load_dense_training_arrays(workspace.data_dir)
    logger.info(
        "Training arrays loaded — rhs {}, solutions {}", rhs_data.shape, solutions_data.shape
    )
    primary_name = _primary_feature_name_from_settings(settings, contract)
    extra_names = _extra_feature_names_from_settings(settings, contract)
    features = _create_feature_configs(arrays, rhs_data, contract, extra_names, primary_name)
    targets = _create_target_configs(solutions_data, contract)
    return arrays, features, targets
