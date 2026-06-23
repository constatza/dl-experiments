"""Dataset assembly helpers for the training workflow.

Translates TOML feature declarations and storage artifacts into resolved
dataset-entry specs, validates the runtime dataset contract, and leaves
third-party entry construction to platform adapters.
"""

from __future__ import annotations

from typing import Any

from dlkit.infrastructure.config.data_entries import (
    DataEntry,
)
from dlkit.infrastructure.config.dataset_settings import DatasetSettings
from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)
from loguru import logger

from neuralls.composition.experiments.runtime_dataset_contract import RuntimeDatasetContract
from neuralls.platform.storage.training_artifacts import (
    ArraySource,
    TrainingArrays,
    load_training_arrays,
)
from neuralls.shared.types import ResolvedDatasetEntrySpec

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
    if target_key is None:
        return
    if target_key != contract.loss_target_key:
        raise ValueError(
            "TRAINING.loss_function.target_key must resolve to the runtime "
            f"supervised target '{contract.loss_target_key}', got '{target_key}'."
        )


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
        if entry.name is None:
            continue
        if entry.name == contract.matrix_input_name:
            continue
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
    contract: RuntimeDatasetContract,
    declared_extra_names: list[str],
    primary_name: str,
) -> list[ResolvedDatasetEntrySpec]:
    """Create resolved feature specs from dataset artifacts.

    The i-th declared extra name maps to parameters_i.zarr by position.

    Args:
        arrays: Training data artifact sources.
        contract: Runtime dataset entry name contract.
        declared_extra_names: Ordered extra feature names declared in [[DATASET.features]]
            beyond the base and matrix entries (TOML declaration order preserved).
        primary_name: TOML-declared name for the primary (RHS/branch) input feature.
            Used as the DataEntry name so DLKit dispatches it to the correct
            forward() parameter (e.g. ``"u"`` for DeepONet).

    Returns:
        List of feature specs: base entries (rhs, matrix) plus any extras
        mapped by index to parameters_i artifacts.

    Raises:
        ValueError: If more extras are declared than parameters_*.zarr files available.
    """
    if len(declared_extra_names) > len(arrays.parameter_sources):
        raise ValueError(
            f"Model declares {len(declared_extra_names)} extra features but dataset has "
            f"{len(arrays.parameter_sources)} parameters_* files."
        )
    base: list[ResolvedDatasetEntrySpec] = [
        _feature_entry_from_source(
            arrays.rhs_source,
            name=primary_name,
            model_input=True,
        )
    ]
    base.append(
        _feature_entry_from_source(
            arrays.matrix_source,
            name=contract.matrix_input_name,
            model_input=False,
        )
    )

    extras: list[ResolvedDatasetEntrySpec] = []
    for i, name in enumerate(declared_extra_names):
        source = arrays.parameter_sources[i]
        extras.append(
            _feature_entry_from_source(
                source,
                name=name,
                model_input=True,
            )
        )
    return [*base, *extras]


def _create_target_configs(
    source: ArraySource,
    contract: RuntimeDatasetContract,
) -> list[ResolvedDatasetEntrySpec]:
    """Create resolved target specs from dataset artifacts.

    Args:
        source: Solutions artifact source.
        contract: Runtime dataset entry name contract.

    Returns:
        The canonical supervised target spec.
    """
    return [_target_entry_from_source(source, name=contract.target_name)]


def _load_and_prepare_data(
    settings: TrainingWorkflowSettings,
    workspace: Any,
    contract: RuntimeDatasetContract,
) -> tuple[TrainingArrays, list[ResolvedDatasetEntrySpec], list[ResolvedDatasetEntrySpec]]:
    """Load training data and create resolved feature/target specs.

    Args:
        settings: DLKit training or optimization workflow settings.
        workspace: Experiment workspace (provides data_dir path).
        contract: Runtime dataset contract defining canonical entry names.

    Returns:
        Tuple of (arrays, features, targets) where:
            - arrays: Resolved data artifact sources
            - features: List of resolved feature specs for platform adapters
            - targets: List of resolved target specs for platform adapters
    """
    arrays = load_training_arrays(workspace.data_dir)
    logger.info(
        "Loading training artifact sources ({} samples) from {}",
        arrays.sample_count,
        workspace.data_dir,
    )
    primary_name = _primary_feature_name_from_settings(settings, contract)
    extra_names = _extra_feature_names_from_settings(settings, contract)
    features = _create_feature_configs(arrays, contract, extra_names, primary_name)
    targets = _create_target_configs(arrays.solutions_source, contract)
    return arrays, features, targets


def _feature_entry_from_source(
    source: ArraySource,
    *,
    name: str,
    model_input: bool,
) -> ResolvedDatasetEntrySpec:
    """Create a feature spec from one resolved artifact source."""
    return ResolvedDatasetEntrySpec(
        name=name,
        path=source.path,
        format=source.format,
        role="feature",
        model_input=model_input,
    )


def _target_entry_from_source(source: ArraySource, *, name: str) -> ResolvedDatasetEntrySpec:
    """Create a target spec from one resolved artifact source."""
    return ResolvedDatasetEntrySpec(
        name=name,
        path=source.path,
        format=source.format,
        role="target",
        model_input=True,
    )
