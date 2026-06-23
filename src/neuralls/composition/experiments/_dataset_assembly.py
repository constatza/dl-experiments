"""Dataset assembly helpers for the training workflow.

Translates TOML feature declarations and storage artifacts into resolved
dlkit DataEntry objects, validates the runtime dataset contract, and leaves
third-party entry construction to platform adapters.
"""

from __future__ import annotations

from dlkit.infrastructure.config.data_entries import DataEntry
from dlkit.infrastructure.config.dataset_settings import DatasetSettings
from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)
from loguru import logger

from neuralls.composition.experiments.runtime_dataset_contract import RuntimeDatasetContract
from neuralls.platform.config.dataset_entries import entry_from_path
from neuralls.platform.config.models.workspace import ExperimentWorkspace
from neuralls.platform.storage.training_artifacts import (
    ArraySource,
    TrainingArrays,
    load_training_arrays,
)

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
    determines the name used when forwarding the RHS tensor to the model.

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

    Preserves TOML declaration order — the i-th returned name maps to parameters_i.

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


def _create_feature_entries(
    arrays: TrainingArrays,
    contract: RuntimeDatasetContract,
    declared_extra_names: list[str],
    primary_name: str,
) -> list[DataEntry]:
    """Create resolved feature entries from dataset artifacts.

    The i-th declared extra name maps to parameters_i by position.

    Args:
        arrays: Training data artifact sources.
        contract: Runtime dataset entry name contract.
        declared_extra_names: Ordered extra feature names declared in [[DATASET.features]].
        primary_name: TOML-declared name for the primary (RHS/branch) input feature.

    Returns:
        List of feature entries: base entries (rhs, matrix) plus any extras.

    Raises:
        ValueError: If more extras are declared than parameter sources available.
    """
    if len(declared_extra_names) > len(arrays.parameter_sources):
        raise ValueError(
            f"Model declares {len(declared_extra_names)} extra features but dataset has "
            f"{len(arrays.parameter_sources)} parameters_* files."
        )
    base: list[DataEntry] = [
        _feature_entry_from_source(arrays.rhs_source, name=primary_name, model_input=True),
        _feature_entry_from_source(
            arrays.matrix_source, name=contract.matrix_input_name, model_input=False
        ),
    ]
    extras: list[DataEntry] = [
        _feature_entry_from_source(arrays.parameter_sources[i], name=name, model_input=True)
        for i, name in enumerate(declared_extra_names)
    ]
    return [*base, *extras]


def _create_target_entries(
    source: ArraySource,
    contract: RuntimeDatasetContract,
) -> list[DataEntry]:
    """Create resolved target entries from dataset artifacts.

    Args:
        source: Solutions artifact source.
        contract: Runtime dataset entry name contract.

    Returns:
        The canonical supervised target entry.
    """
    return [_target_entry_from_source(source, name=contract.target_name)]


def _load_and_prepare_data(
    settings: TrainingWorkflowSettings,
    workspace: ExperimentWorkspace,
    contract: RuntimeDatasetContract,
) -> tuple[TrainingArrays, list[DataEntry], list[DataEntry]]:
    """Resolve training data artifacts and build dlkit entries.

    Args:
        settings: DLKit training or optimization workflow settings.
        workspace: Experiment workspace (provides data_dir path).
        contract: Runtime dataset contract defining canonical entry names.

    Returns:
        Tuple of (arrays, features, targets) where:
            - arrays: Resolved data artifact sources
            - features: List of DataEntry objects for platform adapters
            - targets: List of DataEntry objects for platform adapters
    """
    arrays = load_training_arrays(workspace.data_dir)
    fmt = arrays.rhs_source.path.suffix.lstrip(".") or "zarr"
    logger.info(
        "Resolved training artifact sources ({} samples, format={}) from {}",
        arrays.sample_count,
        fmt,
        workspace.data_dir,
    )
    primary_name = _primary_feature_name_from_settings(settings, contract)
    extra_names = _extra_feature_names_from_settings(settings, contract)
    features = _create_feature_entries(arrays, contract, extra_names, primary_name)
    targets = _create_target_entries(arrays.solutions_source, contract)
    return arrays, features, targets


def _feature_entry_from_source(
    source: ArraySource,
    *,
    name: str,
    model_input: bool,
) -> DataEntry:
    """Create a feature DataEntry from one resolved artifact source."""
    return entry_from_path(source.path, name=name, model_input=model_input, role="feature")


def _target_entry_from_source(source: ArraySource, *, name: str) -> DataEntry:
    """Create a target DataEntry from one resolved artifact source."""
    return entry_from_path(source.path, name=name, model_input=True, role="target")
