"""Pydantic models for the master experiments registry."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


def resolve_display_name(entity_id: str, display_name: str | None) -> str:
    """Return the human-facing label for a registry entry."""
    if display_name is None:
        return entity_id
    stripped = display_name.strip()
    return stripped or entity_id


class MlflowTopologyConfig(BaseModel):
    """MLflow topology configuration.

    Attributes:
        tracking_uri: MLflow tracking URI.
        artifacts_destination: Optional artifacts root for local sqlite tracking.
    """

    tracking_uri: str = Field(default="http://127.0.0.1:5000", min_length=1)
    artifacts_destination: str | None = None
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentNamesConfig(BaseModel):
    """MLflow experiment names configuration.

    Attributes:
        training: Name for training experiments.
        comparison: Name for comparison experiments.
    """

    training: str = "neuralls-training"
    comparison: str = "Comparisons"
    model_config = ConfigDict(extra="forbid", frozen=True)


class RegistryEntry(BaseModel):
    """Single registry entry with an explicit config path.

    Attributes:
        id: Stable lookup id used by the master registry.
        path: Relative or absolute config path.
        display_name: Optional human-facing label.
    """

    id: str = Field(..., min_length=1)
    path: Path
    display_name: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_key(cls, data: object) -> object:
        """Accept legacy ``key`` while normalizing to ``id`` immediately."""
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        if "key" in raw and "id" not in raw:
            raw["id"] = raw.pop("key")
        return raw

    @property
    def effective_display_name(self) -> str:
        """Return the configured label or fall back to the id."""
        return resolve_display_name(self.id, self.display_name)


class ExperimentEntry(BaseModel):
    """Single experiment entry from the ``[[experiments]]`` TOML table.

    Attributes:
        id: Stable experiment identifier.
        dataset_id: Dataset registry id.
        model_id: Model registry id.
        display_name: Optional human-facing label.
    """

    id: str = Field(..., min_length=1)
    dataset_id: str = Field(..., alias="dataset")
    model_id: str = Field(..., alias="model")
    checkpoint_path: Path | None = None
    display_name: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_key(cls, data: object) -> object:
        """Accept legacy ``key`` while normalizing to ``id`` immediately."""
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        if "key" in raw and "id" not in raw:
            raw["id"] = raw.pop("key")
        return raw

    @property
    def effective_display_name(self) -> str:
        """Return the configured label or fall back to the id."""
        return resolve_display_name(self.id, self.display_name)


class RunEntry(BaseModel):
    """Single run entry from the ``[[run]]`` TOML table.

    Uses direct relative config paths instead of short name references.

    Attributes:
        id: Stable identifier for the run.
        model_config_path: Relative path to model config.
        data_config_path: Relative path to data config.
    """

    id: str
    model_config_path: str = Field(alias="model_config")
    data_config_path: str = Field(alias="data_config")

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_key(cls, data: object) -> object:
        """Accept legacy ``key`` while normalizing to ``id`` immediately."""
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        if "key" in raw and "id" not in raw:
            raw["id"] = raw.pop("key")
        return raw

    @property
    def effective_display_name(self) -> str:
        """Direct runs do not carry separate display names."""
        return self.id


class ComparisonRegistryEntry(RegistryEntry):
    """Comparison registry entry with optional experiment filter.

    Attributes:
        experiments: Optional list of experiment ids to include in this
            comparison. If absent or empty, all active experiments are used.
    """

    experiments: list[str] = Field(default_factory=list)


def _dedupe_ids(kind: str, entries: Sequence[RegistryEntry]) -> None:
    """Reject duplicate registry ids with a focused error."""
    seen: set[str] = set()
    duplicates = sorted({entry.id for entry in entries if entry.id in seen or seen.add(entry.id)})
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"Duplicate {kind} ids in experiments config: {joined}.")


def _dedupe_experiment_ids(entries: list[ExperimentEntry]) -> None:
    """Reject duplicate experiment ids with a focused error."""
    seen: set[str] = set()
    duplicates = sorted({entry.id for entry in entries if entry.id in seen or seen.add(entry.id)})
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"Duplicate experiment ids in experiments config: {joined}.")


def _registry_ids(entries: list[RegistryEntry]) -> set[str]:
    """Return registry ids for fast membership checks."""
    return {entry.id for entry in entries}


def _validate_comparison_experiment_filter_refs(
    comparisons: list[ComparisonRegistryEntry],
    experiment_ids: set[str],
) -> None:
    """Reject comparison experiment filter refs that reference unknown experiment ids."""
    for entry in comparisons:
        unknown = [eid for eid in entry.experiments if eid not in experiment_ids]
        if unknown:
            joined = ", ".join(unknown)
            raise ValueError(
                f"Comparison '{entry.id}' experiments filter references unknown "
                f"experiment ids: {joined}."
            )


def _validate_experiment_registry_refs(
    experiments: list[ExperimentEntry],
    *,
    dataset_ids: set[str],
    model_ids: set[str],
) -> None:
    """Reject experiments that reference undefined dataset/model registry ids."""
    for entry in experiments:
        if entry.dataset_id not in dataset_ids:
            raise ValueError(
                f"Experiment '{entry.id}' references dataset id "
                f"'{entry.dataset_id}', but [[datasets]] does not define it."
            )
        if entry.model_id not in model_ids:
            raise ValueError(
                f"Experiment '{entry.id}' references model id "
                f"'{entry.model_id}', but [[models]] does not define it."
            )


class ExperimentsConfig(BaseModel):
    """Top-level master registry configuration.

    Attributes:
        output_dir: Training output directory (optional; derived from mlflow if absent).
        datasets: Dataset registry entries.
        models: Model registry entries.
        comparisons: Comparison registry entries.
        experiments: Experiment entries referencing registry ids.
        run: Legacy direct-path entries.
        project_root: Optional project root path.
        mlflow: MLflow topology config (tracking URI, etc.).
        names: MLflow experiment names for training and comparison.
    """

    output_dir: Path | None = Field(default=None, description="Training output directory")
    datasets: list[RegistryEntry] = Field(default_factory=list)
    models: list[RegistryEntry] = Field(default_factory=list)
    comparisons: list[ComparisonRegistryEntry] = Field(default_factory=list)
    experiments: list[ExperimentEntry] = Field(default_factory=list)
    run: list[RunEntry] = Field(default_factory=list)
    project_root: Path | None = Field(default=None)
    mlflow: MlflowTopologyConfig = Field(default_factory=MlflowTopologyConfig)
    names: ExperimentNamesConfig = Field(default_factory=ExperimentNamesConfig)

    model_config = ConfigDict(extra="allow", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_keys(cls, data: object) -> object:
        """Support legacy singular table names alongside the new registry schema."""
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        if "experiment" in raw and "experiments" not in raw:
            raw["experiments"] = raw["experiment"]
        if "comparison_profiles" in raw:
            raise ValueError(
                "Legacy 'comparison_profiles' is no longer supported. "
                "Use [[comparisons]] entries in experiments config."
            )
        return raw

    @model_validator(mode="after")
    def validate_unique_ids(self) -> ExperimentsConfig:
        """Reject duplicate ids across registry collections."""
        _dedupe_ids("dataset registry", self.datasets)
        _dedupe_ids("model registry", self.models)
        _dedupe_ids("comparison registry", self.comparisons)
        _dedupe_experiment_ids(self.experiments)
        _validate_experiment_registry_refs(
            self.experiments,
            dataset_ids=_registry_ids(self.datasets),
            model_ids=_registry_ids(self.models),
        )
        _validate_comparison_experiment_filter_refs(
            self.comparisons,
            experiment_ids={e.id for e in self.experiments},
        )
        return self
