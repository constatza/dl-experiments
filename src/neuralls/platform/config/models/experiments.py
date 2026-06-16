"""Pydantic models for the top-level case config."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from neuralls.shared.constants import DEFAULT_RTOL, DEFAULT_ATOL, DEFAULT_M_MAX
from neuralls.platform.config.models.preconditioner import PreconditionerConfig
from neuralls.platform.config.models.id_generation import (
    _build_display_lookup,
    _infer_comparison_display_name,
    _infer_comparison_id,
    _infer_experiment_display_name,
    _infer_experiment_id,
)


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

    tracking_uri: str | None = None
    artifacts_destination: str | None = None
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("tracking_uri", "artifacts_destination", mode="before")
    @classmethod
    def _expand_if_placeholder(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Expand ${NEURALLS_*} placeholders only; plain URIs pass through unchanged.

        Args:
            v: Raw string value from config field.
            info: Pydantic validation info carrying context.

        Returns:
            Expanded path string if v contains a placeholder, otherwise v unchanged.
        """
        if v is None or info.context is None or "${" not in v:
            return v
        from neuralls.platform.config.context import ConfigContext, expand_config_path

        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))

    @model_validator(mode="after")
    def validate_topology(self) -> MlflowTopologyConfig:
        """Reject ambiguous partial topology definitions."""
        if self.artifacts_destination is not None and self.tracking_uri is None:
            raise ValueError(
                "Case config [mlflow] cannot set artifacts_destination without tracking_uri."
            )
        return self


class ExperimentNamesConfig(BaseModel):
    """MLflow experiment names configuration.

    Attributes:
        training: Name for training experiments.
        comparison: Name for comparison experiments.
    """

    training: str = "Train"
    comparison: str = "Comparisons"
    model_config = ConfigDict(extra="forbid", frozen=True)


class RegistryEntry(BaseModel):
    """Single registry entry with an explicit config path.

    Attributes:
        id: Stable lookup id used by the case config registry.
        path: Relative or absolute config path.
        display_name: Optional human-facing label.
    """

    id: str = Field(..., min_length=1)
    path: Path
    display_name: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("path", mode="before")
    @classmethod
    def _expand_path(cls, v: object, info: ValidationInfo) -> object:
        """Expand ${NEURALLS_*} placeholders and resolve to absolute path.

        Args:
            v: Raw value from config field.
            info: Pydantic validation info carrying context.

        Returns:
            Resolved absolute path string, or original value if not a string.
        """
        if info.context is None or not isinstance(v, str):
            return v
        from neuralls.platform.config.context import ConfigContext, expand_config_path

        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))

    @property
    def effective_display_name(self) -> str:
        """Return the configured label or fall back to the id."""
        return resolve_display_name(self.id, self.display_name)


class ExperimentEntry(BaseModel):
    """Single experiment entry from the ``[[experiments]]`` case table.

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

    @field_validator("checkpoint_path", mode="before")
    @classmethod
    def _expand_checkpoint_path(cls, v: object, info: ValidationInfo) -> object:
        """Expand ${NEURALLS_*} placeholders and resolve checkpoint path.

        Args:
            v: Raw value from config field.
            info: Pydantic validation info carrying context.

        Returns:
            Resolved absolute path string, or original value if not a string.
        """
        if v is None or info.context is None or not isinstance(v, str):
            return v
        from neuralls.platform.config.context import ConfigContext, expand_config_path

        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))

    @property
    def effective_display_name(self) -> str:
        """Return the configured label or fall back to the id."""
        return resolve_display_name(self.id, self.display_name)


class RunEntry(BaseModel):
    """Single run entry from the ``[[run]]`` TOML table.

    Uses direct relative config paths instead of short name references.

    Attributes:
        id: Stable identifier for the legacy direct-path run.
        model_config_path: Relative path to model config.
        data_config_path: Relative path to data config.
    """

    id: str
    model_config_path: str = Field(alias="model_config")
    data_config_path: str = Field(alias="data_config")

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @field_validator("model_config_path", "data_config_path", mode="before")
    @classmethod
    def _expand_config_paths(cls, v: object, info: ValidationInfo) -> object:
        """Expand ${NEURALLS_*} placeholders and resolve config paths.

        Args:
            v: Raw value from config field.
            info: Pydantic validation info carrying context.

        Returns:
            Resolved absolute path string, or original value if not a string.
        """
        if info.context is None or not isinstance(v, str):
            return v
        from neuralls.platform.config.context import ConfigContext, expand_config_path

        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))

    @property
    def effective_display_name(self) -> str:
        """Direct runs do not carry separate display names."""
        return self.id


class ComparisonDefaults(BaseModel):
    """Shared methodology defaults applied to all ``[[comparisons]]`` entries.

    Attributes:
        rtol: Relative convergence tolerance.
        atol: Absolute convergence tolerance.
        max_iterations: Maximum solver iterations.
        stopping_criterion: Convergence check strategy.
        m_max: FCG orthogonalization window.
        breakdown_tol: Breakdown detection threshold.
        normalize_system: Normalization applied to the test system.
        preconditioners: Classical preconditioner list (neural preconditioners are
            auto-generated from case experiments at runtime).
    """

    rtol: float = Field(default=DEFAULT_RTOL, gt=0.0)
    atol: float = Field(default=DEFAULT_ATOL, ge=0.0)
    max_iterations: int = Field(default=100, ge=1)
    stopping_criterion: Literal["residual_norm", "fixed_iterations"] = "residual_norm"
    m_max: int = Field(default=DEFAULT_M_MAX, ge=-1)
    breakdown_tol: float | None = Field(default=None, ge=0.0)
    normalize_system: Literal["none", "matrix", "rhs", "both", "diagonal", "spectral"] = "matrix"
    preconditioners: list[PreconditionerConfig] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComparisonRegistryEntry(BaseModel):
    """Comparison registry entry: data binding + optional methodology override.

    The test matrix and RHS are identified by dataset IDs from the case
    ``[[datasets]]`` registry.  Solver parameters and the base preconditioner
    list are inherited from ``[comparison_defaults]`` unless a ``method`` file
    is given to override them.

    Attributes:
        id: Stable comparison identifier.
        matrix_dataset: Dataset id (from ``[[datasets]]``) used as the test matrix.
        rhs_dataset: Dataset id (from ``[[datasets]]``) used as the test RHS.
        matrix_index: Matrix sample index when the dataset stores multiple matrices.
            Defaults to 0.
        rhs_index: RHS sample index when the dataset stores multiple RHS vectors.
            Defaults to 0.
        method: Optional path to a comparison TOML that overrides defaults.
        experiments: Optional experiment id filter; empty means all experiments.
        display_name: Optional human-facing label.
    """

    id: str = Field(..., min_length=1)
    matrix_dataset: str
    rhs_dataset: str
    matrix_index: int = 0
    rhs_index: int = 0
    method: Path | None = None
    experiments: list[str] = Field(default_factory=list)
    display_name: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("method", mode="before")
    @classmethod
    def _expand_method_path(cls, v: object, info: ValidationInfo) -> object:
        """Expand ${NEURALLS_*} placeholders in the method file path.

        Args:
            v: Raw value from config field.
            info: Pydantic validation info carrying context.

        Returns:
            Resolved absolute path string, or original value if not a string.
        """
        if v is None or not isinstance(v, str) or info.context is None:
            return v
        from neuralls.platform.config.context import ConfigContext, expand_config_path

        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))

    @property
    def effective_display_name(self) -> str:
        """Return the configured label or fall back to the id."""
        return resolve_display_name(self.id, self.display_name)


def _dedupe_ids(kind: str, entries: Sequence[RegistryEntry | ComparisonRegistryEntry]) -> None:
    """Reject duplicate registry ids with a focused error."""
    seen: set[str] = set()
    duplicates = sorted({entry.id for entry in entries if entry.id in seen or seen.add(entry.id)})
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"Duplicate {kind} ids in case config: {joined}.")


def _dedupe_experiment_ids(entries: list[ExperimentEntry]) -> None:
    """Reject duplicate experiment ids with a focused error."""
    seen: set[str] = set()
    duplicates = sorted({entry.id for entry in entries if entry.id in seen or seen.add(entry.id)})
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"Duplicate experiment ids in case config: {joined}.")


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


def _validate_comparison_dataset_refs(
    comparisons: list[ComparisonRegistryEntry],
    *,
    dataset_ids: set[str],
) -> None:
    """Reject comparison entries that reference undefined dataset ids."""
    for entry in comparisons:
        if entry.matrix_dataset not in dataset_ids:
            raise ValueError(
                f"Comparison '{entry.id}' references matrix_dataset "
                f"'{entry.matrix_dataset}', but [[datasets]] does not define it."
            )
        if entry.rhs_dataset not in dataset_ids:
            raise ValueError(
                f"Comparison '{entry.id}' references rhs_dataset "
                f"'{entry.rhs_dataset}', but [[datasets]] does not define it."
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


class CaseConfig(BaseModel):
    """Top-level case configuration.

    Attributes:
        datasets: Dataset registry entries (training data + comparison reference data).
        models: Model registry entries.
        comparisons: Comparison registry entries with data binding.
        experiments: Experiment entries referencing registry ids.
        run: Optional legacy direct-path entries.
        mlflow: MLflow topology config (tracking URI, etc.).
        names: MLflow experiment names for training and comparison.
        comparison_defaults: Shared solver params and preconditioner list applied to
            all comparisons unless overridden by a per-entry ``method`` file.
    """

    datasets: list[RegistryEntry] = Field(default_factory=list)
    models: list[RegistryEntry] = Field(default_factory=list)
    comparisons: list[ComparisonRegistryEntry] = Field(default_factory=list)
    experiments: list[ExperimentEntry] = Field(default_factory=list)
    run: list[RunEntry] = Field(default_factory=list)
    mlflow: MlflowTopologyConfig = Field(default_factory=MlflowTopologyConfig)
    names: ExperimentNamesConfig = Field(default_factory=ExperimentNamesConfig)
    comparison_defaults: ComparisonDefaults | None = None

    model_config = ConfigDict(extra="allow", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _auto_fill_ids_and_display_names(cls, data: object) -> object:
        """Auto-generate missing ids and display names before child validation.

        Args:
            data: Raw input dict (or other type, passed through unchanged).

        Returns:
            The mutated dict with any missing id/display_name fields filled in,
            or the original data if it is not a dict.

        Raises:
            ValueError: If any inferred or user-supplied id contains invalid characters,
                or if a display_name cannot be slugified into a valid id.
        """
        if not isinstance(data, dict):
            return data
        raw = cast("dict[str, object]", data)

        datasets: list[object] = cast("list[object]", raw.get("datasets", []))
        models: list[object] = cast("list[object]", raw.get("models", []))
        experiments: list[object] = cast("list[object]", raw.get("experiments", []))
        comparisons: list[object] = cast("list[object]", raw.get("comparisons", []))

        dataset_display = _build_display_lookup(datasets)
        model_display = _build_display_lookup(models)

        for exp in experiments:
            if not isinstance(exp, dict):
                continue
            exp_dict = cast("dict[str, object]", exp)
            dataset_id = str(exp_dict.get("dataset") or "")
            model_id = str(exp_dict.get("model") or "")
            raw_id = exp_dict.get("id")
            raw_dn = exp_dict.get("display_name")
            user_id = str(raw_id).strip() if isinstance(raw_id, str) else None
            user_id = user_id or None
            user_dn = str(raw_dn).strip() if isinstance(raw_dn, str) else None
            user_dn = user_dn or None

            exp_dict["id"] = _infer_experiment_id(dataset_id, model_id, user_id, user_dn)

            if not user_dn:
                exp_dict["display_name"] = _infer_experiment_display_name(
                    dataset_id, model_id, user_dn, dataset_display, model_display
                )

        for comp in comparisons:
            if not isinstance(comp, dict):
                continue
            comp_dict = cast("dict[str, object]", comp)
            matrix_id = str(comp_dict.get("matrix_dataset") or "")
            rhs_id = str(comp_dict.get("rhs_dataset") or "")
            raw_id = comp_dict.get("id")
            raw_dn = comp_dict.get("display_name")
            user_id = str(raw_id).strip() if isinstance(raw_id, str) else None
            user_id = user_id or None
            user_dn = str(raw_dn).strip() if isinstance(raw_dn, str) else None
            user_dn = user_dn or None

            comp_dict["id"] = _infer_comparison_id(matrix_id, rhs_id, user_id, user_dn)

            if not user_dn:
                comp_dict["display_name"] = _infer_comparison_display_name(
                    matrix_id,
                    rhs_id,
                    user_dn,
                    dataset_display,
                )

        return data

    @model_validator(mode="before")
    @classmethod
    def reject_unsupported_case_config_tables(cls, data: object) -> object:
        """Reject unsupported case-config table names."""
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        if "experiment" in raw:
            raise ValueError(
                "Unsupported '[[experiment]]' table. Use '[[experiments]]' entries in case config."
            )
        if "comparison_profiles" in raw:
            raise ValueError(
                "Unsupported 'comparison_profiles' table. "
                "Use [[comparisons]] entries in case config."
            )
        return raw

    @model_validator(mode="after")
    def validate_unique_ids(self) -> CaseConfig:
        """Reject duplicate ids and validate all cross-registry references."""
        _dedupe_ids("dataset registry", self.datasets)
        _dedupe_ids("model registry", self.models)
        _dedupe_ids("comparison registry", self.comparisons)
        _dedupe_experiment_ids(self.experiments)
        dataset_ids = _registry_ids(self.datasets)
        _validate_experiment_registry_refs(
            self.experiments,
            dataset_ids=dataset_ids,
            model_ids=_registry_ids(self.models),
        )
        _validate_comparison_experiment_filter_refs(
            self.comparisons,
            experiment_ids={e.id for e in self.experiments},
        )
        _validate_comparison_dataset_refs(self.comparisons, dataset_ids=dataset_ids)
        return self


ExperimentsConfig = CaseConfig
