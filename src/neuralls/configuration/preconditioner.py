"""Preconditioner configuration models.

These Pydantic models validate preconditioner configurations from TOML files.
They support both factory creation and scheduling/comparison concerns.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Any, Annotated
from pathlib import Path

from pydantic import TypeAdapter, BeforeValidator, BaseModel, ConfigDict, Field, model_validator


class PreconditionerType(StrEnum):
    """Preconditioner types."""

    NONE = "none"
    IDENTITY = "identity"
    JACOBI = "jacobi"
    ILU = "ilu"
    IC0 = "ic0"
    ICHOLESKY = "icholesky"
    NEURAL = "neural"


def _normalize_null(data: dict) -> Any:
    """Normalize 'none'/'null' to 'identity' for backward compatibility."""
    if isinstance(data, dict):
        data = data.copy()
        if data.get("type") in ("none", "null"):
            data["type"] = PreconditionerType.IDENTITY
        return data
    return data


class RegisteredModelRefConfig(BaseModel):
    """Reference to a model registered in the MLflow Model Registry.

    Attributes:
        source: Discriminator field, always "registered".
        name: Registered model name.
        alias: Model alias (e.g. ``"@solutions"``); ``@`` prefix is stripped.
        version: Explicit model version number.
        latest: If True, select the latest version.
    """

    source: Literal["registered"] = "registered"
    name: str = Field(..., min_length=1)
    alias: str | None = None
    version: int | None = Field(default=None, ge=1)
    latest: bool | None = None
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_selector(self) -> RegisteredModelRefConfig:
        """Validate that exactly one selector is provided.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If not exactly one of alias, version, or latest=True is set.
        """
        selectors = [self.alias is not None, self.version is not None, self.latest is True]
        if sum(selectors) != 1:
            raise ValueError("Requires exactly one selector: alias, version, or latest=true.")
        return self


class LoggedModelRefConfig(BaseModel):
    """Reference to a model logged within an MLflow run.

    Attributes:
        source: Discriminator field, always "logged".
        run_id: Explicit MLflow run ID to reference.
        latest: If True, find the latest matching model via filters.
        model_name: Filter by logged model name.
        experiment_name: Filter by experiment name.
        experiment_id: Filter by experiment ID.
        run_name: Filter by run name.
        artifact_path: Artifact path within the run (default: "model").
        tags: Optional tag filters.
    """

    source: Literal["logged"] = "logged"
    run_id: str | None = Field(default=None, min_length=1)
    latest: bool | None = None
    model_name: str | None = None
    experiment_name: str | None = None
    experiment_id: str | None = None
    run_name: str | None = None
    artifact_path: str = "model"
    tags: dict[str, str] | None = None
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_selector(self) -> LoggedModelRefConfig:
        """Validate that exactly one selection mode is active.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If not exactly one of run_id or latest=True is set.
        """
        selectors = [self.run_id is not None, self.latest is True]
        if sum(selectors) != 1:
            raise ValueError("Requires exactly one selector: run_id or latest=true.")
        return self

    @model_validator(mode="after")
    def validate_latest_filters(self) -> LoggedModelRefConfig:
        """Validate that latest=True is accompanied by at least one filter.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If latest=True but no filter fields are set.
        """
        if self.latest is not True:
            return self
        filters = [self.model_name, self.experiment_name, self.experiment_id, self.run_name, self.tags]
        if not any(f is not None for f in filters):
            raise ValueError(
                "latest=true requires at least one filter "
                "(model_name, experiment_name, experiment_id, run_name, or tags)."
            )
        return self


ModelRefConfig = Annotated[
    RegisteredModelRefConfig | LoggedModelRefConfig,
    Field(discriminator="source"),
]


class BasePreconditionerConfig(BaseModel):
    """Shared fields for all preconditioners.

    Includes both factory fields (name, type) and scheduling fields
    (limit_iters, fallback) for convenience in comparison workflows.
    """

    name: str
    type: PreconditionerType
    limit_iters: int = Field(
        default=-1, description="Iterations to apply; -1 means unlimited."
    )
    fallback: PreconditionerType = Field(
        default=PreconditionerType.IDENTITY,
        description="Fallback preconditioner type when limited.",
    )

    model_config = ConfigDict(
        extra="ignore",  # Allow extra fields from conversion
        frozen=True,
    )


class StandardPreconditionerConfig(BasePreconditionerConfig):
    """Non-parametric, static preconditioners (identity, jacobi, ilu, icholesky)."""

    type: Literal[  # type: ignore[assignment]
        PreconditionerType.IDENTITY,
        PreconditionerType.JACOBI,
        PreconditionerType.ILU,
        PreconditionerType.ICHOLESKY,
    ]


class IC0PreconditionerConfig(BasePreconditionerConfig):
    """IC(0) preconditioner configuration with threshold parameter."""

    type: Literal[PreconditionerType.IC0] = PreconditionerType.IC0  # type: ignore[assignment]
    threshold: float = Field(
        default=1e-14,
        description="Drop tolerance - entries with |value| < threshold are treated as zeros"
    )


class NeuralPreconditionerConfig(BasePreconditionerConfig):
    """Neural preconditioner configuration."""

    type: Literal[PreconditionerType.NEURAL] = PreconditionerType.NEURAL  # type: ignore[assignment]
    checkpoint_path: Path | None = None
    experiment: str | None = None
    config_path: Path | None = None
    data_config_path: Path | None = None
    limit_iters: int = Field(
        default=-1, description="Iterations to apply; -1 means unlimited."
    )
    model_ref: ModelRefConfig | None = None
    resolved_checkpoint_path: Path | None = None


# Discriminated union with normalization
_StrictPreconditionerConfig = Annotated[
    StandardPreconditionerConfig | IC0PreconditionerConfig | NeuralPreconditionerConfig,
    Field(discriminator="type"),
]

PreconditionerConfig = Annotated[
    _StrictPreconditionerConfig,
    BeforeValidator(_normalize_null),
]

parse_preconditioner_config = TypeAdapter(PreconditionerConfig).validate_python
