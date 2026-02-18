"""Preconditioner configuration models.

These Pydantic models validate preconditioner configurations from TOML files.
They support both factory creation and scheduling/comparison concerns.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Any, Annotated
from pathlib import Path

from pydantic import TypeAdapter, BeforeValidator, BaseModel, ConfigDict, Field


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
