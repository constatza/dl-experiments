"""Pydantic models for solver configuration validation.

These models validate TOML solver configs before conversion to runtime dataclasses.
Separation: Pydantic models (validation) → frozen dataclasses (runtime use).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Any, Annotated
from pathlib import Path


from neuralls.constants import (
    DEFAULT_RTOL,
    DEFAULT_ATOL,
    REORTHOG_STRICT_THRESHOLD,
)


from pydantic import TypeAdapter, BeforeValidator, BaseModel, ConfigDict, Field


class GeneralSolverConfig(BaseModel):
    """Validates [general] section from solver TOML."""

    rtol: float = Field(
        default=DEFAULT_RTOL,
        description="Relative tolerance for convergence",
        gt=0.0,
    )
    atol: float = Field(
        default=DEFAULT_ATOL,
        description="Absolute tolerance for convergence",
        ge=0.0,
    )
    max_iterations: int = Field(
        default=100,
        description="Maximum solver iterations",
        ge=1,
    )
    stopping_criterion: Literal["residual_norm", "fixed_iterations"] = Field(
        default="residual_norm",
        description="Stopping criterion for solver",
    )
    normalize_system: str | bool = Field(
        default="matrix",
        description="System normalization strategy",
    )
    matrix_path: str | None = Field(
        default=None,
        description="Path to matrix file (NPZ)",
    )
    rhs_path: str | None = Field(
        default=None,
        description="Path to RHS file (NPZ)",
    )
    breakdown_tol: float | None = Field(
        default=None,
        description="Breakdown tolerance for CG denominator checks",
        ge=0.0,
    )
    reorthogonalize: Literal["none", "full", "partial", "selective"] = Field(
        default="full",
        description="Reorthogonalization strategy",
    )
    reorthog_window: int = Field(
        default=10,
        description="Window size for partial reorthogonalization",
        ge=1,
    )
    reorthog_threshold: float = Field(
        default=REORTHOG_STRICT_THRESHOLD,
        description="Threshold for selective reorthogonalization",
        ge=0.0,
        le=1.0,
    )
    output_root: Path = Field(
        ...,
        description="Base directory for comparison outputs",
    )

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for extensibility
        frozen=True,
    )


class DataGenerationConfig(BaseModel):
    """Validates [data_generation] section (optional)."""

    normalize: str | bool = Field(
        default="matrix",
        description="Normalization strategy for data generation",
    )

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class PreconditionerType(StrEnum):
    """Preconditioner types."""

    NONE = "none"
    IDENTITY = "identity"
    JACOBI = "jacobi"
    ILU = "ilu"
    NEURAL = "neural"


def _normalize_null(data: dict) -> Any:
    # Handle case where data is a Dict
    if isinstance(data, dict):
        # Avoid side effects by copying
        # (If we modify 'data' directly, it might surprise other parts of the code)
        data = data.copy()

        # The fix: Map synonyms to the strict canonical name
        if data.get("type") in ("none", "null"):
            data["type"] = PreconditionerType.IDENTITY
        return data

    return data


class BasePreconditionerConfig(BaseModel):
    """Shared fields for all preconditioners."""

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
        extra="ignore",  # Allow extra fields from SolverSpec conversion
        frozen=True,
    )


class StandardPreconditionerConfig(BasePreconditionerConfig):
    """Non parametric, static preconditioners."""

    type: Literal[
        PreconditionerType.IDENTITY,
        PreconditionerType.JACOBI,
        PreconditionerType.ILU,
    ]


class NeuralPreconditionerConfig(BasePreconditionerConfig):
    """Neural preconditioner config."""

    type: Literal[PreconditionerType.NEURAL] = PreconditionerType.NEURAL
    checkpoint_path: Path
    config_path: Path | None = None
    data_config_path: Path | None = None
    limit_iters: int = Field(
        default=-1, description="Iterations to apply; -1 means unlimited."
    )


# INNER TYPE: The strict Discriminated Union.
# This assumes the data is ALREADY clean.
_StrictPreconditionerConfig = Annotated[
    StandardPreconditionerConfig | NeuralPreconditionerConfig,
    Field(discriminator="type"),
]

# OUTER TYPE: The public interface.
# This applies the Validator first, THEN passes the result to the Inner Type.
PreconditionerConfig = Annotated[
    _StrictPreconditionerConfig, BeforeValidator(_normalize_null)
]

parse_preconditioner_config = TypeAdapter(PreconditionerConfig).validate_python


class SolverConfigFile(BaseModel):
    """Top-level solver configuration file structure."""

    general: GeneralSolverConfig = Field(
        default_factory=GeneralSolverConfig,
        description="Global solver parameters",
    )
    solvers: list[PreconditionerConfig] = Field(
        default_factory=list,
        description="List of solver/preconditioner configurations",
    )
    data_generation: DataGenerationConfig = Field(
        default_factory=DataGenerationConfig,
        description="Data generation parameters (optional)",
    )

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )
