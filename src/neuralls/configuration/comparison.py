"""Solver comparison configuration models.

These Pydantic models validate solver comparison TOML configs.
Focused on solver parameters, not preconditioner or data generation.
"""

from __future__ import annotations

from typing import Literal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from neuralls.constants import (
    DEFAULT_RTOL,
    DEFAULT_ATOL,
    DEFAULT_M_MAX,
)
from neuralls.configuration.preconditioner import PreconditionerConfig


class GeneralSolverConfig(BaseModel):
    """Validates [general] section from solver comparison TOML."""

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
    m_max: int = Field(
        default=DEFAULT_M_MAX,
        description="FCG(m) orthogonalization window size. Use -1 for full orthogonalization.",
        ge=-1,
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


class ComparisonConfig(BaseModel):
    """Top-level solver comparison configuration file structure."""

    general: GeneralSolverConfig = Field(
        ...,
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
