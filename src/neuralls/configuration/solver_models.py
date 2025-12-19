"""Pydantic models for solver configuration validation.

These models validate TOML solver configs before conversion to runtime dataclasses.
Separation: Pydantic models (validation) → frozen dataclasses (runtime use).
"""

from __future__ import annotations

from typing import Any, Literal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..constants import (
    DEFAULT_RTOL,
    DEFAULT_ATOL,
    REORTHOG_STRICT_THRESHOLD,
)


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


class SolverSpecConfig(BaseModel):
    """Validates individual [[solvers]] entries from solver TOML."""

    name: str = Field(
        ...,
        description="Display name for this solver/preconditioner",
    )
    type: str = Field(
        ...,
        description="Solver/preconditioner type (none, jacobi, ilu, neural, etc.)",
    )
    # Preconditioner control parameters
    limit_iters: int = Field(
        default=0,
        description="Apply for first N iterations (0 = unlimited)",
        ge=0,
    )
    apply_every: int = Field(
        default=1,
        description="Apply preconditioner every K iterations",
        ge=1,
    )
    first_n: int | None = Field(
        default=None,
        description="Only apply for first N total iterations",
        ge=1,
    )
    fallback: Literal["identity", "jacobi", "ilu"] = Field(
        default="identity",
        description="Fallback preconditioner after limit_iters",
    )
    # Neural-specific: checkpoint resolution via EITHER explicit path OR experiment reference
    checkpoint_path: Path | None = Field(
        default=None,
        description="Explicit path to neural network checkpoint (for neural type)",
    )
    experiment: str | None = Field(
        default=None,
        description="Reference to experiment ID from experiments.toml (resolves checkpoint at runtime)",
    )

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for different solver types
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


class SolverConfigFile(BaseModel):
    """Top-level solver configuration file structure."""

    general: GeneralSolverConfig = Field(
        default_factory=GeneralSolverConfig,
        description="Global solver parameters",
    )
    solvers: list[SolverSpecConfig] = Field(
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
