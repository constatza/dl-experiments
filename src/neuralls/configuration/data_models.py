"""Pydantic models for data configuration validation.

This module provides Pydantic models for validating data config TOML structure.
These models validate the structure of data generation/collection configuration files.
"""

from __future__ import annotations

from typing import Literal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class FlowConfig(BaseModel):
    """Validates [flow] section from data config.

    This section typically contains flow-level metadata.
    Usually empty but kept for future extensibility.
    """

    id: str | None = Field(
        default=None,
        description="Flow identifier",
    )
    dataset: str | None = Field(
        default=None,
        description="Dataset name or identifier",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceConfig(BaseModel):
    """Validates [source] section from data config.

    Specifies source data locations for matrix and solution vectors.
    """

    type: Literal["rhs_archive", "generated", "solution_archive"] | None = Field(
        default=None,
        description="Source data type",
    )
    case_path: str | None = Field(
        default=None,
        description="Path to case directory",
    )
    matrix_file: str | None = Field(
        default=None,
        description="Matrix filename (relative to case_path)",
    )
    matrix_path: str | None = Field(
        default=None,
        description="Full path to matrix file",
    )
    rhs_path: str | None = Field(
        default=None,
        description="Path to RHS vector files",
    )
    rhs_pattern: str | None = Field(
        default=None,
        description="Glob pattern for RHS files",
    )
    solutions_path: str | None = Field(
        default=None,
        description="Path or glob pattern for solution vectors",
    )
    sample_id_regex: str | None = Field(
        default=None,
        description="Regex used to extract sample IDs from glob filenames for source pairing",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class StrategyConfig(BaseModel):
    """Validates [[generation.strategy]] entry from data config.

    Each strategy defines a data generation approach (e.g., cg_residual, solution_archive).
    """

    name: str = Field(
        ...,
        description="Strategy name (cg_residual, solution_archive, etc.)",
    )
    samples: int = Field(
        default=0,
        description="Number of samples to generate (-1 for all available)",
    )
    krylov_iters: int | None = Field(
        default=None,
        description="Number of Krylov iterations for krylov-based strategies",
        ge=1,
    )
    residual_iters: int | None = Field(
        default=None,
        description="Number of residual iterations for residual-based strategies",
        ge=1,
    )
    solutions_glob: str | None = Field(
        default=None,
        description="Glob pattern for solution vector files",
    )
    rhs_glob: str | None = Field(
        default=None,
        description="Glob pattern for RHS vector files",
    )

    model_config = ConfigDict(
        extra="allow", frozen=True
    )  # Allow strategy-specific parameters


class GenerationConfig(BaseModel):
    """Validates [generation] section from data config.

    Controls data generation/collection behavior and normalization.
    """

    normalize: str | bool = Field(
        default="matrix",
        description="Normalization strategy (matrix, rhs, both, none, or False)",
    )
    shuffle: bool = Field(
        default=True,
        description="Shuffle generated samples",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    num_samples: int | None = Field(
        default=None,
        description="Total number of samples (legacy, use strategy.samples)",
        ge=1,
    )
    strategy: list[StrategyConfig] = Field(
        default_factory=list,
        description="List of generation strategies",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class OutputConfig(BaseModel):
    """Validates [output] section from data config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_dir: Path | None = Field(
        default=None,
        description="Output directory for generated data",
    )
    dataset_format: Literal["npy_coo"] = Field(
        default="npy_coo",
        description="Dataset storage format",
    )
    matrix_codec: Literal["coo"] = Field(
        default="coo",
        description="Sparse matrix codec",
    )
    matrix_replication: Literal["duplicate_per_sample"] = Field(
        default="duplicate_per_sample",
        description="Matrix replication policy",
    )
    dtype: Literal["float64"] = Field(
        default="float64",
        description="Numeric dtype for persisted arrays",
    )


class DataTestConfig(BaseModel):
    """Validates [test] section from data config.

    Optional section specifying test data locations.
    """

    solutions_path: str | None = Field(
        default=None,
        description="Path or glob pattern for test solution vectors",
    )
    rhs_path: str | None = Field(
        default=None,
        description="Path or glob pattern for test RHS vectors",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfigFile(BaseModel):
    """Complete data config TOML structure with validation.

    This model validates the top-level sections of a data configuration file
    used for data generation and collection workflows.
    """

    flow: FlowConfig = Field(
        default_factory=FlowConfig,
        description="Flow metadata",
    )
    source: SourceConfig = Field(
        default_factory=SourceConfig,
        description="Source data locations",
    )
    generation: GenerationConfig = Field(
        default_factory=GenerationConfig,
        description="Data generation configuration",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output paths configuration",
    )
    test: DataTestConfig = Field(
        default_factory=DataTestConfig,
        description="Test data configuration",
    )

    model_config = ConfigDict(
        extra="allow", frozen=True
    )  # Allow additional sections for future extension
