"""Pydantic models for individual data generation strategy configurations."""

from __future__ import annotations


from pydantic import BaseModel, ConfigDict, ConfigDict, Field

from ..constants import (
    DEFAULT_KRYLOV_ITERATIONS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RESIDUAL_TRACE_ITERS,
    DEFAULT_SHUFFLE,
    MIN_TOLERANCE,
    MAX_ITERATIONS_UPPER_LIMIT,
    EIGENVECTOR_SELECT_SMALLEST,
    EigenvectorSelectionMode,
)

class BaseStrategyConfig(BaseModel):
    samples: int = Field(
        ...,
        description="Number of samples to generate for this strategy. (0=skip, -1=all, >0=exact count)",
        ge=-1, # samples can be -1 for all
    )
    seed: int | None = Field(
        DEFAULT_RANDOM_SEED, description="Random seed for reproducibility."
    )
    shuffle: bool = Field(DEFAULT_SHUFFLE, description="Whether to shuffle the generated samples.")

    model_config = ConfigDict(
        extra="forbid",  # Forbid extra fields to catch typos
        frozen=True,  # Make configs immutable
    )


class BaseEigenvectorConfig(BaseStrategyConfig):
    which: EigenvectorSelectionMode = Field(
        EIGENVECTOR_SELECT_SMALLEST, description="Which eigenvalues to compute."
    )
    include_eigenvectors: bool = Field(
        True, description="Whether to include eigenvectors in the generated solutions."
    )
    num_eigenvectors: int = Field(
        1, description="Number of eigenvectors to include.", ge=1
    )


class EigenvectorForwardConfig(BaseEigenvectorConfig):
    """Configuration for EigenvectorForwardStrategy."""
    pass


class EigenvectorInverseConfig(BaseEigenvectorConfig):
    """Configuration for EigenvectorInverseStrategy."""
    pass


class KrylovConfig(BaseStrategyConfig):
    krylov_iters: int = Field(
        DEFAULT_KRYLOV_ITERATIONS,
        description="Number of Krylov iterations to perform.",
        ge=1,
    )


class RandomNormalConfig(BaseStrategyConfig):
    target_rhs_scale: float = Field(
        1.0,
        description="Target scale for the generated RHS vectors (Euclidean norm).",
        gt=0.0,
    )


class ResidualErrorConfig(BaseStrategyConfig):
    residual_iters: int = Field(
        DEFAULT_RESIDUAL_TRACE_ITERS,
        description="Number of residual iterations to trace.",
        ge=0,
    )
    archive_solutions: bool = Field(
        False,
        description="Whether to archive intermediate solutions for each iteration step.",
    )
    archive_rhs: bool = Field(
        False,
        description="Whether to archive intermediate RHS vectors for each iteration step.",
    )


class ResidualTraceConfig(BaseStrategyConfig):
    residual_iters: int = Field(
        DEFAULT_RESIDUAL_TRACE_ITERS,
        description="Number of residual iterations to trace.",
        ge=0,
    )
    archive_solutions: bool = Field(
        False,
        description="Whether to archive intermediate solutions for each iteration step.",
    )
    archive_rhs: bool = Field(
        False,
        description="Whether to archive intermediate RHS vectors for each iteration step.",
    )


class RhsArchiveConfig(BaseStrategyConfig):
    rhs_glob: str = Field(
        ..., description="Glob pattern for RHS files to load. Overrides source.rhs_path."
    )
    solve_systems: bool = Field(
        True, description="Whether to solve the systems (A x = b) to get solutions."
    )
    cg_tolerance: float = Field(
        MIN_TOLERANCE, description="CG convergence tolerance.", ge=MIN_TOLERANCE
    )
    cg_max_iters: int = Field(
        MAX_ITERATIONS_UPPER_LIMIT,
        description="Maximum CG iterations.",
        ge=1,
        le=MAX_ITERATIONS_UPPER_LIMIT,
    )


class SolutionArchiveConfig(BaseStrategyConfig):
    solutions_glob: str = Field(
        ..., description="Glob pattern for solution files to load. Overrides source.solutions_path."
    )


class MixedStrategyConfig(BaseModel):
    """Configuration for a mix of strategies. (Not directly used by `generate` methods)"""
    pass

class GenerationConfig(BaseModel):
    """Overall configuration for data generation."""
    # This represents the structure of the [generation] section in the config
    # Add fields as needed to match the actual data generation config structure.
    # For now, it's a placeholder.
    pass