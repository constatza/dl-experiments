"""Pydantic models for individual data generation strategy configurations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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


class SolveConfig(BaseModel):
    """Configuration for solving linear systems.

    Provides unified solving configuration across all inverse strategies
    (strategies that solve x = A^-1 @ b instead of computing b = A @ x).

    Attributes:
        method: Solving method ("direct" or "cg")
            - "direct": Uses scipy.linalg.solve (O(n³), exact modulo floating point)
            - "cg": Uses scipy.sparse.linalg.cg (O(n² * iters), iterative)
        rtol: Relative tolerance for CG solver (ignored for direct)
        atol: Absolute tolerance for CG solver (ignored for direct)
        max_iters: Maximum CG iterations (ignored for direct)
        assume_pos_def: Assume matrix is positive-definite (direct solve only)

    Examples:
        >>> # Direct solve (default)
        >>> SolveConfig(method="direct", assume_pos_def=True)

        >>> # CG solve with custom tolerances
        >>> SolveConfig(method="cg", rtol=1e-10, atol=1e-12, max_iters=1000)
    """

    method: Literal["direct", "cg"] = Field(
        "direct",
        description="Solving method: 'direct' (O(n³) exact) or 'cg' (O(n²*iters) iterative)",
    )
    rtol: float = Field(
        MIN_TOLERANCE,
        description="Relative tolerance for CG solver (ignored for direct solve)",
        ge=0.0,
    )
    atol: float = Field(
        0.0,
        description="Absolute tolerance for CG solver (ignored for direct solve)",
        ge=0.0,
    )
    max_iters: int = Field(
        MAX_ITERATIONS_UPPER_LIMIT,
        description="Maximum CG iterations (ignored for direct solve)",
        ge=1,
        le=MAX_ITERATIONS_UPPER_LIMIT,
    )
    assume_pos_def: bool = Field(
        True,
        description="Assume matrix is positive-definite (for direct solve assume_a='pos')",
    )

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
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
    """Configuration for EigenvectorInverseStrategy.

    Generates RHS vectors from eigenvector combinations, then solves for solutions.
    Supports both direct solve (O(n³)) and CG solve (O(n² * iters)).
    """

    solve_config: SolveConfig = Field(
        default_factory=lambda: SolveConfig(  # pyright: ignore[reportCallIssue]
            method="direct",
            rtol=MIN_TOLERANCE,
            atol=0.0,
            max_iters=MAX_ITERATIONS_UPPER_LIMIT,
            assume_pos_def=True,
        ),
        description="Configuration for solving linear systems x = A^-1 @ b",
    )


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


class BaseTraceConfig(BaseStrategyConfig):
    """Shared configuration for CG trace-collection strategies."""

    cg_iters: int = Field(
        DEFAULT_RESIDUAL_TRACE_ITERS,
        description="Number of CG iterations to trace.",
        ge=1,
    )
    solutions_glob: str | None = Field(
        None,
        description="Glob pattern for archive solution files to seed generation.",
    )
    archive_solutions: bool = Field(
        False,
        description="Whether to archive intermediate solutions for each iteration step.",
    )
    archive_rhs: bool = Field(
        False,
        description="Whether to archive intermediate RHS vectors for each iteration step.",
    )
    every_n: int = Field(
        1,
        description="Keep one trace vector every N CG iterations.",
        ge=1,
    )


class ResidualErrorConfig(BaseTraceConfig):
    """Configuration for ResidualErrorStrategy (cg_residual_error)."""


class ResidualTraceConfig(BaseTraceConfig):
    """Configuration for ResidualTraceStrategy (cg_residual)."""


class SearchDirectionsConfig(BaseStrategyConfig):
    """Configuration for SearchDirectionsStrategy.

    Collects (A @ p_k, p_k) pairs from CG iterations for training neural preconditioners.
    Training mapping: NN(A @ p_k) ≈ p_k, so NN ≈ A^{-1}
    """
    cg_iters: int = Field(
        DEFAULT_RESIDUAL_TRACE_ITERS,
        description="Number of CG iterations to collect search direction pairs.",
        ge=1,
    )
    every_n: int = Field(
        1,
        description="Keep one trace vector every N CG iterations.",
        ge=1,
    )


class RhsArchiveConfig(BaseStrategyConfig):
    rhs_glob: str = Field(
        ..., description="Glob pattern for RHS files to load. Overrides source.rhs_path."
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


class ValidatedArchiveConfig(BaseStrategyConfig):
    """Configuration for ValidatedArchiveStrategy.

    Loads both solutions and RHS from archives, then verifies A @ x = b consistency.
    """

    solutions_glob: str = Field(
        ..., description="Glob pattern for solution files to load."
    )
    rhs_glob: str = Field(
        ..., description="Glob pattern for RHS files to load."
    )
    verification_tolerance: float = Field(
        1e-10,
        description="Maximum relative residual ||A@x - b|| / ||b|| to accept as valid.",
        ge=0.0,
    )
    fail_on_invalid: bool = Field(
        True,
        description="Whether to raise error if any pair fails verification.",
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