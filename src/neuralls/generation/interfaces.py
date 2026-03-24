"""Interfaces and data containers for generation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from ..normalization import ErrorTraceSamples, ResidualTraceSamples


@dataclass(frozen=True)
class GeneratedSamples:
    """Container for generated data."""

    matrix: np.ndarray
    rhs: np.ndarray | None
    solutions: np.ndarray | None
    residual_traces: ResidualTraceSamples | None = None
    error_traces: ErrorTraceSamples | None = None


@dataclass(frozen=True)
class ArchiveData:
    """Container for pre-computed solutions and RHS vectors.

    Used when loading from solution archives instead of generating random data.

    Attributes:
        solutions: Solution vectors, shape (N, n)
        rhs_vectors: Optional pre-computed RHS vectors, shape (N, n)
    """

    solutions: np.ndarray
    rhs_vectors: np.ndarray | None = None


@runtime_checkable
class MatrixGenerationStrategy(Protocol):
    """Protocol for strategies that generate from the matrix alone."""

    name: str
    ConfigType: object

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict[str, Any],
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate samples from matrix.

        Args:
            matrix: System matrix, shape (n, n)
            cfg: Strategy configuration dictionary (validated by strategy)
            archive: Optional pre-computed archive data

        Returns:
            GeneratedSamples with matrix, rhs, solutions, and optional traces
        """
        ...


@runtime_checkable
class SingleRhsGenerationStrategy(Protocol):
    """Protocol for strategies that support explicit single-RHS generation."""

    name: str
    ConfigType: object

    def generate(
        self,
        matrix: np.ndarray,
        *,
        cfg: dict[str, Any],
        single_rhs: np.ndarray | None = None,
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate samples from matrix with an optional shared RHS."""
        ...
