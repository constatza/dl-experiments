"""Interfaces and data containers for generation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
from pydantic import BaseModel


@dataclass(frozen=True)
class GeneratedSamples:
    """Container for generated data."""

    matrix: np.ndarray
    rhs: np.ndarray | None
    solutions: np.ndarray | None
    residual_traces: Any | None = None
    error_traces: Any | None = None
    search_directions_traces: Any | None = None
    scale: Any | None = None


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
class GenerationStrategy(Protocol):
    """Protocol for all generation strategies.

    All strategies must implement:
    - name: str - Strategy identifier for registration
    - generate(matrix, cfg, archive) - Generate samples from input

    Trace strategies (residual_traces, residuals, gaussian_residuals, search_directions) also
    accept an optional single_rhs parameter, but this is handled via duck-typing
    at runtime by the dispatcher.

    This unified protocol uses structural typing (duck-typing) rather than
    explicit inheritance, allowing strategies to have different signatures while
    sharing a common interface.
    """

    name: str

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

        Note:
            Trace strategies also accept optional single_rhs parameter,
            but this is not part of the protocol signature (duck-typing).
        """
        ...
