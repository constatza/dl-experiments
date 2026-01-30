"""Interfaces and data containers for generation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

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


ConfigType = TypeVar("ConfigType", bound=BaseModel)


class IDataGenerationStrategy(Protocol[ConfigType]):
    """Strategy contract for data generation."""

    name: str

    def requires_rhs(self) -> bool:
        """Return True if strategy needs an RHS input."""
        ...

    def generate(
        self,
        matrix: np.ndarray,
        rhs: np.ndarray | None,
        *,
        cfg: ConfigType,
        archive: ArchiveData | None = None,
    ) -> GeneratedSamples:
        """Generate samples from the given inputs."""
        ...


class IMatrixOnlyGenerationStrategy(IDataGenerationStrategy[ConfigType], Protocol):
    """Marker for strategies that do not require an RHS."""

    def requires_rhs(self) -> bool:
        return False
