"""Interfaces and data containers for generation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class GeneratedSamples:
    """Container for generated data."""

    matrix: np.ndarray
    rhs: np.ndarray | None
    solutions: np.ndarray | None
    residual_traces: Any | None = None
    error_traces: Any | None = None
    scale: Any | None = None


class IDataGenerationStrategy(Protocol):
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
        cfg: dict[str, Any],
    ) -> GeneratedSamples:
        """Generate samples from the given inputs."""
        ...


class IMatrixOnlyGenerationStrategy(IDataGenerationStrategy, Protocol):
    """Marker for strategies that do not require an RHS."""

    def requires_rhs(self) -> bool:
        return False
