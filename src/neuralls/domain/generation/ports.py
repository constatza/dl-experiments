"""Ports used by generation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from numpy.typing import NDArray

from .payloads import GeneratedDatasetPayload


class TracingSolverPort(Protocol):
    """Callable protocol for traced linear solves."""

    def __call__(
        self,
        A: NDArray,
        b: NDArray,
        x0: NDArray,
        *,
        maxiter: int,
        rtol: float,
        atol: float,
    ) -> tuple[NDArray, Any]: ...


class DatasetWriterPort(Protocol):
    """Storage port for persisting generated datasets."""

    def write_dataset(
        self,
        dataset_dir: Path,
        payload: GeneratedDatasetPayload,
    ) -> None: ...
