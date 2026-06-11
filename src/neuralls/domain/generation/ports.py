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


class SparseAccumulatorPort(Protocol):
    """Protocol for accumulating sparse matrix samples during dataset generation."""

    def append_dense_matrix(self, matrix: NDArray, repeats: int) -> None: ...

    def append_sparse_components(
        self,
        *,
        indices: NDArray,
        values: NDArray,
        size: tuple[int, int],
        repeats: int,
    ) -> None: ...

    def build_arrays(self) -> tuple[NDArray, NDArray, NDArray, tuple[int, int]]: ...


class ZarrAccumulatorPort(Protocol):
    """Streaming zarr-backed accumulator port.

    Implementations write each matrix sample directly to disk via ZarrPackWriter,
    preventing in-memory OOM accumulation. Call finalize() once all samples have
    been appended.
    """

    def append_dense_matrix(self, matrix: NDArray, repeats: int) -> None: ...

    def append_sparse_components(
        self,
        *,
        indices: NDArray,
        values: NDArray,
        size: tuple[int, int],
        repeats: int,
    ) -> None: ...

    def finalize(self) -> Path:
        """Close the writer and return the on-disk pack directory path."""
        ...

    @property
    def matrix_size(self) -> tuple[int, int] | None: ...
