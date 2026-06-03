"""Streaming source readers for matrix and vector sample ingestion.

This module keeps source discovery and per-sample loading decoupled from
generation strategy logic. It supports:
- single .txt matrix/vector files
- single .npy matrix/vector files (including stacked samples via mmap)
- glob patterns of .txt/.npy files (one sample per file)
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

_GLOB_CHARS = ("*", "?", "[", "]")
_DEFAULT_SAMPLE_ID_REGEX = r"(\d+)(?!.*\d)"


class EnumerateBy(StrEnum):
    """Criterion for assigning sequential IDs to glob-matched files.

    Use when filenames carry no natural integer ID (e.g. parameter-encoded names
    like ``E1_3000_E2_78000_matrix.txt``).  Files are sorted by the chosen
    criterion and assigned sequential IDs 0, 1, 2, …
    """

    NAME = "name"
    CTIME = "ctime"
    MTIME = "mtime"


def _sort_key_for(path: Path, by: EnumerateBy) -> float | str:
    """Return the sort key for *path* under the given *by* strategy."""
    match by:
        case EnumerateBy.NAME:
            return path.name
        case EnumerateBy.CTIME:
            return path.stat().st_ctime
        case EnumerateBy.MTIME:
            return path.stat().st_mtime


def _enumerate_files(paths: Sequence[Path], by: EnumerateBy) -> dict[int, Path]:
    """Assign sequential IDs to *paths* sorted by *by*, returning ``{id: path}``."""
    return {i: p for i, p in enumerate(sorted(paths, key=lambda p: _sort_key_for(p, by)))}


def _is_glob_expression(expr: str) -> bool:
    """Return True when the path expression contains glob meta characters."""
    return any(char in expr for char in _GLOB_CHARS)


def _normalize_vector(array: np.ndarray, source: Path) -> np.ndarray:
    """Normalize vector shape to 1D float64."""
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
        return arr.reshape(-1)
    raise ValueError(f"Expected vector from {source}, got shape {arr.shape}")


def _extract_sample_id(path: Path, pattern: re.Pattern[str]) -> int:
    """Extract integer sample id from filename stem."""
    match = pattern.search(path.stem)
    if match is None:
        raise ValueError(
            f"Could not extract sample id from '{path.name}' using regex '{pattern.pattern}'."
        )
    return int(match.group(1))


def _dense_to_sparse_components(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Convert dense matrix to COO payload components."""
    rows, cols = np.nonzero(matrix)
    values = np.asarray(matrix[rows, cols], dtype=np.float64)
    indices = np.vstack((rows, cols)).astype(np.int64, copy=False)
    return indices, values, (int(matrix.shape[0]), int(matrix.shape[1]))


@dataclass(frozen=True)
class DenseMatrixSample:
    """One dense matrix sample."""

    sample_id: int
    matrix: np.ndarray


@dataclass(frozen=True)
class SparseMatrixSample:
    """One sparse matrix sample as COO payload arrays."""

    sample_id: int
    indices: np.ndarray
    values: np.ndarray
    size: tuple[int, int]


@dataclass(frozen=True)
class VectorSample:
    """One RHS/solution vector sample."""

    sample_id: int
    vector: np.ndarray


@dataclass(frozen=True)
class SystemBinding:
    """ID-level binding across matrix/rhs/solution sources."""

    sample_id: int
    matrix_sample_id: int
    rhs_sample_id: int | None = None
    solution_sample_id: int | None = None


@runtime_checkable
class MatrixSampleStream(Protocol):
    """Protocol for lazily loading matrix samples."""

    @property
    def sample_ids(self) -> tuple[int, ...]:
        """Available sample IDs."""
        ...

    def load_dense_sample(self, sample_id: int) -> DenseMatrixSample:
        """Load one matrix sample in dense float64 format."""
        ...

    def load_sparse_sample(self, sample_id: int) -> SparseMatrixSample:
        """Load one matrix sample as sparse COO components."""
        ...

    def iter_dense_samples(self) -> Iterator[DenseMatrixSample]:
        """Iterate all dense matrix samples."""
        ...

    def iter_sparse_samples(self) -> Iterator[SparseMatrixSample]:
        """Iterate all sparse matrix samples."""
        ...


@runtime_checkable
class VectorSampleStream(Protocol):
    """Protocol for lazily loading vector samples."""

    @property
    def sample_ids(self) -> tuple[int, ...]:
        """Available sample IDs."""
        ...

    def load_sample(self, sample_id: int) -> VectorSample:
        """Load one vector sample."""
        ...

    def iter_samples(self) -> Iterator[VectorSample]:
        """Iterate all vector samples."""
        ...


class NpyMatrixStream:
    """Matrix stream backed by a single .npy file with mmap."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._array = np.load(path, mmap_mode="r")
        if self._array.ndim == 2:
            self._sample_ids = (0,)
        elif self._array.ndim == 3:
            self._sample_ids = tuple(range(int(self._array.shape[0])))
        else:
            raise ValueError(
                f"Matrix npy file must have shape (n,n) or (N,n,n), got {self._array.shape}"
            )

    @property
    def sample_ids(self) -> tuple[int, ...]:
        return self._sample_ids

    def load_dense_sample(self, sample_id: int) -> DenseMatrixSample:
        if sample_id not in self._sample_ids:
            raise KeyError(f"Unknown matrix sample id {sample_id} for {self._path}")
        if self._array.ndim == 2:
            matrix = np.asarray(self._array, dtype=np.float64)
        else:
            matrix = np.asarray(self._array[sample_id], dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"Matrix sample must be 2D, got shape {matrix.shape}")
        return DenseMatrixSample(sample_id=sample_id, matrix=matrix)

    def load_sparse_sample(self, sample_id: int) -> SparseMatrixSample:
        dense = self.load_dense_sample(sample_id)
        indices, values, size = _dense_to_sparse_components(dense.matrix)
        return SparseMatrixSample(
            sample_id=sample_id,
            indices=indices,
            values=values,
            size=size,
        )

    def iter_dense_samples(self) -> Iterator[DenseMatrixSample]:
        for sample_id in self._sample_ids:
            yield self.load_dense_sample(sample_id)

    def iter_sparse_samples(self) -> Iterator[SparseMatrixSample]:
        for sample_id in self._sample_ids:
            yield self.load_sparse_sample(sample_id)


class TxtMatrixStream:
    """Matrix stream backed by a single .txt file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def sample_ids(self) -> tuple[int, ...]:
        return (0,)

    def load_dense_sample(self, sample_id: int) -> DenseMatrixSample:
        if sample_id != 0:
            raise KeyError(f"Unknown matrix sample id {sample_id} for {self._path}")
        matrix = np.loadtxt(self._path, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(
                f"Matrix txt source must contain a single 2D matrix, got shape {matrix.shape}"
            )
        return DenseMatrixSample(sample_id=0, matrix=matrix)

    def load_sparse_sample(self, sample_id: int) -> SparseMatrixSample:
        dense = self.load_dense_sample(sample_id)
        indices, values, size = _dense_to_sparse_components(dense.matrix)
        return SparseMatrixSample(
            sample_id=sample_id,
            indices=indices,
            values=values,
            size=size,
        )

    def iter_dense_samples(self) -> Iterator[DenseMatrixSample]:
        yield self.load_dense_sample(0)

    def iter_sparse_samples(self) -> Iterator[SparseMatrixSample]:
        yield self.load_sparse_sample(0)


class GlobMatrixStream:
    """Matrix stream backed by glob-matched .txt/.npy files."""

    def __init__(
        self,
        expr: str,
        sample_id_regex: str | None = None,
        enumerate_by: EnumerateBy | None = None,
    ) -> None:
        pattern_path = Path(expr)
        parent = pattern_path.parent
        if not parent.exists():
            raise FileNotFoundError(f"Matrix glob parent directory not found: {parent}")
        paths = sorted(parent.glob(pattern_path.name))
        if not paths:
            raise FileNotFoundError(f"No matrix files match glob: {expr}")
        if enumerate_by is not None:
            mapping: dict[int, Path] = _enumerate_files(paths, enumerate_by)
        else:
            regex = re.compile(sample_id_regex or _DEFAULT_SAMPLE_ID_REGEX)
            mapping = {}
            for path in paths:
                sample_id = _extract_sample_id(path, regex)
                if sample_id in mapping:
                    raise ValueError(
                        f"Duplicate matrix sample id {sample_id} for files {mapping[sample_id]} and {path}"
                    )
                mapping[sample_id] = path
        self._mapping = mapping
        self._sample_ids = tuple(sorted(mapping.keys()))

    @property
    def sample_ids(self) -> tuple[int, ...]:
        return self._sample_ids

    def load_dense_sample(self, sample_id: int) -> DenseMatrixSample:
        path = self._mapping.get(sample_id)
        if path is None:
            raise KeyError(f"Unknown matrix sample id {sample_id}")
        if path.suffix == ".npy":
            matrix = np.load(path, mmap_mode="r")
            if matrix.ndim != 2:
                raise ValueError(
                    f"Glob matrix file {path} must contain exactly one 2D matrix, got shape {matrix.shape}"
                )
            dense = np.asarray(matrix, dtype=np.float64)
        elif path.suffix == ".txt":
            dense = np.loadtxt(path, dtype=np.float64)
            if dense.ndim != 2:
                raise ValueError(f"Glob matrix txt file {path} must be 2D, got {dense.shape}")
        else:
            raise ValueError(f"Unsupported matrix file extension in glob: {path.suffix}")
        return DenseMatrixSample(sample_id=sample_id, matrix=dense)

    def load_sparse_sample(self, sample_id: int) -> SparseMatrixSample:
        dense = self.load_dense_sample(sample_id)
        indices, values, size = _dense_to_sparse_components(dense.matrix)
        return SparseMatrixSample(
            sample_id=sample_id,
            indices=indices,
            values=values,
            size=size,
        )

    def iter_dense_samples(self) -> Iterator[DenseMatrixSample]:
        for sample_id in self._sample_ids:
            yield self.load_dense_sample(sample_id)

    def iter_sparse_samples(self) -> Iterator[SparseMatrixSample]:
        for sample_id in self._sample_ids:
            yield self.load_sparse_sample(sample_id)


class NpyVectorStream:
    """Vector stream backed by a .npy file with mmap."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._array = np.load(path, mmap_mode="r")
        if self._array.ndim == 1:
            self._sample_ids = (0,)
        elif self._array.ndim == 2:
            self._sample_ids = tuple(range(int(self._array.shape[0])))
        else:
            raise ValueError(
                f"Vector npy source must have shape (n,) or (N,n), got {self._array.shape}"
            )

    @property
    def sample_ids(self) -> tuple[int, ...]:
        return self._sample_ids

    def load_sample(self, sample_id: int) -> VectorSample:
        if sample_id not in self._sample_ids:
            raise KeyError(f"Unknown vector sample id {sample_id} for {self._path}")
        if self._array.ndim == 1:
            vector = _normalize_vector(np.asarray(self._array, dtype=np.float64), self._path)
        else:
            vector = _normalize_vector(
                np.asarray(self._array[sample_id], dtype=np.float64), self._path
            )
        return VectorSample(sample_id=sample_id, vector=vector)

    def iter_samples(self) -> Iterator[VectorSample]:
        for sample_id in self._sample_ids:
            yield self.load_sample(sample_id)


class TxtVectorStream:
    """Vector stream backed by one .txt file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def sample_ids(self) -> tuple[int, ...]:
        return (0,)

    def load_sample(self, sample_id: int) -> VectorSample:
        if sample_id != 0:
            raise KeyError(f"Unknown vector sample id {sample_id} for {self._path}")
        vector = _normalize_vector(np.loadtxt(self._path, dtype=np.float64), self._path)
        return VectorSample(sample_id=0, vector=vector)

    def iter_samples(self) -> Iterator[VectorSample]:
        yield self.load_sample(0)


class GlobVectorStream:
    """Vector stream backed by glob-matched .txt/.npy files."""

    def __init__(
        self,
        expr: str,
        sample_id_regex: str | None = None,
        enumerate_by: EnumerateBy | None = None,
    ) -> None:
        pattern_path = Path(expr)
        parent = pattern_path.parent
        if not parent.exists():
            raise FileNotFoundError(f"Vector glob parent directory not found: {parent}")
        paths = sorted(parent.glob(pattern_path.name))
        if not paths:
            raise FileNotFoundError(f"No vector files match glob: {expr}")
        if enumerate_by is not None:
            mapping: dict[int, Path] = _enumerate_files(paths, enumerate_by)
        else:
            regex = re.compile(sample_id_regex or _DEFAULT_SAMPLE_ID_REGEX)
            mapping = {}
            for path in paths:
                sample_id = _extract_sample_id(path, regex)
                if sample_id in mapping:
                    raise ValueError(
                        f"Duplicate vector sample id {sample_id} for files {mapping[sample_id]} and {path}"
                    )
                mapping[sample_id] = path
        self._mapping = mapping
        self._sample_ids = tuple(sorted(mapping.keys()))

    @property
    def sample_ids(self) -> tuple[int, ...]:
        return self._sample_ids

    def load_sample(self, sample_id: int) -> VectorSample:
        path = self._mapping.get(sample_id)
        if path is None:
            raise KeyError(f"Unknown vector sample id {sample_id}")
        if path.suffix == ".npy":
            arr = np.load(path, mmap_mode="r")
        elif path.suffix == ".txt":
            arr = np.loadtxt(path, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported vector file extension in glob: {path.suffix}")
        vector = _normalize_vector(np.asarray(arr, dtype=np.float64), path)
        return VectorSample(sample_id=sample_id, vector=vector)

    def iter_samples(self) -> Iterator[VectorSample]:
        for sample_id in self._sample_ids:
            yield self.load_sample(sample_id)


def open_matrix_stream(
    matrix_path_expr: str,
    sample_id_regex: str | None = None,
    enumerate_by: EnumerateBy | None = None,
) -> MatrixSampleStream:
    """Create a matrix sample stream from path expression."""
    if _is_glob_expression(matrix_path_expr):
        return GlobMatrixStream(
            matrix_path_expr,
            sample_id_regex=sample_id_regex,
            enumerate_by=enumerate_by,
        )
    path = Path(matrix_path_expr)
    if not path.exists():
        raise FileNotFoundError(f"Matrix source not found: {path}")
    if path.suffix == ".npy":
        return NpyMatrixStream(path)
    if path.suffix == ".txt":
        return TxtMatrixStream(path)
    raise ValueError(
        f"Unsupported matrix source '{path}'. Supported: .txt, .npy, or glob patterns."
    )


def open_vector_stream(
    vector_path_expr: str,
    sample_id_regex: str | None = None,
    enumerate_by: EnumerateBy | None = None,
) -> VectorSampleStream:
    """Create a vector sample stream from path expression."""
    if _is_glob_expression(vector_path_expr):
        return GlobVectorStream(
            vector_path_expr,
            sample_id_regex=sample_id_regex,
            enumerate_by=enumerate_by,
        )
    path = Path(vector_path_expr)
    if not path.exists():
        raise FileNotFoundError(f"Vector source not found: {path}")
    if path.suffix == ".npy":
        return NpyVectorStream(path)
    if path.suffix == ".txt":
        return TxtVectorStream(path)
    raise ValueError(
        f"Unsupported vector source '{path}'. Supported: .txt, .npy, or glob patterns."
    )


def bind_sources(
    matrix_ids: tuple[int, ...],
    rhs_ids: tuple[int, ...] | None = None,
    solution_ids: tuple[int, ...] | None = None,
) -> list[SystemBinding]:
    """Bind matrix/rhs/solution ids with single-matrix broadcast semantics.

    Rules:
    - If only one matrix id exists and vectors have many ids, broadcast matrix id.
    - Otherwise bindings are keyed by matrix ids.
    - Provided vector ids must match binding ids (except single-matrix broadcast).
    """
    if not matrix_ids:
        raise ValueError("No matrix samples available to bind.")

    matrix_set = set(matrix_ids)
    rhs_set = set(rhs_ids or ())
    solution_set = set(solution_ids or ())

    if len(matrix_set) == 1:
        matrix_sample_id = next(iter(matrix_set))
        candidate_ids = rhs_set | solution_set
        if not candidate_ids:
            candidate_ids = {matrix_sample_id}
        bound_ids = sorted(candidate_ids)
        return [
            SystemBinding(
                sample_id=sample_id,
                matrix_sample_id=matrix_sample_id,
                rhs_sample_id=sample_id if sample_id in rhs_set else None,
                solution_sample_id=sample_id if sample_id in solution_set else None,
            )
            for sample_id in bound_ids
        ]

    bound_ids = sorted(matrix_set)
    if rhs_set and rhs_set != matrix_set:
        missing = sorted(matrix_set - rhs_set)
        extra = sorted(rhs_set - matrix_set)
        raise ValueError(
            f"RHS IDs must match matrix IDs for multi-matrix sources. Missing={missing}, extra={extra}"
        )
    if solution_set and solution_set != matrix_set:
        missing = sorted(matrix_set - solution_set)
        extra = sorted(solution_set - matrix_set)
        raise ValueError(
            f"Solution IDs must match matrix IDs for multi-matrix sources. Missing={missing}, extra={extra}"
        )
    return [
        SystemBinding(
            sample_id=sample_id,
            matrix_sample_id=sample_id,
            rhs_sample_id=sample_id if sample_id in rhs_set else None,
            solution_sample_id=sample_id if sample_id in solution_set else None,
        )
        for sample_id in bound_ids
    ]


__all__ = [
    "EnumerateBy",
    "DenseMatrixSample",
    "SparseMatrixSample",
    "VectorSample",
    "SystemBinding",
    "MatrixSampleStream",
    "VectorSampleStream",
    "GlobMatrixStream",
    "GlobVectorStream",
    "open_matrix_stream",
    "open_vector_stream",
    "bind_sources",
    "_enumerate_files",
]
