"""Shared data pipeline primitives for data generation/collection flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .common import ensure_dir
from .normalization import (
    NormalizationResult,
    ResidualTraceSamples,
    apply_normalization,
)


@dataclass
class DataContext:
    """Context describing a data-generation request."""

    matrix_path: Path
    dataset_dir: Path
    normalize: str
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawSamples:
    """Raw (unnormalized) samples prior to normalization."""

    matrix: np.ndarray
    rhs: np.ndarray
    solutions: np.ndarray
    mother_rhs: np.ndarray
    residual_traces: ResidualTraceSamples | None = None


@dataclass
class PipelineResult:
    """Result of running the shared data pipeline."""

    normalized: NormalizationResult
    metadata: dict[str, Any]


class SampleBuilder(Protocol):
    """Protocol for building raw samples from a matrix."""

    def __call__(self, context: DataContext, matrix: np.ndarray) -> RawSamples:
        ...


def load_matrix(matrix_path: Path) -> np.ndarray:
    """Load matrix data from disk as float64 array."""

    return np.loadtxt(matrix_path, dtype=np.float64)


def rhs_from_solutions(matrix: np.ndarray, solutions: np.ndarray) -> np.ndarray:
    """Compute RHS vectors given a matrix and solution samples."""

    if solutions.ndim != 2:
        raise ValueError("Solutions must be a 2D array of shape (num_samples, dimension)")

    if matrix.ndim != 2:
        raise ValueError("Matrix must be 2D")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square for linear system generation")

    if solutions.shape[1] != matrix.shape[1]:
        raise ValueError(
            "Solutions dimension does not match matrix columns: "
            f"{solutions.shape[1]} vs {matrix.shape[1]}"
        )

    return solutions @ matrix.T


def mother_rhs(rhs_samples: np.ndarray) -> np.ndarray:
    """Select the reference RHS sample."""

    if rhs_samples.ndim != 2 or rhs_samples.shape[0] == 0:
        raise ValueError("RHS samples must be a non-empty 2D array")

    return rhs_samples[0].copy()


def normalize_samples(context: DataContext, samples: RawSamples) -> NormalizationResult:
    """Normalize raw samples according to the context."""

    return apply_normalization(
        normalize=context.normalize,
        A_original=samples.matrix,
        R=samples.rhs.copy(),
        X=samples.solutions.copy(),
        dataset_dir=context.dataset_dir,
        residual_traces=samples.residual_traces,
    )


def persist_normalized_samples(
    dataset_dir: Path,
    normalized: NormalizationResult,
    mother_rhs_vector: np.ndarray,
) -> None:
    """Persist normalized samples and metadata-free artifacts to disk."""

    ensure_dir(dataset_dir)

    np.save(
        dataset_dir / "matrix.npy",
        normalized.matrix.astype(np.float64, copy=False),
    )
    np.save(
        dataset_dir / "rhs-samples.npy",
        normalized.rhs_samples.astype(np.float64, copy=False),
    )
    np.save(
        dataset_dir / "sol-samples.npy",
        normalized.sol_samples.astype(np.float64, copy=False),
    )
    np.save(
        dataset_dir / "rhs-mother.npy",
        mother_rhs_vector.astype(np.float64, copy=False),
    )

    if normalized.residual_traces is not None:
        traces = normalized.residual_traces
        np.save(
            dataset_dir / "cg-residuals.npy",
            traces.residuals.astype(np.float64, copy=False),
        )
        np.save(
            dataset_dir / "cg-solutions.npy",
            traces.solutions.astype(np.float64, copy=False),
        )
        np.savez(
            dataset_dir / "cg-trace-meta.npz",
            sample_indices=traces.sample_indices.astype(np.int64, copy=False),
            iteration_indices=traces.iteration_indices.astype(np.int64, copy=False),
        )


def persist_graph_samples(
    dataset_dir: Path,
    normalized: NormalizationResult,
    mother_rhs_vector: np.ndarray,
) -> None:
    """Persist normalized samples for graph neural network models.

    For graph models, we save the same files as standard models but document
    that graph models will use both matrix and RHS during training.

    Args:
        dataset_dir: Directory to save dataset files
        normalized: Normalized samples (matrix, RHS, solutions)
        mother_rhs_vector: Reference RHS vector

    Note:
        Graph models (e.g., GNNs) need both matrix and RHS as inputs:
        - matrix.npy: System matrix A (shared across samples)
        - rhs-samples.npy: RHS vectors b (one per sample)
        - sol-samples.npy: Solution vectors x (one per sample)

        The training data loader will combine these appropriately.
    """
    # For now, use the same persistence as standard models
    # The difference is in how the data loader consumes them during training
    persist_normalized_samples(dataset_dir, normalized, mother_rhs_vector)

    # Future enhancement: Could save additional graph-specific metadata here
    # e.g., sparsity pattern, graph topology, edge features, etc.


__all__ = [
    "DataContext",
    "RawSamples",
    "PipelineResult",
    "SampleBuilder",
    "load_matrix",
    "rhs_from_solutions",
    "mother_rhs",
    "normalize_samples",
    "persist_normalized_samples",
    "persist_graph_samples",
]
