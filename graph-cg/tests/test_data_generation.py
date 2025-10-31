from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data_collection import _append_raw_samples
from src.data_generation import generate_mixture
from src.data_pipeline import RawSamples
from src.normalization import ResidualTraceSamples


def test_generate_mixture_residual_traces() -> None:
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 0.0], dtype=np.float64)

    rhs, solutions, residuals = generate_mixture(
        A=A,
        b=b,
        mix={"cg_residual": 1.0},
        total=2,
        residual_iters=3,
        seed=7,
        shuffle=False,
        normalize="none",
    )

    assert residuals is not None
    assert rhs.shape == (2, 2)
    assert solutions.shape == (2, 2)
    assert residuals.residuals.shape[1] == A.shape[0]
    assert residuals.solutions.shape == residuals.residuals.shape
    assert residuals.sample_indices.max() < rhs.shape[0]
    assert np.all(residuals.iteration_indices >= 0)


def test_append_raw_samples_offsets_residuals(tmp_path: Path) -> None:
    matrix = np.eye(2, dtype=np.float64)

    base = RawSamples(
        matrix=matrix,
        rhs=np.array([[1.0, 0.0]], dtype=np.float64),
        solutions=np.array([[0.5, 0.0]], dtype=np.float64),
        mother_rhs=np.array([1.0, 0.0], dtype=np.float64),
        residual_traces=ResidualTraceSamples(
            residuals=np.array([[0.1, 0.0]], dtype=np.float64),
            solutions=np.array([[0.05, 0.0]], dtype=np.float64),
            sample_indices=np.array([0], dtype=np.int64),
            iteration_indices=np.array([0], dtype=np.int64),
        ),
    )

    addition = RawSamples(
        matrix=matrix,
        rhs=np.array([[0.0, 1.0]], dtype=np.float64),
        solutions=np.array([[0.0, 0.5]], dtype=np.float64),
        mother_rhs=np.array([0.0, 1.0], dtype=np.float64),
        residual_traces=ResidualTraceSamples(
            residuals=np.array([[0.0, 0.2]], dtype=np.float64),
            solutions=np.array([[0.0, 0.1]], dtype=np.float64),
            sample_indices=np.array([0], dtype=np.int64),
            iteration_indices=np.array([0], dtype=np.int64),
        ),
    )

    merged = _append_raw_samples(base, addition)

    assert merged.rhs.shape == (2, 2)
    assert merged.residual_traces is not None
    assert merged.residual_traces.sample_indices.tolist() == [0, 1]
    assert merged.residual_traces.iteration_indices.tolist() == [0, 0]
