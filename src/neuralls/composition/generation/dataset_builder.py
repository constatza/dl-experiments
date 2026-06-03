"""Dataset persistence composition for generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neuralls.domain.generation.data_types import NormalizeType
from neuralls.domain.generation.orchestration import build_dataset_payload
from neuralls.domain.generation.ports import DatasetWriterPort, TracingSolverPort
from neuralls.platform.storage.datasets import SparseDatasetWriter


def build_dataset(
    matrix_path: str,
    dataset_dir: str,
    *,
    counts: dict[str, int] | None = None,
    mix: dict[str, float] | None = None,
    total: int | None = None,
    rhs_path: str | None = None,
    sample_id_regex: str | None = None,
    normalize: NormalizeType = "matrix",
    matrix_norm_type: str = "spectral",
    shuffle: bool = True,
    seed: int = 42,
    strategy_overrides: dict[str, dict[str, Any]] | None = None,
    solver_overrides: dict[str, TracingSolverPort] | None = None,
    writer: DatasetWriterPort | None = None,
) -> str:
    """Build a persisted dataset by composing domain payload generation with storage."""
    payload = build_dataset_payload(
        matrix_path=matrix_path,
        counts=counts,
        mix=mix,
        total=total,
        rhs_path=rhs_path,
        sample_id_regex=sample_id_regex,
        normalize=normalize,
        matrix_norm_type=matrix_norm_type,
        shuffle=shuffle,
        seed=seed,
        strategy_overrides=strategy_overrides,
        solver_overrides=solver_overrides,
    )
    dataset_writer = writer or SparseDatasetWriter()
    dataset_writer.write_dataset(Path(dataset_dir), payload)
    return dataset_dir
