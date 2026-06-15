"""Dataset persistence composition for generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neuralls.domain.generation.data_types import NormalizeType
from neuralls.domain.generation.orchestration import build_dataset_payload
from neuralls.domain.generation.ports import DatasetWriterPort, ZarrAccumulatorPort
from neuralls.domain.generation.source_streams import EnumerateBy
from neuralls.platform.storage.datasets import (
    DenseDatasetWriter,
    DenseZarrAccumulator,
    resolve_dataset_paths,
)


def build_dataset(
    matrix_path: str,
    dataset_dir: str,
    *,
    counts: dict[str, int] | None = None,
    mix: dict[str, float] | None = None,
    total: int | None = None,
    rhs_path: str | None = None,
    solution_path: str | None = None,
    parameters_paths: tuple[str, ...] = (),
    sample_id_regex: str | None = None,
    enumerate_by: EnumerateBy | None = None,
    replacement: bool = False,
    normalize: NormalizeType = "matrix",
    matrix_norm_type: str = "spectral",
    shuffle: bool = True,
    seed: int = 42,
    strategy_overrides: dict[str, dict[str, Any]] | None = None,
    solver_overrides: dict[str, Any] | None = None,
    writer: DatasetWriterPort | None = None,
    accumulator: ZarrAccumulatorPort | None = None,
) -> str:
    """Build a persisted dataset by composing domain payload generation with storage."""
    dataset_path = Path(dataset_dir)
    paths = resolve_dataset_paths(dataset_path)
    paths.root.mkdir(parents=True, exist_ok=True)
    acc: ZarrAccumulatorPort = accumulator or DenseZarrAccumulator(paths.matrix_zarr_dir)
    payload = build_dataset_payload(
        matrix_path=matrix_path,
        counts=counts,
        mix=mix,
        total=total,
        rhs_path=rhs_path,
        solution_path=solution_path,
        parameters_paths=parameters_paths,
        sample_id_regex=sample_id_regex,
        enumerate_by=enumerate_by,
        replacement=replacement,
        normalize=normalize,
        matrix_norm_type=matrix_norm_type,
        shuffle=shuffle,
        seed=seed,
        strategy_overrides=strategy_overrides,
        solver_overrides=solver_overrides,
        accumulator=acc,
    )
    dataset_writer: DatasetWriterPort = writer or DenseDatasetWriter()
    dataset_writer.write_dataset(dataset_path, payload)
    return dataset_dir
