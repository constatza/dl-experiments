from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.comparison._input_resolution import resolve_comparison_input
from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.payloads import GeneratedDatasetPayload
from neuralls.platform.storage.datasets import DenseDatasetWriter, DenseZarrAccumulator
from neuralls.shared.enum_codecs import encode_row_kind_array
from neuralls.shared.types import ComparisonRhsGenerationKind, RowKind


def _build_safe_dataset(root: Path, *, residual: bool = False) -> Path:
    matrix = np.array([[4.0, -1.0], [-1.0, 3.0]], dtype=np.float64)
    matrix_path = root / "matrix.npy"
    np.save(matrix_path, matrix)
    dataset_dir = root / ("residual_dataset" if residual else "safe_dataset")
    if residual:
        build_dataset(
            matrix_path=str(matrix_path),
            dataset_dir=str(dataset_dir),
            counts={"neutral_ones": 2},
            normalize="none",
            shuffle=False,
            seed=7,
            dataset_format="npy",
        )
    else:
        build_dataset(
            matrix_path=str(matrix_path),
            dataset_dir=str(dataset_dir),
            counts={"neutral_ones": 3},
            normalize="none",
            shuffle=False,
            seed=42,
            dataset_format="npy",
        )
    return dataset_dir


def _build_legacy_dense_dataset(root: Path) -> Path:
    matrix = np.array([[4.0, -1.0], [-1.0, 3.0]], dtype=np.float64)
    dataset_dir = root / "legacy_dense_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    acc = DenseZarrAccumulator(dataset_dir / "matrix.zarr")
    acc.append_dense_matrix(matrix, repeats=1)
    zarr_path = acc.finalize()
    DenseDatasetWriter().write_dataset(
        dataset_dir,
        GeneratedDatasetPayload(
            rhs=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            solutions=np.array([[0.3, 0.1], [0.1, 0.4]], dtype=np.float64),
            matrix_artifact_path=zarr_path,
            matrix_size=matrix.shape,
            normalization_type="none",
            matrix_norm=1.0,
            matrix_norm_type="spectral",
        ),
    )
    return dataset_dir


def test_resolve_comparison_input_selects_safe_dataset_rhs(tmp_path: Path) -> None:
    dataset_dir = _build_safe_dataset(tmp_path)

    resolved = resolve_comparison_input(
        matrix_path=dataset_dir,
        matrix_dataset_id="safe-dataset",
        matrix_index=None,
        rhs_path=dataset_dir,
        rhs_dataset_id="safe-dataset",
        rhs_index=None,
        require_non_residual_rhs=True,
        seed=11,
        rhs_generation_kind=None,
        rhs_generation_params=None,
    )

    assert resolved.rhs_source_type == "dataset"
    assert resolved.rhs_dataset_id == "safe-dataset"
    assert resolved.rhs_index is not None
    assert resolved.rhs_kind is RowKind.STANDARD
    assert resolved.matrix.shape == (2, 2)
    assert resolved.rhs.shape == (2,)


def test_resolve_comparison_input_rejects_unsafe_dataset_rhs(tmp_path: Path) -> None:
    dataset_dir = _build_safe_dataset(tmp_path, residual=True)
    # Overwrite rhs_kind codes with all-RESIDUAL to simulate a dataset with no safe rows.
    # (build_dataset now correctly marks r_0=b as NON_RESIDUAL; this tests the filter itself.)
    np.save(
        dataset_dir / "row_kind.npy",
        encode_row_kind_array([RowKind.CG_INTERNAL, RowKind.CG_INTERNAL]),
    )

    with pytest.raises(ValueError, match="No safe RHS rows"):
        resolve_comparison_input(
            matrix_path=dataset_dir,
            matrix_dataset_id="residual-dataset",
            matrix_index=None,
            rhs_path=dataset_dir,
            rhs_dataset_id="residual-dataset",
            rhs_index=None,
            require_non_residual_rhs=True,
            seed=3,
            rhs_generation_kind=None,
            rhs_generation_params=None,
        )


def test_resolve_comparison_input_generates_gaussian_rhs(tmp_path: Path) -> None:
    dataset_dir = _build_safe_dataset(tmp_path)

    resolved = resolve_comparison_input(
        matrix_path=dataset_dir,
        matrix_dataset_id="safe-dataset",
        matrix_index=0,
        rhs_path=None,
        rhs_dataset_id=None,
        rhs_index=None,
        require_non_residual_rhs=True,
        seed=5,
        rhs_generation_kind=ComparisonRhsGenerationKind.GAUSSIAN,
        rhs_generation_params={"mean": 0.0, "std": 1.0},
    )

    assert resolved.rhs_source_type == "generated"
    assert resolved.generator_kind is ComparisonRhsGenerationKind.GAUSSIAN
    assert resolved.rhs.shape == (2,)
    assert resolved.rhs_dataset_id is None


def test_resolve_comparison_input_generates_sparse_rhs(tmp_path: Path) -> None:
    dataset_dir = _build_safe_dataset(tmp_path)

    resolved = resolve_comparison_input(
        matrix_path=dataset_dir,
        matrix_dataset_id="safe-dataset",
        matrix_index=0,
        rhs_path=None,
        rhs_dataset_id=None,
        rhs_index=None,
        require_non_residual_rhs=True,
        seed=None,
        rhs_generation_kind=ComparisonRhsGenerationKind.SPARSE,
        rhs_generation_params={"indices": [0, 1], "values": [3.0, -2.0]},
    )

    np.testing.assert_allclose(resolved.rhs, np.array([3.0, -2.0], dtype=np.float64))
    assert resolved.generator_kind is ComparisonRhsGenerationKind.SPARSE


def test_resolve_comparison_input_accepts_legacy_dataset_without_rhs_kind(
    tmp_path: Path,
) -> None:
    dataset_dir = _build_legacy_dense_dataset(tmp_path)

    resolved = resolve_comparison_input(
        matrix_path=dataset_dir,
        matrix_dataset_id="legacy-dataset",
        matrix_index=None,
        rhs_path=dataset_dir,
        rhs_dataset_id="legacy-dataset",
        rhs_index=None,
        require_non_residual_rhs=True,
        seed=7,
        rhs_generation_kind=None,
        rhs_generation_params=None,
    )

    assert resolved.rhs_source_type == "dataset"
    assert resolved.rhs_dataset_id == "legacy-dataset"
    assert resolved.rhs_index in (0, 1)
    assert resolved.rhs_kind is None
