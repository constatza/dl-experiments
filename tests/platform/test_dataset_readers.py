from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.platform.storage.dataset_readers import resolve_canonical_training_triplet
from neuralls.shared.constants import DATASET_MANIFEST_FILENAME
from neuralls.shared.enum_codecs import encode_row_kind_array
from neuralls.shared.types import RowKind


def _build_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    matrix = np.array([[4.0, -1.0], [-1.0, 3.0]], dtype=np.float64)
    matrix_path = root / "matrix.npy"
    np.save(matrix_path, matrix)
    dataset_dir = root / "dataset"
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


def _drop_manifest_matrix_sample_index(dataset_dir: Path) -> None:
    manifest_path = dataset_dir / DATASET_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["matrix_sample_index"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_resolve_canonical_training_triplet_selects_first_standard_row(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)

    triplet = resolve_canonical_training_triplet(dataset_dir, sample_index=None)

    assert triplet.sample_index == 0
    assert triplet.matrix_index == 0
    assert triplet.row_kind is RowKind.STANDARD
    assert triplet.matrix_binding_enforced is True
    assert triplet.matrix.shape == (2, 2)
    assert triplet.rhs.shape == (2,)
    assert triplet.lhs.shape == (2,)


def test_resolve_canonical_training_triplet_rejects_explicit_residual_row(
    tmp_path: Path,
) -> None:
    dataset_dir = _build_dataset(tmp_path)
    np.save(
        dataset_dir / "row_kind.npy",
        encode_row_kind_array([RowKind.STANDARD, RowKind.CG_INTERNAL, RowKind.STANDARD]),
    )

    with pytest.raises(ValueError, match="CG_INTERNAL"):
        resolve_canonical_training_triplet(dataset_dir, sample_index=1)


def test_resolve_canonical_training_triplet_requires_matrix_binding(
    tmp_path: Path,
) -> None:
    dataset_dir = _build_dataset(tmp_path)
    _drop_manifest_matrix_sample_index(dataset_dir)

    with pytest.raises(ValueError, match="matrix_sample_index"):
        resolve_canonical_training_triplet(dataset_dir, sample_index=None)


def test_resolve_canonical_training_triplet_allows_explicit_legacy_matrix_index(
    tmp_path: Path,
) -> None:
    dataset_dir = _build_dataset(tmp_path)
    _drop_manifest_matrix_sample_index(dataset_dir)

    triplet = resolve_canonical_training_triplet(dataset_dir, sample_index=0, matrix_index=0)

    assert triplet.matrix_index == 0
    assert triplet.matrix_binding_enforced is False
