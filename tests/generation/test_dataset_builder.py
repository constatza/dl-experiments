"""Tests for format-overwrite protection in dataset_builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.composition.generation.dataset_builder import _guard_format_conflict
from neuralls.platform.storage.manifest import DatasetArtifact, DatasetNormalization
from neuralls.platform.storage.manifest_io import make_dataset_manifest, save_dataset_manifest


def _artifact(fmt: str) -> DatasetArtifact:
    return DatasetArtifact(path="x", format=fmt, dtype="float64", shape=(1,))


@pytest.fixture
def zarr_manifest_dir(tmp_path: Path) -> Path:
    """Dataset directory pre-populated with a zarr manifest."""
    manifest = make_dataset_manifest(
        matrix=_artifact("zarr"),
        rhs=_artifact("zarr"),
        solutions=_artifact("zarr"),
        normalization=DatasetNormalization(
            type="matrix", matrix_norm=1.0, matrix_norm_type="spectral", scale={}
        ),
    )
    save_dataset_manifest(tmp_path, manifest)
    return tmp_path


class TestGuardFormatConflict:
    def test_no_manifest_allows_any_format(self, tmp_path: Path) -> None:
        _guard_format_conflict(tmp_path, "zarr")
        _guard_format_conflict(tmp_path, "npy")

    def test_same_format_allows_overwrite(self, zarr_manifest_dir: Path) -> None:
        _guard_format_conflict(zarr_manifest_dir, "zarr")

    def test_different_format_raises(self, zarr_manifest_dir: Path) -> None:
        with pytest.raises(ValueError, match="zarr.*npy|npy.*zarr"):
            _guard_format_conflict(zarr_manifest_dir, "npy")
