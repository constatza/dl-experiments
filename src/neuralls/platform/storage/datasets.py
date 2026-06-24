"""Compatibility facade for manifest-driven dataset storage helpers."""

from __future__ import annotations

import shutil

import zarr

from neuralls.platform.storage.dataset_readers import (
    DatasetArtifacts,
    DatasetPaths,
    load_dense_training_arrays,
    load_matrix_dense_sample,
    read_training_sample_count,
    resolve_dataset_artifacts,
    resolve_dataset_paths,
)
from neuralls.platform.storage.generation_formats import (
    DenseHdf5Accumulator,
    DenseNpyAccumulator,
    DenseZarrAccumulator,
    GenerationDatasetStorage,
    Hdf5GenerationStorage,
    NpyGenerationStorage,
    ZarrGenerationStorage,
    make_generation_dataset_storage,
)
from neuralls.platform.storage.manifest import (
    DatasetArtifact,
    DatasetManifest,
    DatasetNormalization,
)
from neuralls.platform.storage.manifest_io import load_dataset_manifest, read_dataset_manifest

DenseDatasetWriter = ZarrGenerationStorage

__all__ = [
    "DatasetArtifact",
    "DatasetArtifacts",
    "DatasetManifest",
    "DatasetNormalization",
    "DatasetPaths",
    "DenseDatasetWriter",
    "DenseHdf5Accumulator",
    "DenseNpyAccumulator",
    "DenseZarrAccumulator",
    "GenerationDatasetStorage",
    "Hdf5GenerationStorage",
    "NpyGenerationStorage",
    "ZarrGenerationStorage",
    "load_dataset_manifest",
    "load_dense_training_arrays",
    "load_matrix_dense_sample",
    "make_generation_dataset_storage",
    "read_dataset_manifest",
    "read_training_sample_count",
    "resolve_dataset_artifacts",
    "resolve_dataset_paths",
    "shutil",
    "zarr",
]
