"""Typed manifest models and serialization for generated datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from neuralls.shared.constants import DATASET_MANIFEST_FILENAME

_DATASET_SCHEMA = "neuralls.dataset.v2"


@dataclass(frozen=True)
class DatasetArtifact:
    """One persisted dataset artifact declared in the manifest."""

    path: str
    format: str
    dtype: str
    shape: tuple[int, ...]
    index: int | None = None
    n_matrix_samples: int | None = None
    broadcast: bool | None = None


@dataclass(frozen=True)
class DatasetNormalization:
    """Normalization metadata stored in the dataset manifest."""

    type: str
    matrix_norm: float
    matrix_norm_type: str
    scale: dict[str, Any]


@dataclass(frozen=True)
class DatasetManifest:
    """Typed view over the dataset manifest contract."""

    schema: str
    matrix: DatasetArtifact
    rhs: DatasetArtifact
    solutions: DatasetArtifact
    normalization: DatasetNormalization
    params: tuple[DatasetArtifact, ...] = ()


def manifest_path_for(dataset_dir: str | Path) -> Path:
    """Return the canonical manifest path for a dataset directory."""
    return Path(dataset_dir) / DATASET_MANIFEST_FILENAME


def manifest_to_dict(manifest: DatasetManifest) -> dict[str, Any]:
    """Serialize a typed manifest into the on-disk JSON shape."""
    payload = asdict(manifest)
    payload["matrix"]["shape"] = list(manifest.matrix.shape)
    payload["rhs"]["shape"] = list(manifest.rhs.shape)
    payload["solutions"]["shape"] = list(manifest.solutions.shape)
    payload["params"] = [
        {**entry, "shape": list(param.shape)}
        for entry, param in zip(payload["params"], manifest.params, strict=True)
    ] or None
    return payload


def save_dataset_manifest(dataset_dir: str | Path, manifest: DatasetManifest) -> None:
    """Write a typed dataset manifest to disk."""
    path = manifest_path_for(dataset_dir)
    path.write_text(
        json.dumps(manifest_to_dict(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_dataset_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    """Load the raw manifest dictionary and validate the schema marker."""
    path = manifest_path_for(dataset_dir)
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not str(manifest.get("schema", "")).startswith(
        "neuralls.dataset"
    ):
        raise ValueError(f"Invalid dataset manifest schema in {path}")
    return manifest


def read_dataset_manifest(dataset_dir: str | Path) -> DatasetManifest:
    """Load the typed dataset manifest."""
    raw = load_dataset_manifest(dataset_dir)

    def _artifact(payload: dict[str, Any]) -> DatasetArtifact:
        return DatasetArtifact(
            path=str(payload["path"]),
            format=str(payload["format"]),
            dtype=str(payload["dtype"]),
            shape=tuple(int(dim) for dim in payload["shape"]),
            index=int(payload["index"]) if payload.get("index") is not None else None,
            n_matrix_samples=(
                int(payload["n_matrix_samples"])
                if payload.get("n_matrix_samples") is not None
                else None
            ),
            broadcast=bool(payload["broadcast"]) if payload.get("broadcast") is not None else None,
        )

    params_raw = raw.get("params") or []
    normalization_raw = raw["normalization"]
    return DatasetManifest(
        schema=str(raw.get("schema", _DATASET_SCHEMA)),
        matrix=_artifact(raw["matrix"]),
        rhs=_artifact(raw["rhs"]),
        solutions=_artifact(raw["solutions"]),
        normalization=DatasetNormalization(
            type=str(normalization_raw["type"]),
            matrix_norm=float(normalization_raw["matrix_norm"]),
            matrix_norm_type=str(normalization_raw["matrix_norm_type"]),
            scale=dict(normalization_raw.get("scale") or {}),
        ),
        params=tuple(_artifact(payload) for payload in params_raw),
    )


def make_dataset_manifest(
    *,
    matrix: DatasetArtifact,
    rhs: DatasetArtifact,
    solutions: DatasetArtifact,
    normalization: DatasetNormalization,
    params: tuple[DatasetArtifact, ...] = (),
) -> DatasetManifest:
    """Construct a typed dataset manifest with the repo schema marker."""
    return DatasetManifest(
        schema=_DATASET_SCHEMA,
        matrix=matrix,
        rhs=rhs,
        solutions=solutions,
        normalization=normalization,
        params=params,
    )
