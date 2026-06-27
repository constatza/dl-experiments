"""Artifact staging for resolved comparison input systems."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neuralls.shared.types import ComparisonRhsGenerationKind, RhsKind


@dataclass(frozen=True)
class ComparisonInputArtifacts:
    """Platform-owned payload for staging resolved comparison inputs."""

    matrix: np.ndarray
    rhs: np.ndarray
    matrix_dataset_id: str
    matrix_index: int
    rhs_source_type: str
    rhs_dataset_id: str | None = None
    rhs_index: int | None = None
    rhs_kind: RhsKind | None = None
    generator_kind: ComparisonRhsGenerationKind | None = None
    generator_params: dict[str, object] | None = None


def stage_comparison_inputs(root: Path, resolved: ComparisonInputArtifacts) -> Path:
    """Persist the resolved `(A, b)` pair and provenance for MLflow upload."""
    output_dir = root / "comparison_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "matrix.npy", resolved.matrix)
    np.save(output_dir / "rhs.npy", resolved.rhs)
    provenance = {
        "matrix_dataset_id": resolved.matrix_dataset_id,
        "matrix_index": resolved.matrix_index,
        "rhs_source_type": resolved.rhs_source_type,
        "rhs_dataset_id": resolved.rhs_dataset_id,
        "rhs_index": resolved.rhs_index,
        "rhs_kind": str(resolved.rhs_kind) if resolved.rhs_kind is not None else None,
        "generator_kind": (
            str(resolved.generator_kind) if resolved.generator_kind is not None else None
        ),
        "generator_params": resolved.generator_params,
        "matrix_shape": list(resolved.matrix.shape),
        "rhs_shape": list(resolved.rhs.shape),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_dir
