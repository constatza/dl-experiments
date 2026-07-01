from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neuralls.platform.reporting.comparison_inputs import (
    ComparisonInputArtifacts,
    stage_comparison_inputs,
)
from neuralls.shared.types import ComparisonRhsGenerationKind, RowKind


def test_stage_comparison_inputs_writes_artifacts(tmp_path: Path) -> None:
    resolved = ComparisonInputArtifacts(
        matrix=np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64),
        rhs=np.array([1.0, 4.0], dtype=np.float64),
        matrix_dataset_id="matrix-ds",
        matrix_index=0,
        rhs_source_type="generated",
        rhs_kind=RowKind.STANDARD,
        generator_kind=ComparisonRhsGenerationKind.GAUSSIAN,
        generator_params={"mean": 0.0, "std": 1.0},
    )

    output_dir = stage_comparison_inputs(tmp_path, resolved)

    assert (output_dir / "matrix.npy").exists()
    assert (output_dir / "rhs.npy").exists()
    assert (output_dir / "provenance.json").exists()
    np.testing.assert_allclose(np.load(output_dir / "matrix.npy"), resolved.matrix)
    np.testing.assert_allclose(np.load(output_dir / "rhs.npy"), resolved.rhs)
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["matrix_dataset_id"] == "matrix-ds"
    assert provenance["generator_kind"] == "gaussian"
