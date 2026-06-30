from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.comparison.single_run import (
    _resolve_comparison_paths,
    compare_preconditioners,
)
from neuralls.platform.config.loaders import load_comparison_config
from neuralls.platform.storage.datasets import DenseDatasetWriter, DenseZarrAccumulator
from neuralls.domain.generation.payloads import GeneratedDatasetPayload


def _write_dataset(root: Path, A: np.ndarray, b: np.ndarray) -> None:
    solutions = np.linalg.solve(A, b)
    rhs = b.reshape(1, -1)
    sols = solutions.reshape(1, -1)
    acc = DenseZarrAccumulator(root / "matrix.zarr")
    acc.append_dense_matrix(A, repeats=1)
    zarr_path = acc.finalize()
    payload = GeneratedDatasetPayload(
        rhs=rhs,
        solutions=sols,
        matrix_artifact_path=zarr_path,
        matrix_size=(int(A.shape[0]), int(A.shape[1])),
        normalization_type="matrix",
        matrix_norm=float(np.linalg.norm(A, ord=2)),
        matrix_norm_type="spectral",
    )
    DenseDatasetWriter().write_dataset(root, payload)


def _write_comparison_config(path: Path, system_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[general]",
                "",
                "[general.params]",
                "rtol = 1e-6",
                "atol = 1e-14",
                "max_iterations = 50",
                'stopping_criterion = "residual_norm"',
                "",
                "[general.data]",
                f'matrix_path = "{system_path.as_posix()}"',
                f'rhs_path = "{system_path.as_posix()}"',
                'normalize_system = "matrix"',
                "",
                "[[preconditioners]]",
                'name = "none"',
                'type = "identity"',
                "",
                "[[preconditioners]]",
                'name = "jacobi"',
                'type = "jacobi"',
            ]
        ),
        encoding="utf-8",
    )


def _write_data_config(path: Path, data_root: Path) -> None:
    """Write minimal data config for test."""
    path.write_text(
        "\n".join(
            [
                'id = "test-dataset"',
                "",
                "[output]",
                f'data_dir = "{data_root.as_posix()}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_model_config(path: Path, checkpoint_dir: Path) -> None:
    """Write minimal model config for test."""
    profile_path = path.with_name(f"{path.stem}-profile.toml")
    profile_path.write_text(
        "\n".join(
            [
                "[model]",
                'name = "ScaleEquivariantFFNN"',
                'module_path = "dlkit.nn"',
                "",
                "[data]",
                'name = "FlexibleDataset"',
                "",
                "[data.module]",
                'name = "ArrayDataModule"',
            ]
        ),
        encoding="utf-8",
    )
    path.write_text(
        "\n".join(
            [
                "[run]",
                'type = "train"',
                "seed = 42",
                f'model = "{profile_path.name}"',
                f'data = "{profile_path.name}"',
                "",
                "[experiment]",
                'name = "test-experiment"',
                "",
                "[training.trainer]",
                "max_epochs = 1",
                "",
                "[training.optimizer.default_optimizer]",
                'name = "AdamW"',
                "lr = 0.001",
            ]
        ),
        encoding="utf-8",
    )


def test_compare_preconditioners_workflow(tmp_path: Path, neuralls_settings) -> None:
    """End-to-end check that compare_preconditioners loads config and data."""
    A = np.array([[4.0, -1.0], [-1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 2.0], dtype=np.float64)
    _write_dataset(tmp_path, A, b)

    comparison_cfg = tmp_path / "comparison_config.toml"
    _write_comparison_config(comparison_cfg, tmp_path)

    # Create minimal data config pointing to tmp_path
    data_cfg = tmp_path / "data_config.toml"
    _write_data_config(data_cfg, tmp_path)

    # Create minimal model config
    model_cfg = tmp_path / "model_config.toml"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    _write_model_config(model_cfg, checkpoint_dir)

    comparison_cfg_model = load_comparison_config(comparison_cfg, neuralls_settings)
    results = compare_preconditioners(
        general_params=comparison_cfg_model.general,
        preconditioner_configs=comparison_cfg_model.preconditioners,
    )

    # Access typed solver results from ComparisonResult
    comparison_results = results.results
    assert set(comparison_results.keys()) == {"none", "jacobi"}
    for name, info in comparison_results.items():
        assert info.iterations > 0, f"{name} did not run"


def test_resolve_comparison_paths_expands_tilde_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    neuralls_settings,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "windows-home"))

    comparison_cfg = tmp_path / "comparison_config.toml"
    _write_comparison_config(comparison_cfg, tmp_path)
    comparison_cfg_model = load_comparison_config(comparison_cfg, neuralls_settings)

    paths = _resolve_comparison_paths(
        general_params=comparison_cfg_model.general,
        output_root=Path(r"~\comparison-output"),
        figures_root=None,
    )

    assert paths.output == (home / "comparison-output").resolve()
    assert paths.figures == (home / "comparison-output" / "figures").resolve()
