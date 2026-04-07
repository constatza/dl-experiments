from __future__ import annotations

from pathlib import Path

import numpy as np

from neuralls.composition.comparison.single_run import compare_preconditioners
from neuralls.platform.config.loaders import load_comparison_config
from neuralls.platform.storage.datasets import save_dataset


def _write_dataset(root: Path, A: np.ndarray, b: np.ndarray) -> None:
    solutions = np.linalg.solve(A, b)
    rhs = b.reshape(1, -1)
    sols = solutions.reshape(1, -1)
    save_dataset(
        dataset_dir=root,
        rhs=rhs,
        solutions=sols,
        matrix=A,
        normalization_type="matrix",
        matrix_norm=float(np.linalg.norm(A, ord=2)),
        matrix_norm_type="spectral",
        scale_metadata={},
    )


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
                f'matrix_path = "{system_path}"',
                f'rhs_path = "{system_path}"',
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
                "[flow]",
                'dataset = "test-dataset"',
                "",
                "[output]",
                f'data_dir = "{data_root}"',
                f'output_root = "{data_root}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_model_config(path: Path, checkpoint_dir: Path) -> None:
    """Write minimal model config for test."""
    path.write_text(
        "\n".join(
            [
                "[SESSION]",
                'name = "test-experiment"',
                "",
                "[MODEL]",
                'name = "NormScaledConstantWidthFFNN"',
                'module_path = "dlkit.nn"',
                "",
                "[TRAINING.trainer]",
                "max_epochs = 1",
                "",
                "[TRAINING.optimizer]",
                "lr = 0.001",
                'name = "AdamW"',
                "",
                "[DATAMODULE.dataloader]",
                "batch_size = 32",
                "",
                "[DATASET]",
                'name = "FlexibleDataset"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_compare_preconditioners_workflow(tmp_path: Path) -> None:
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

    comparison_cfg_model = load_comparison_config(comparison_cfg)
    results = compare_preconditioners(
        general_params=comparison_cfg_model.general,
        preconditioner_configs=comparison_cfg_model.preconditioners,
    )

    # Access typed solver results from ComparisonResult
    comparison_results = results.results
    assert set(comparison_results.keys()) == {"none", "jacobi"}
    for name, info in comparison_results.items():
        assert info.iterations > 0, f"{name} did not run"
