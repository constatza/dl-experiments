from __future__ import annotations

from pathlib import Path

import numpy as np

from src.workflows.compare import compare_preconditioners
from src.io.comparison import load_solver_config
from src.preconditioner_factory import build_preconditioner_configs


def _write_normalized_dataset(root: Path, A: np.ndarray, b: np.ndarray) -> None:
    solutions = np.linalg.solve(A, b)
    rhs = b.reshape(1, -1)
    sols = solutions.reshape(1, -1)
    np.savez(root / "normalized.npz", matrix=A, rhs=rhs, solutions=sols)


def _write_solver_config(path: Path, system_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[general]",
                "rtol = 1e-6",
                "atol = 1e-14",
                "max_iterations = 50",
                'stopping_criterion = "residual_norm"',
                f'matrix = "{system_path}"',
                f'rhs = "{system_path}"',
                "",
                "[[solvers]]",
                'name = "none"',
                'type = "none"',
                "",
                "[[solvers]]",
                'name = "jacobi"',
                'type = "jacobi"',
                "",
                "[data_generation]",
                'normalize = "matrix"',
                "",
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
                f'processed_dir = "{data_root}"',
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
                'module_path = "dlkit.nn.ffnn"',
                "",
                "[TRAINING.trainer]",
                "max_epochs = 1",
                "",
                "[TRAINING.optimizer]",
                "lr = 0.001",
                "name = \"AdamW\"",
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
    """End-to-end check that compare_preconditioners loads solver config and data."""
    A = np.array([[4.0, -1.0], [-1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 2.0], dtype=np.float64)
    _write_normalized_dataset(tmp_path, A, b)

    normalized_npz = tmp_path / "normalized.npz"

    solver_cfg = tmp_path / "solver_config.toml"
    _write_solver_config(solver_cfg, normalized_npz)

    # Create minimal data config pointing to tmp_path
    data_cfg = tmp_path / "data_config.toml"
    _write_data_config(data_cfg, tmp_path)

    # Create minimal model config
    model_cfg = tmp_path / "model_config.toml"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    _write_model_config(model_cfg, checkpoint_dir)

    general_params, solver_entries = load_solver_config(solver_cfg)
    precond_configs = build_preconditioner_configs(
        [{"name": spec.name, "type": spec.type, **spec.args} for spec in solver_entries]
    )

    results = compare_preconditioners(
        general_params=general_params,
        preconditioner_configs=precond_configs,
        output_root=tmp_path / "output",
        figures_root=tmp_path / "figures",
    )

    comparison_results = results["results"]
    assert set(comparison_results.keys()) == {"none", "jacobi"}
    for name, info in comparison_results.items():
        assert info.iterations > 0, f"{name} did not run"
