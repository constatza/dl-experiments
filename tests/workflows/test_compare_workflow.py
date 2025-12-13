from __future__ import annotations

from pathlib import Path

import numpy as np

from src.cli.comparison import compare_preconditioners


def _write_normalized_dataset(root: Path, A: np.ndarray, b: np.ndarray) -> None:
    solutions = np.linalg.solve(A, b)
    rhs = b.reshape(1, -1)
    sols = solutions.reshape(1, -1)
    np.savez(root / "normalized.npz", matrix=A, rhs=rhs, solutions=sols)


def _write_solver_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[general]",
                "rtol = 1e-6",
                "atol = 1e-14",
                "max_iterations = 50",
                'stopping_criterion = "residual_norm"',
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
                "[metadata]",
                'project_name = "test-project"',
                'experiment_name = "test-experiment"',
                "",
                "[paths]",
                f'checkpoint = "{checkpoint_dir / "model.pt"}"',
                "",
                "[model]",
                'type = "ffnn"',
                "",
                "[model.architecture]",
                "hidden_sizes = [64, 64]",
                "",
                "[training]",
                "batch_size = 32",
                "learning_rate = 0.001",
                "epochs = 100",
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

    solver_cfg = tmp_path / "solver_config.toml"
    _write_solver_config(solver_cfg)

    # Create minimal data config pointing to tmp_path
    data_cfg = tmp_path / "data_config.toml"
    _write_data_config(data_cfg, tmp_path)

    # Create minimal model config
    model_cfg = tmp_path / "model_config.toml"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    _write_model_config(model_cfg, checkpoint_dir)

    # Path to the npz file containing both matrix and rhs
    normalized_npz = tmp_path / "normalized.npz"

    results = compare_preconditioners(
        config_path=model_cfg,
        data_config_path=data_cfg,
        matrix_path=normalized_npz,
        rhs_path=normalized_npz,
        solver_config_path=solver_cfg,
        save_plots=False,
    )

    comparison_results = results["results"]
    assert set(comparison_results.keys()) == {"none", "jacobi"}
    for name, info in comparison_results.items():
        assert info.iterations > 0, f"{name} did not run"
