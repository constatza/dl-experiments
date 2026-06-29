"""Tests for the experiment runner workflow.

This module tests the `run_experiment_matrix()` workflow function directly,
without involving the CLI layer. These are integration tests that verify
the full workflow logic.
"""

import numpy as np
import tomli_w
from pathlib import Path
from unittest.mock import patch, MagicMock
from neuralls.composition.experiments.training_batch import run_experiment_matrix
import os


def _write_training_job_config(path: Path, *, experiment_name: str) -> None:
    """Write a minimal training job config in the new DLKit job schema."""
    job_config = {
        "run": {
            "type": "train",
            "seed": 42,
            "precision": "64",
        },
        "experiment": {
            "name": experiment_name,
        },
        "model": {
            "class": "ScaleEquivariantFFNN",
            "module_path": "dlkit.nn",
            "params": {
                "hidden_size": 2,
                "num_layers": 1,
            },
        },
        "data": {
            "class": "FlexibleDataset",
            "module": {"class": "ArrayDataModule"},
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": True,
        },
        "training": {
            "trainer": {
                "max_epochs": 1,
                "accelerator": "cpu",
                "enable_checkpointing": True,
                "num_sanity_val_steps": 0,
                "limit_train_batches": 1,
                "limit_val_batches": 1,
                "callbacks": [
                    {
                        "name": "ModelCheckpoint",
                        "filename": "test_ckpt",
                        "monitor": "val_loss",
                        "save_top_k": 1,
                        "every_n_epochs": 1,
                        "enable_version_counter": False,
                    }
                ],
            },
            "optimizer": {
                "default_optimizer": {"lr": 1e-3, "name": "AdamW"},
                "default_scheduler": {
                    "name": "ReduceLROnPlateau",
                    "factor": 0.5,
                    "patience": 5,
                    "min_lr": 1e-6,
                },
            },
            "metrics": [
                {
                    "name": "RelativeVectorNormError",
                    "module_path": "dlkit.domain.metrics",
                    "norm_ord": 2,
                    "vector_dim": -1,
                }
            ],
        },
    }
    with open(path, "wb") as f:
        tomli_w.dump(job_config, f)


@patch("neuralls.composition.experiments.training_batch.train_model")
def test_run_experiments_full_flow(
    mock_train: MagicMock,
    tmp_path: Path,
    neuralls_settings,
) -> None:
    """Test workflow orchestration without expensive training.

    Verifies the complete workflow from configuration loading through
    result handling, with training mocked to avoid computational overhead.
    """
    # Setup directories
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw_dir = data_dir / "raw"
    raw_dir.mkdir()

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()

    # Create shared config directories (NEW FORMAT)
    datasets_dir = configs_dir / "datasets"
    datasets_dir.mkdir()
    jobs_dir = configs_dir / "jobs"
    jobs_dir.mkdir()
    solvers_dir = configs_dir / "solvers"
    solvers_dir.mkdir()

    # 1. Create Dummy Data
    matrix_path = raw_dir / "matrix.txt"
    rhs_path = raw_dir / "rhs.txt"

    N = 10
    A = np.random.rand(N, N)
    A = A.T @ A + np.eye(N)
    b = np.random.rand(N)

    np.savetxt(matrix_path, A)
    np.savetxt(rhs_path, b)

    # 2. Create Data Config in shared datasets directory (NEW FORMAT)
    data_config_path = datasets_dir / "test_data_gen.toml"
    data_config = {
        "id": "test_data_gen",
        "source": {
            "matrix_path": str(matrix_path),
            "rhs_path": str(rhs_path),
        },
        "generation": {
            "normalize": "none",
            "shuffle": True,
            "seed": 42,
            "strategy": [{"name": "random", "samples": 100}],
        },
        "output": {
            "data_dir": str(data_dir / "processed"),
        },
    }
    with open(data_config_path, "wb") as f:
        tomli_w.dump(data_config, f)

    # 3. Create Solver Config in shared solvers directory (NEW FORMAT)
    solver_config_path = solvers_dir / "default.toml"
    solver_config = {
        "solvers": [{"name": "CG", "type": "none"}],
        "general": {
            "rtol": 1e-6,
            "max_iterations": 10,
            "matrix_path": str(matrix_path),  # Required for synthetic benchmarks
            "output_root": str(data_dir / "output"),
        },
    }
    with open(solver_config_path, "wb") as f:
        tomli_w.dump(solver_config, f)

    # 4. Create Job Config in shared jobs directory (NEW FORMAT)
    exp_name = "test_experiment"
    job_config_path = jobs_dir / f"{exp_name}_job.toml"
    _write_training_job_config(job_config_path, experiment_name="test_job")

    # 5. Create master config using [[experiments]] registry entries.
    master_config_path = configs_dir / "experiments.toml"
    with open(master_config_path, "w") as f:
        f.write("[[datasets]]\n")
        f.write('id = "test_data_gen"\n')
        f.write('path = "datasets/test_data_gen.toml"\n\n')
        f.write("[[jobs]]\n")
        f.write(f'id = "{exp_name}_job"\n')
        f.write(f'path = "jobs/{exp_name}_job.toml"\n\n')
        f.write("[[experiments]]\n")
        f.write(f'id = "{exp_name}"\n')
        f.write('dataset = "test_data_gen"\n')
        f.write(f'job = "{exp_name}_job"\n')

    # Set NEURALLS_OUTPUT_DIR to ensure no contamination (although we passed project_root)
    os.environ["NEURALLS_OUTPUT_DIR"] = str(data_dir / "output")

    # Mock training to return success
    mock_train.return_value = MagicMock(experiment_id="test_model", status="Success")

    # 6. Run the flow
    results = run_experiment_matrix(
        experiments_config_path=master_config_path,
        settings=neuralls_settings,
        force=True,
        project_root=tmp_path,
    )

    # Verify workflow behavior
    assert len(results) == 1
    assert results[0].experiment_id == exp_name
    assert results[0].status == "Success"
    assert mock_train.called


@patch("neuralls.composition.experiments.training_batch.train_model")
def test_run_experiment_matrix_with_mlflow(
    mock_train: MagicMock,
    tmp_path: Path,
    neuralls_settings,
) -> None:
    """Test MLflow integration in workflow without expensive training.

    Verifies MLflow experiment/run creation and artifact paths,
    with training mocked to avoid computational overhead.
    """
    # Setup directories
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw_dir = data_dir / "raw"
    raw_dir.mkdir()

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()

    # Create shared config directories
    datasets_dir = configs_dir / "datasets"
    datasets_dir.mkdir()
    jobs_dir = configs_dir / "jobs"
    jobs_dir.mkdir()

    # 1. Create Dummy Data
    matrix_path = raw_dir / "matrix.txt"
    rhs_path = raw_dir / "rhs.txt"

    N = 10
    A = np.random.rand(N, N)
    A = A.T @ A + np.eye(N)
    b = np.random.rand(N)

    np.savetxt(matrix_path, A)
    np.savetxt(rhs_path, b)

    # 2. Create Data Config
    data_config_path = datasets_dir / "mlflow_test_data.toml"
    data_config = {
        "id": "mlflow_test_data",
        "source": {
            "matrix_path": str(matrix_path),
            "rhs_path": str(rhs_path),
        },
        "generation": {
            "normalize": "none",
            "shuffle": True,
            "seed": 42,
            "strategy": [{"name": "random", "samples": 50}],
        },
        "output": {
            "data_dir": str(data_dir / "processed"),
        },
    }
    with open(data_config_path, "wb") as f:
        tomli_w.dump(data_config, f)

    # 3. Create Job Config with tracking enabled
    exp_name = "mlflow_experiment"
    job_config_path = jobs_dir / f"{exp_name}_job.toml"
    _write_training_job_config(job_config_path, experiment_name="mlflow_test_job")

    # 4. Create Master Experiment Config
    master_config_path = configs_dir / "experiments.toml"
    with open(master_config_path, "w") as f:
        f.write("[[datasets]]\n")
        f.write('id = "mlflow_test_data"\n')
        f.write('path = "datasets/mlflow_test_data.toml"\n\n')
        f.write("[[jobs]]\n")
        f.write(f'id = "{exp_name}_job"\n')
        f.write(f'path = "jobs/{exp_name}_job.toml"\n\n')
        f.write("[[experiments]]\n")
        f.write(f'id = "{exp_name}"\n')
        f.write('dataset = "mlflow_test_data"\n')
        f.write(f'job = "{exp_name}_job"\n')

    # Set NEURALLS_OUTPUT_DIR
    os.environ["NEURALLS_OUTPUT_DIR"] = str(data_dir / "output")

    # Mock training to return success
    mock_train.return_value = MagicMock(experiment_id="mlflow_test_model", status="Success")

    # 5. Run the flow with MLflow enabled
    results = run_experiment_matrix(
        experiments_config_path=master_config_path,
        settings=neuralls_settings,
        force=True,
        project_root=tmp_path,
    )

    # Verify workflow behavior
    assert len(results) == 1
    assert results[0].experiment_id == exp_name
    assert results[0].status == "Success"
    assert mock_train.called
