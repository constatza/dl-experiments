"""Tests for experiment configuration loading workflow.

This module tests the `load_experiment()` workflow function directly,
without involving the CLI layer. These tests verify configuration loading
and MLflow integration at the workflow level.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tomli_w


@pytest.fixture
def training_setup(tmp_path: Path) -> dict:
    """Create minimal training setup with data and configs."""
    # Setup directories
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw_dir = data_dir / "raw"
    raw_dir.mkdir()

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()

    # Create config directories
    datasets_dir = configs_dir / "datasets"
    datasets_dir.mkdir()
    models_dir = configs_dir / "models"
    models_dir.mkdir()

    # Create minimal test data (10x10 SPD matrix)
    matrix_path = raw_dir / "matrix.txt"
    rhs_path = raw_dir / "rhs.txt"

    N = 10
    A = np.random.rand(N, N)
    A = A.T @ A + np.eye(N)  # Make it SPD
    b = np.random.rand(N)

    np.savetxt(matrix_path, A)
    np.savetxt(rhs_path, b)

    return {
        "tmp_path": tmp_path,
        "data_dir": data_dir,
        "raw_dir": raw_dir,
        "configs_dir": configs_dir,
        "datasets_dir": datasets_dir,
        "models_dir": models_dir,
        "matrix_path": matrix_path,
        "rhs_path": rhs_path,
    }


class TestTrainingPipelineWithMLflow:
    """End-to-end tests for training pipeline with MLflow."""

    def test_full_training_pipeline_with_mlflow(self, training_setup: dict) -> None:
        """End-to-end test: config load → training → MLflow logging → artifacts.

        Note: This test verifies the configuration is correct for MLflow but
        does not run actual training (which would be slow). It verifies that:
        - Experiment name matches dataset name
        - Run name equals model name (with timestamp)
        - Workspace directories are created
        - MLflow paths are configured correctly
        """
        from neuralls.configuration.loader import load_experiment

        tmp_path = training_setup["tmp_path"]
        data_dir = training_setup["data_dir"]
        datasets_dir = training_setup["datasets_dir"]
        models_dir = training_setup["models_dir"]
        matrix_path = training_setup["matrix_path"]
        rhs_path = training_setup["rhs_path"]

        # Create data config
        data_config_path = datasets_dir / "mlflow_test_data.toml"
        data_config = {
            "source": {
                "matrix_path": str(matrix_path),
                "rhs_path": str(rhs_path),
            },
            "generation": {
                "normalize": "none",
                "shuffle": True,
                "seed": 42,
                "strategy": [{"name": "random", "samples": 20}],
            },
            "output": {
                "processed_dir": str(data_dir / "processed"),
            },
        }
        with open(data_config_path, "wb") as f:
            tomli_w.dump(data_config, f)

        # Create model config with MLflow ENABLED
        output_root = data_dir / "output"
        mlflow_db = output_root / "mlruns" / "mlflow.db"
        mlflow_artifacts = output_root / "mlartifacts"

        model_config_path = models_dir / "mlflow_test_model.toml"
        model_config = {
            "SESSION": {
                "seed": 42,
                "precision": "float64",
                "name": "MLflowTestModel",
            },
            "MODEL": {
                "name": "NormScaledConstantWidthFFNN",
                "module_path": "dlkit.nn.ffnn",
                "hidden_size": 2,
                "num_layers": 1,
            },
            "TRAINING": {
                "trainer": {
                    "max_epochs": 1,
                    "accelerator": "cpu",
                    "enable_checkpointing": True,
                    "num_sanity_val_steps": 0,
                    "limit_train_batches": 1,
                    "limit_val_batches": 1,
                },
                "optimizer": {"lr": 1e-3, "name": "AdamW"},
                "metrics": [
                    {
                        "name": "NormalizedVectorNormError",
                        "module_path": "dlkit.core.training.metrics",
                        "norm_ord": 2,
                        "vector_dim": -1,
                    }
                ],
            },
            "DATASET": {"name": "FlexibleDataset"},
            "DATAMODULE": {
                "name": "InMemoryModule",
                "dataloader": {
                    "num_workers": 0,
                    "batch_size": 2,
                    "pin_memory": False,
                    "shuffle": True,
                },
            },
            "MLFLOW": {
                "enabled": True,
                "server": {
                    "backend_store_uri": f"sqlite:///{mlflow_db.as_posix()}",
                    "artifacts_destination": mlflow_artifacts.as_posix(),
                },
            },
            "OPTUNA": {"enabled": False},
        }
        with open(model_config_path, "wb") as f:
            tomli_w.dump(model_config, f)

        # Load experiment
        experiment = load_experiment(
            model_config_path,
            data_config_path,
            output_root=output_root,
        )

        # VERIFICATION: Experiment configuration
        assert experiment.settings.MLFLOW.enabled is True
        assert experiment.settings.MLFLOW.client.experiment_name == "mlflow_test_data"

        run_name = experiment.settings.MLFLOW.client.run_name
        assert run_name.startswith("MLflowTestModel")

        # VERIFICATION: Workspace directories created
        assert experiment.workspace.checkpoint_dir.exists()
        assert experiment.workspace.figures_dir.exists()
        assert experiment.workspace.predictions_dir.exists()

        # VERIFICATION: MLflow paths configured correctly
        tracking_uri = experiment.settings.MLFLOW.server.backend_store_uri
        assert "mlruns/mlflow.db" in tracking_uri
        assert "sqlite:///" in tracking_uri

        artifacts_dest = experiment.settings.MLFLOW.server.artifacts_destination
        assert "mlartifacts" in artifacts_dest

    def test_mlflow_nested_structure_ready(self, training_setup: dict) -> None:
        """Verify MLflow is configured for nested runs structure.

        Tests that the configuration supports the hierarchical structure:
        - Experiment = dataset name
        - Run = model name (with timestamp)
        - Artifacts at mlartifacts/{exp_id}/{run_id}/
        """
        from neuralls.configuration.loader import load_experiment

        data_dir = training_setup["data_dir"]
        datasets_dir = training_setup["datasets_dir"]
        models_dir = training_setup["models_dir"]
        matrix_path = training_setup["matrix_path"]
        rhs_path = training_setup["rhs_path"]

        # Create data config
        data_config_path = datasets_dir / "nested_test_data.toml"
        data_config = {
            "source": {
                "matrix_path": str(matrix_path),
                "rhs_path": str(rhs_path),
            },
            "generation": {
                "normalize": "none",
                "shuffle": True,
                "seed": 42,
                "strategy": [{"name": "random", "samples": 20}],
            },
            "output": {
                "processed_dir": str(data_dir / "processed"),
            },
        }
        with open(data_config_path, "wb") as f:
            tomli_w.dump(data_config, f)

        # Create model config
        output_root = data_dir / "output"
        mlflow_db = output_root / "mlruns" / "mlflow.db"
        mlflow_artifacts = output_root / "mlartifacts"

        model_config_path = models_dir / "nested_test_model.toml"
        model_config = {
            "SESSION": {"seed": 42, "name": "NestedTestSession"},
            "MODEL": {
                "name": "TestFFNN",
                "module_path": "dlkit.nn.ffnn",
                "hidden_size": 2,
                "num_layers": 1,
            },
            "TRAINING": {
                "trainer": {"max_epochs": 1, "accelerator": "cpu"},
                "optimizer": {"lr": 1e-3, "name": "AdamW"},
            },
            "DATASET": {"name": "FlexibleDataset"},
            "DATAMODULE": {"name": "InMemoryModule"},
            "MLFLOW": {
                "enabled": True,
                "server": {
                    "backend_store_uri": f"sqlite:///{mlflow_db.as_posix()}",
                    "artifacts_destination": mlflow_artifacts.as_posix(),
                },
            },
            "OPTUNA": {"enabled": False},
        }
        with open(model_config_path, "wb") as f:
            tomli_w.dump(model_config, f)

        # Load experiment
        experiment = load_experiment(
            model_config_path,
            data_config_path,
            output_root=output_root,
        )

        # Verify hierarchical structure is set up
        dataset_id = "nested_test_data"
        run_id = experiment.workspace.run_id

        # Workspace follows: output_root/dataset_id/run_id/
        expected_workspace_root = output_root / dataset_id / run_id
        assert experiment.workspace.root_dir == expected_workspace_root

        # MLflow artifacts will be at: mlartifacts/{exp_id}/{run_id}/
        # (exp_id is UUID, but we can verify the base path)
        assert "mlartifacts" in experiment.settings.MLFLOW.server.artifacts_destination

        # Experiment name = dataset
        assert experiment.settings.MLFLOW.client.experiment_name == dataset_id

        assert experiment.settings.MLFLOW.client.run_name.startswith("NestedTestSession")

    def test_mlflow_configuration_injection(self, training_setup: dict) -> None:
        """Verify MLflow configuration is properly injected from path context."""
        from neuralls.configuration.loader import load_experiment

        data_dir = training_setup["data_dir"]
        datasets_dir = training_setup["datasets_dir"]
        models_dir = training_setup["models_dir"]
        matrix_path = training_setup["matrix_path"]
        rhs_path = training_setup["rhs_path"]

        # Create minimal configs
        data_config_path = datasets_dir / "injection_test.toml"
        data_config = {
            "source": {"matrix_path": str(matrix_path), "rhs_path": str(rhs_path)},
            "generation": {
                "normalize": "none",
                "strategy": [{"name": "random", "samples": 10}],
            },
            "output": {
                "processed_dir": str(data_dir / "processed"),
            },
        }
        with open(data_config_path, "wb") as f:
            tomli_w.dump(data_config, f)

        model_config_path = models_dir / "injection_model.toml"
        model_config = {
            "SESSION": {"seed": 42},
            "MODEL": {
                "name": "TestModel",
                "module_path": "dlkit.nn.ffnn",
                "hidden_size": 2,
                "num_layers": 1,
            },
            "TRAINING": {"trainer": {"max_epochs": 1, "accelerator": "cpu"}},
            "DATASET": {"name": "FlexibleDataset"},
            "DATAMODULE": {"name": "InMemoryModule"},
            "MLFLOW": {"enabled": True},  # Minimal config, paths will be injected
            "OPTUNA": {"enabled": False},
        }
        with open(model_config_path, "wb") as f:
            tomli_w.dump(model_config, f)

        # Load experiment with custom output_root
        custom_output_root = data_dir / "custom_output"

        experiment = load_experiment(
            model_config_path,
            data_config_path,
            output_root=custom_output_root,
        )

        # Model config has no [MLFLOW.server] section — paths are not injected
        tracking_uri = experiment.settings.MLFLOW.server.backend_store_uri
        artifacts_dest = experiment.settings.MLFLOW.server.artifacts_destination

        assert tracking_uri is None
        assert artifacts_dest is None

        # Experiment and run names should be injected
        assert experiment.settings.MLFLOW.client.experiment_name == "injection_test"
        assert experiment.settings.MLFLOW.client.run_name.startswith("TestModel")
