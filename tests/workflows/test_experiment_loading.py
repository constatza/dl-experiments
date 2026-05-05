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

    def test_full_training_pipeline_with_mlflow(
        self,
        training_setup: dict,
        neuralls_settings,
    ) -> None:
        """End-to-end test: config load → training → MLflow logging → artifacts.

        Note: This test verifies the configuration is correct for MLflow but
        does not run actual training (which would be slow). It verifies that:
        - MLflow is enabled at runtime
        - Workspace directories are created
        - Output paths are rooted under the requested output directory
        """
        from neuralls.composition.experiments.assembler import load_experiment

        training_setup["tmp_path"]
        data_dir = training_setup["data_dir"]
        datasets_dir = training_setup["datasets_dir"]
        models_dir = training_setup["models_dir"]
        matrix_path = training_setup["matrix_path"]
        rhs_path = training_setup["rhs_path"]

        # Create data config
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
                "strategy": [{"name": "random", "samples": 20}],
            },
            "output": {
                "data_dir": str(data_dir / "processed"),
            },
        }
        with open(data_config_path, "wb") as f:
            tomli_w.dump(data_config, f)

        # Create model config with MLflow enabled but no infrastructure fields.
        output_root = data_dir / "output"

        model_config_path = models_dir / "mlflow_test_model.toml"
        model_config = {
            "SESSION": {
                "seed": 42,
                "workflow": "train",
                "precision": "float64",
                "name": "MLflowTestModel",
            },
            "MODEL": {
                "name": "NormScaledConstantWidthFFNN",
                "module_path": "dlkit.nn",
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
                "optimizer": {
                    "default_optimizer": {"lr": 1e-3, "name": "AdamW"},
                },
                "metrics": [
                    {
                        "name": "NormalizedVectorNormError",
                        "module_path": "dlkit.domain.metrics",
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
            },
            "OPTUNA": {"enabled": False},
        }
        with open(model_config_path, "wb") as f:
            tomli_w.dump(model_config, f)

        # Load experiment
        experiment = load_experiment(
            model_config_path,
            data_config_path,
            neuralls_settings=neuralls_settings,
            output_root=output_root,
            dataset_registry_id=data_config_path.stem,
        )

        # VERIFICATION: Experiment configuration
        assert experiment.settings.MLFLOW is not None
        # VERIFICATION: Workspace directories created
        assert experiment.workspace.checkpoint_dir.exists()
        assert experiment.workspace.figures_dir.exists()
        assert experiment.workspace.predictions_dir.exists()

        # VERIFICATION: Output paths are resolved from output_root, not model TOML.
        assert experiment.settings.PATHS.output_dir == str(output_root)
        assert not hasattr(experiment.settings.MLFLOW, "tracking_uri")
        assert not hasattr(experiment.settings.MLFLOW, "artifacts_destination")

    def test_mlflow_nested_structure_ready(
        self,
        training_setup: dict,
        neuralls_settings,
    ) -> None:
        """Verify MLflow is configured for nested runs structure.

        Tests that the configuration supports the hierarchical structure:
        - Experiment = dataset name
        - Run = model name (with timestamp)
        - Artifacts at mlartifacts/{exp_id}/{run_id}/
        """
        from neuralls.composition.experiments.assembler import load_experiment

        data_dir = training_setup["data_dir"]
        datasets_dir = training_setup["datasets_dir"]
        models_dir = training_setup["models_dir"]
        matrix_path = training_setup["matrix_path"]
        rhs_path = training_setup["rhs_path"]

        # Create data config
        data_config_path = datasets_dir / "nested_test_data.toml"
        data_config = {
            "id": "nested_test_data",
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
                "data_dir": str(data_dir / "processed"),
            },
        }
        with open(data_config_path, "wb") as f:
            tomli_w.dump(data_config, f)

        # Create model config
        output_root = data_dir / "output"

        model_config_path = models_dir / "nested_test_model.toml"
        model_config = {
            "SESSION": {"seed": 42, "workflow": "train", "name": "NestedTestSession"},
            "MODEL": {
                "name": "TestFFNN",
                "module_path": "dlkit.nn",
                "hidden_size": 2,
                "num_layers": 1,
            },
            "TRAINING": {
                "trainer": {"max_epochs": 1, "accelerator": "cpu"},
                "optimizer": {
                    "default_optimizer": {"lr": 1e-3, "name": "AdamW"},
                },
            },
            "DATASET": {"name": "FlexibleDataset"},
            "DATAMODULE": {"name": "InMemoryModule"},
            "MLFLOW": {"enabled": True},
            "OPTUNA": {"enabled": False},
        }
        with open(model_config_path, "wb") as f:
            tomli_w.dump(model_config, f)

        # Load experiment
        experiment = load_experiment(
            model_config_path,
            data_config_path,
            neuralls_settings=neuralls_settings,
            output_root=output_root,
            dataset_registry_id=data_config_path.stem,
        )

        # Verify hierarchical structure is set up
        dataset_id = "nested_test_data"
        run_id = experiment.workspace.run_id

        # Workspace follows: output_root/dataset_id/run_id/
        expected_workspace_root = output_root / dataset_id / run_id
        assert experiment.workspace.root_dir == expected_workspace_root

        assert experiment.settings.PATHS.output_dir == str(output_root)

    def test_mlflow_configuration_injection(
        self,
        training_setup: dict,
        neuralls_settings,
    ) -> None:
        """Verify experiments topology enables runtime MLflow without infra leakage."""
        from neuralls.composition.experiments.assembler import load_experiment

        tmp_path = training_setup["tmp_path"]
        data_dir = training_setup["data_dir"]
        datasets_dir = training_setup["datasets_dir"]
        models_dir = training_setup["models_dir"]
        matrix_path = training_setup["matrix_path"]
        rhs_path = training_setup["rhs_path"]

        # Create minimal configs
        data_config_path = datasets_dir / "injection_test.toml"
        data_config = {
            "id": "injection_test",
            "source": {"matrix_path": str(matrix_path), "rhs_path": str(rhs_path)},
            "generation": {
                "normalize": "none",
                "strategy": [{"name": "random", "samples": 10}],
            },
            "output": {
                "data_dir": str(data_dir / "processed"),
            },
        }
        with open(data_config_path, "wb") as f:
            tomli_w.dump(data_config, f)

        model_config_path = models_dir / "injection_model.toml"
        model_config = {
            "SESSION": {"seed": 42},
            "MODEL": {
                "name": "TestModel",
                "module_path": "dlkit.nn",
                "hidden_size": 2,
                "num_layers": 1,
            },
            "TRAINING": {"trainer": {"max_epochs": 1, "accelerator": "cpu"}},
            "DATASET": {"name": "FlexibleDataset"},
            "DATAMODULE": {"name": "InMemoryModule"},
            "OPTUNA": {"enabled": False},
        }
        with open(model_config_path, "wb") as f:
            tomli_w.dump(model_config, f)

        experiments_config_path = tmp_path / "experiments.toml"
        custom_output_root = data_dir / "custom_output"
        experiments_config = {
            "raw_dir": str(training_setup["raw_dir"]),
            "processed_dir": str(data_dir / "processed"),
            "output_dir": str(custom_output_root),
            "mlflow": {
                "tracking_uri": f"sqlite:///{(custom_output_root / 'mlruns' / 'mlflow.db').as_posix()}",
            },
            "names": {
                "training": "Train",
                "comparison": "Comparisons",
            },
            "datasets": [{"id": "injection_test", "path": "datasets/injection_test.toml"}],
            "models": [{"id": "injection_model", "path": "models/injection_model.toml"}],
            "experiments": [
                {
                    "id": "ignored",
                    "dataset": "injection_test",
                    "model": "injection_model",
                }
            ],
        }
        with open(experiments_config_path, "wb") as f:
            tomli_w.dump(experiments_config, f)

        experiment = load_experiment(
            model_config_path,
            data_config_path,
            neuralls_settings=neuralls_settings,
            case_config_path=experiments_config_path,
            dataset_registry_id=data_config_path.stem,
        )

        assert experiment.settings.MLFLOW is not None
        assert experiment.settings.PATHS.output_dir == str(custom_output_root)
        assert not hasattr(experiment.settings.MLFLOW, "tracking_uri")
        assert not hasattr(experiment.settings.MLFLOW, "artifacts_destination")
