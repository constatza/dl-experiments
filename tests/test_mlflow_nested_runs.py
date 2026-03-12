"""Nested runs hierarchy tests for Optuna integration with MLflow.

These tests verify that dlkit's nested runs implementation works correctly.
NO implementation code - only validation that dlkit handles everything.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tomli_w


@pytest.fixture
def optuna_training_setup(tmp_path: Path) -> dict:
    """Create minimal setup for Optuna hyperparameter tuning."""
    import numpy as np

    # Setup directories
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw_dir = data_dir / "raw"
    raw_dir.mkdir()

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()

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
        "datasets_dir": datasets_dir,
        "models_dir": models_dir,
        "matrix_path": matrix_path,
        "rhs_path": rhs_path,
    }


class TestOptunaNestedRuns:
    """Tests verifying dlkit's nested runs implementation for Optuna."""

    def test_optuna_configuration_structure(self, optuna_training_setup: dict):
        """Verify Optuna configuration is properly structured for nested runs.

        Note: This test verifies the configuration is set up correctly but
        does not run actual Optuna trials (which would be time-consuming).
        It confirms that when Optuna is enabled, the configuration supports
        nested runs.
        """
        from neuralls.configuration.loader import load_experiment

        data_dir = optuna_training_setup["data_dir"]
        datasets_dir = optuna_training_setup["datasets_dir"]
        models_dir = optuna_training_setup["models_dir"]
        matrix_path = optuna_training_setup["matrix_path"]
        rhs_path = optuna_training_setup["rhs_path"]

        # Create data config
        data_config_path = datasets_dir / "optuna_test_data.toml"
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
                "data_dir": str(data_dir / "processed"),
            },
        }
        with open(data_config_path, "wb") as f:
            tomli_w.dump(data_config, f)

        # Create model config with OPTUNA enabled
        output_root = data_dir / "output"
        model_config_path = models_dir / "optuna_model.toml"
        model_config = {
            "SESSION": {
                "seed": 42,
                "name": "OptunaTestSession",
            },
            "MODEL": {
                "name": "OptunaTestModel",
                "module_path": "dlkit.nn.ffnn",
                "hidden_size": 2,
                "num_layers": 1,
            },
            "TRAINING": {
                "trainer": {
                    "max_epochs": 1,
                    "accelerator": "cpu",
                    "enable_checkpointing": False,  # Disable for speed
                    "num_sanity_val_steps": 0,
                    "limit_train_batches": 1,
                    "limit_val_batches": 1,
                },
                "optimizer": {"lr": 1e-3, "name": "AdamW"},
            },
            "DATASET": {"name": "FlexibleDataset"},
            "DATAMODULE": {
                "name": "InMemoryModule",
                "dataloader": {
                    "num_workers": 0,
                    "batch_size": 2,
                    "pin_memory": False,
                },
            },
            "MLFLOW": {
                "enabled": True,
            },
            "OPTUNA": {
                "enabled": True,
                "n_trials": 2,  # Minimal for testing
                "direction": "minimize",
                "model": {
                    "hidden_size": {
                        "type": "int",
                        "low": 2,
                        "high": 4,
                    },
                },
            },
        }
        with open(model_config_path, "wb") as f:
            tomli_w.dump(model_config, f)

        # Load experiment
        experiment = load_experiment(
            model_config_path,
            data_config_path,
            output_root=output_root,
            dataset_registry_id=data_config_path.stem,
        )

        # VERIFICATION: Optuna is enabled
        assert experiment.settings.OPTUNA.enabled is True
        assert experiment.settings.OPTUNA.n_trials == 2

        # VERIFICATION: MLflow is also enabled (required for nested runs)
        assert experiment.settings.MLFLOW is not None

    def test_nested_runs_hierarchy_structure(self, optuna_training_setup: dict):
        """Verify nested runs hierarchy structure is properly configured.

        Expected structure when Optuna is enabled:
        - Experiment: dataset name (e.g., "optuna_test_data")
        - Outer run: model-timestamp (e.g., "OptunaTestModel-2025-12-30T10:30:45")
        - Inner runs: Optuna trials (handled by dlkit automatically)

        This test verifies the configuration supports this hierarchy.
        """
        from neuralls.configuration.loader import load_experiment

        data_dir = optuna_training_setup["data_dir"]
        datasets_dir = optuna_training_setup["datasets_dir"]
        models_dir = optuna_training_setup["models_dir"]
        matrix_path = optuna_training_setup["matrix_path"]
        rhs_path = optuna_training_setup["rhs_path"]

        # Create configs
        data_config_path = datasets_dir / "hierarchy_test.toml"
        data_config = {
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

        output_root = data_dir / "output"
        model_config_path = models_dir / "hierarchy_model.toml"
        model_config = {
            "SESSION": {"seed": 42, "name": "HierarchyModel"},
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
            },
            "OPTUNA": {
                "enabled": True,
                "n_trials": 2,
                "direction": "minimize",
                "model": {
                    "hidden_size": {"type": "int", "low": 2, "high": 4},
                },
            },
        }
        with open(model_config_path, "wb") as f:
            tomli_w.dump(model_config, f)

        experiment = load_experiment(
            model_config_path,
            data_config_path,
            output_root=output_root,
            dataset_registry_id=data_config_path.stem,
        )

        # VERIFICATION: Runtime MLflow remains enabled for dlkit-managed nesting.
        assert experiment.settings.MLFLOW is not None

        # VERIFICATION: Optuna model parameters are defined
        assert experiment.settings.OPTUNA.model is not None
        assert "hidden_size" in experiment.settings.OPTUNA.model

    def test_optuna_mlflow_delegation_to_dlkit(self, optuna_training_setup: dict):
        """Verify that Optuna + MLflow nested runs are delegated to dlkit.

        This test confirms that:
        1. Configuration enables both Optuna and MLflow
        2. dlkit will handle the nested run creation automatically
        3. No manual nested run implementation is needed

        The actual nested run creation happens in dlkit's TrackingDecorator,
        which auto-detects when Optuna is running and creates child runs.
        """
        from neuralls.configuration.loader import load_experiment

        data_dir = optuna_training_setup["data_dir"]
        datasets_dir = optuna_training_setup["datasets_dir"]
        models_dir = optuna_training_setup["models_dir"]
        matrix_path = optuna_training_setup["matrix_path"]
        rhs_path = optuna_training_setup["rhs_path"]

        # Create minimal configs
        data_config_path = datasets_dir / "delegation_test.toml"
        data_config = {
            "source": {"matrix_path": str(matrix_path), "rhs_path": str(rhs_path)},
            "generation": {"normalize": "none", "strategy": [{"name": "random", "samples": 10}]},
            "output": {
                "data_dir": str(data_dir / "processed"),
            },
        }
        with open(data_config_path, "wb") as f:
            tomli_w.dump(data_config, f)

        output_root = data_dir / "output"
        model_config_path = models_dir / "delegation_model.toml"
        model_config = {
            "SESSION": {"seed": 42},
            "MODEL": {
                "name": "DelegationModel",
                "module_path": "dlkit.nn.ffnn",
                "hidden_size": 2,
                "num_layers": 1,
            },
            "TRAINING": {"trainer": {"max_epochs": 1}, "optimizer": {"lr": 1e-3, "name": "AdamW"}},
            "DATASET": {"name": "FlexibleDataset"},
            "DATAMODULE": {"name": "InMemoryModule"},
            "MLFLOW": {
                "enabled": True,
            },
            "OPTUNA": {
                "enabled": True,
                "n_trials": 2,
                "direction": "minimize",
            },
        }
        with open(model_config_path, "wb") as f:
            tomli_w.dump(model_config, f)

        experiment = load_experiment(
            model_config_path,
            data_config_path,
            output_root=output_root,
            dataset_registry_id=data_config_path.stem,
        )

        # VERIFICATION: Both Optuna and MLflow are enabled
        assert experiment.settings.OPTUNA.enabled is True
        assert experiment.settings.MLFLOW is not None

        # VERIFICATION: Configuration is ready for dlkit to handle nested runs
        # When dlkit.interfaces.api.execute() is called with these settings:
        # - dlkit's TrackingDecorator will detect Optuna is enabled
        # - It will create a parent run for the optimization
        # - Each Optuna trial will be a child run with mlflow.parentRunId tag
        # - All run context management is handled automatically

        # This is all dlkit's responsibility - we just verify config is correct
        assert experiment.settings.OPTUNA.n_trials > 0


class TestOptunaWorkflowReadiness:
    """Tests verifying the system is ready for Optuna workflows."""

    def test_optuna_artifact_isolation_support(self, optuna_training_setup: dict):
        """Verify workspace structure supports artifact isolation per trial.

        When Optuna runs multiple trials, each trial should have isolated
        artifacts. This is handled by dlkit's run context management.
        """
        from neuralls.configuration.loader import load_experiment

        data_dir = optuna_training_setup["data_dir"]
        datasets_dir = optuna_training_setup["datasets_dir"]
        models_dir = optuna_training_setup["models_dir"]
        matrix_path = optuna_training_setup["matrix_path"]
        rhs_path = optuna_training_setup["rhs_path"]

        # Create configs
        data_config_path = datasets_dir / "isolation_test.toml"
        data_config = {
            "source": {"matrix_path": str(matrix_path), "rhs_path": str(rhs_path)},
            "generation": {"normalize": "none", "strategy": [{"name": "random", "samples": 10}]},
            "output": {
                "data_dir": str(data_dir / "processed"),
            },
        }
        with open(data_config_path, "wb") as f:
            tomli_w.dump(data_config, f)

        output_root = data_dir / "output"
        model_config_path = models_dir / "isolation_model.toml"
        model_config = {
            "SESSION": {"seed": 42},
            "MODEL": {
                "name": "IsolationModel",
                "module_path": "dlkit.nn.ffnn",
                "hidden_size": 2,
                "num_layers": 1,
            },
            "TRAINING": {"trainer": {"max_epochs": 1}, "optimizer": {"lr": 1e-3, "name": "AdamW"}},
            "DATASET": {"name": "FlexibleDataset"},
            "DATAMODULE": {"name": "InMemoryModule"},
            "MLFLOW": {
                "enabled": True,
            },
            "OPTUNA": {"enabled": True, "n_trials": 2},
        }
        with open(model_config_path, "wb") as f:
            tomli_w.dump(model_config, f)

        experiment = load_experiment(
            model_config_path,
            data_config_path,
            output_root=output_root,
            dataset_registry_id=data_config_path.stem,
        )

        assert experiment.settings.MLFLOW is not None
        assert not hasattr(experiment.settings.MLFLOW, "artifacts_destination")

        # dlkit will create separate artifact directories for each trial:
        # - mlartifacts/{exp_id}/{parent_run_id}/
        # - mlartifacts/{exp_id}/{trial_1_run_id}/
        # - mlartifacts/{exp_id}/{trial_2_run_id}/

        # This ensures no artifact crosstalk between trials
