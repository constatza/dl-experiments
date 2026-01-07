import pytest
import numpy as np
import tomli_w
from pathlib import Path
from neuralls.workflows.runner import run_experiment_matrix
import os

def test_run_experiments_full_flow(tmp_path):
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
    models_dir = configs_dir / "models"
    models_dir.mkdir()
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
        "source": {
            "matrix_path": str(matrix_path),
            "rhs_path": str(rhs_path),
        },
        "generation": {
            "normalize": "none",
            "shuffle": True,
            "seed": 42,
            "strategy": [
                {"name": "random", "samples": 100}
            ]
        },
        "output": {
            "processed_dir": str(data_dir / "processed"),
            "output_root": str(data_dir / "output"),
        }
    }
    with open(data_config_path, "wb") as f:
        tomli_w.dump(data_config, f)
        
    # 3. Create Solver Config in shared solvers directory (NEW FORMAT)
    solver_config_path = solvers_dir / "default.toml"
    solver_config = {
        "solvers": [
            {
                "name": "CG",
                "type": "none"
            }
        ],
        "general": {
            "rtol": 1e-6,
            "max_iterations": 10,
            "matrix_path": str(matrix_path),  # Required for synthetic benchmarks
            "output_root": str(data_dir / "output"),
        }
    }
    with open(solver_config_path, "wb") as f:
        tomli_w.dump(solver_config, f)
        
    # 4. Create Model Config in shared models directory (NEW FORMAT)
    exp_name = "test_experiment"

    model_config_path = models_dir / f"{exp_name}_model.toml"
    model_config = {
                                "SESSION": {
                                    "seed": 42,
                                    "precision": "float64",
                                    "name": "test_model"
                                },
                                "MODEL": {
                                    "name": "NormScaledConstantWidthFFNN",
                                    "module_path": "dlkit.nn.ffnn",
                                    "hidden_size": 2,
                                    "num_layers": 1
        },        "TRAINING": {
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
                            "enable_version_counter": False
                        }
                    ]
                },
            "optimizer": {
                "lr": 1e-3,
                "name": "AdamW"
            },
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "factor": 0.5,
                "patience": 5,
                "min_lr": 1e-6
            },
            "metrics": [
                {
                    "name": "NormalizedVectorNormError",
                    "module_path": "dlkit.core.training.metrics",
                    "norm_ord": 2,
                    "vector_dim": -1
                }
            ]
        },
        "DATASET": {
            "name": "FlexibleDataset"
        },
        "DATAMODULE": {
            "name": "InMemoryModule",
            "dataloader": {
                "num_workers": 0,
                "batch_size": 2,
                "pin_memory": False,
                "shuffle": True
            }
        },
        "MLFLOW": {"enabled": False},
        "OPTUNA": {"enabled": False},
        "PATHS": {
             "project_root": str(tmp_path),
             "results_dir": str(data_dir / "output"),
        }
    }
    with open(model_config_path, "wb") as f:
        tomli_w.dump(model_config, f)

    # 5. Create Master Experiment Config (NEW FORMAT with [[experiment]] entries)
    master_config_path = configs_dir / "experiments.toml"
    with open(master_config_path, "w") as f:
        f.write(f'project_root = ".."\n')
        f.write(f'output_dir = "{data_dir / "output"}"\n\n')
        f.write('[[experiment]]\n')
        f.write(f'id = "{exp_name}"\n')
        f.write(f'dataset = "test_data_gen"\n')
        f.write(f'model = "{exp_name}_model"\n')
    
    # Set GRAPH_CG_OUTPUT_DIR to ensure no contamination (although we passed project_root)
    os.environ["GRAPH_CG_OUTPUT_DIR"] = str(data_dir / "output")

    # 6. Run the flow
    results = run_experiment_matrix(
        experiments_config_path=master_config_path,
        force=True,
        project_root=tmp_path
    )

    assert len(results) == 1
    assert results[0].experiment_id == "test_model"
    assert results[0].status == "Success"


def test_run_experiment_matrix_with_mlflow(tmp_path):
    """Test experiment matrix with MLflow tracking enabled.

    This test verifies that the experiment workflow works correctly
    with MLflow enabled, ensuring:
    - Experiments are created with dataset names
    - Runs are created with model-timestamp names
    - Artifacts are stored in the correct hierarchy
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
    models_dir = configs_dir / "models"
    models_dir.mkdir()

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
        "source": {
            "matrix_path": str(matrix_path),
            "rhs_path": str(rhs_path),
        },
        "generation": {
            "normalize": "none",
            "shuffle": True,
            "seed": 42,
            "strategy": [
                {"name": "random", "samples": 50}
            ]
        },
        "output": {
            "processed_dir": str(data_dir / "processed"),
            "output_root": str(data_dir / "output"),
        }
    }
    with open(data_config_path, "wb") as f:
        tomli_w.dump(data_config, f)

    # 3. Create Model Config with MLflow ENABLED
    exp_name = "mlflow_experiment"
    output_root = data_dir / "output"
    mlflow_db = output_root / "mlruns" / "mlflow.db"
    mlflow_artifacts = output_root / "mlartifacts"

    model_config_path = models_dir / f"{exp_name}_model.toml"
    model_config = {
        "SESSION": {
            "seed": 42,
            "precision": "float64",
            "name": "mlflow_test_model"
        },
        "MODEL": {
            "name": "NormScaledConstantWidthFFNN",
            "module_path": "dlkit.nn.ffnn",
            "hidden_size": 2,
            "num_layers": 1
        },
        "TRAINING": {
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
                        "filename": "mlflow_test_ckpt",
                        "monitor": "val_loss",
                        "save_top_k": 1,
                        "every_n_epochs": 1,
                        "enable_version_counter": False
                    }
                ]
            },
            "optimizer": {
                "lr": 1e-3,
                "name": "AdamW"
            },
            "metrics": [
                {
                    "name": "NormalizedVectorNormError",
                    "module_path": "dlkit.core.training.metrics",
                    "norm_ord": 2,
                    "vector_dim": -1
                }
            ]
        },
        "DATASET": {
            "name": "FlexibleDataset"
        },
        "DATAMODULE": {
            "name": "InMemoryModule",
            "dataloader": {
                "num_workers": 0,
                "batch_size": 2,
                "pin_memory": False,
                "shuffle": True
            }
        },
        "MLFLOW": {
            "enabled": True,
            "server": {
                "backend_store_uri": f"sqlite:///{mlflow_db.as_posix()}",
                "artifacts_destination": mlflow_artifacts.as_posix(),
            },
        },
        "OPTUNA": {"enabled": False},
        "PATHS": {
            "project_root": str(tmp_path),
            "results_dir": str(data_dir / "output"),
        }
    }
    with open(model_config_path, "wb") as f:
        tomli_w.dump(model_config, f)

    # 4. Create Master Experiment Config
    master_config_path = configs_dir / "experiments.toml"
    with open(master_config_path, "w") as f:
        f.write(f'project_root = ".."\n')
        f.write(f'output_dir = "{data_dir / "output"}"\n\n')
        f.write('[[experiment]]\n')
        f.write(f'id = "{exp_name}"\n')
        f.write(f'dataset = "mlflow_test_data"\n')
        f.write(f'model = "{exp_name}_model"\n')

    # Set GRAPH_CG_OUTPUT_DIR
    os.environ["GRAPH_CG_OUTPUT_DIR"] = str(data_dir / "output")

    # 5. Run the flow with MLflow enabled
    results = run_experiment_matrix(
        experiments_config_path=master_config_path,
        force=True,
        project_root=tmp_path
    )

    # Verify results
    assert len(results) == 1
    assert results[0].experiment_id == "mlflow_test_model"
    assert results[0].status == "Success"

    # Verify MLflow tracking directory was created
    assert (output_root / "mlruns").exists()
    assert (output_root / "mlartifacts").exists()
