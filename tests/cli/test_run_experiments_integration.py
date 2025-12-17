import pytest
import numpy as np
import tomli_w
from pathlib import Path
from src.workflows.runner import run_experiment_matrix
import os

def test_run_experiments_full_flow(tmp_path):
    # Setup directories
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw_dir = data_dir / "raw"
    raw_dir.mkdir()
    
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    experiments_dir = configs_dir / "experiments"
    experiments_dir.mkdir()
    default_dir = experiments_dir / "default"
    default_dir.mkdir()
    
    # 1. Create Dummy Data
    matrix_path = raw_dir / "matrix.txt"
    rhs_path = raw_dir / "rhs.txt"
    
    N = 10
    A = np.random.rand(N, N)
    A = A.T @ A + np.eye(N)
    b = np.random.rand(N)
    
    np.savetxt(matrix_path, A)
    np.savetxt(rhs_path, b)
    
    # 2. Create Data Config (pointing to dummy data)
    data_config_path = configs_dir / "test_data_gen.toml"
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
        
    # 3. Create Solver Config
    solver_config_path = default_dir / "solver.toml"
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
        }
    }
    with open(solver_config_path, "wb") as f:
        tomli_w.dump(solver_config, f)
        
    # 4. Create Model Config
    exp_name = "test_experiment"
    exp_dir = experiments_dir / exp_name
    exp_dir.mkdir()
    
    model_config_path = exp_dir / "model.toml"
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
        
    # 5. Create Experiment Data Pointer
    exp_data_config_path = exp_dir / "data.toml"
    exp_data_config = {
        "dataconfig": str(data_config_path) # Absolute path
    }
    with open(exp_data_config_path, "wb") as f:
        tomli_w.dump(exp_data_config, f)
        
    # 6. Create Master Experiment Config
    master_config_path = configs_dir / "experiments.toml"
    master_config = {
        "run": [exp_name]
    }
    with open(master_config_path, "wb") as f:
        tomli_w.dump(master_config, f)
    
    # Set GRAPH_CG_OUTPUT_DIR to ensure no contamination (although we passed project_root)
    os.environ["GRAPH_CG_OUTPUT_DIR"] = str(data_dir / "output")

    # 7. Run the flow
    results = run_experiment_matrix(
        experiments_config_path=master_config_path,
        force=True,
        project_root=tmp_path
    )

    assert len(results) == 1
    assert results[0].experiment_id == "test_model"
    assert results[0].status == "Success"
