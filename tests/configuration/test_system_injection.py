"""Tests for experiments-config-driven MLflow injection."""

from __future__ import annotations

from pathlib import Path

import tomli_w

from neuralls.configuration.loader import load_experiment


def test_load_experiment_injects_mlflow_from_experiments_config(tmp_path: Path) -> None:
    """Model configs without [MLFLOW] get runtime-injected topology."""
    model_path = tmp_path / "model.toml"
    data_path = tmp_path / "data.toml"
    experiments_path = tmp_path / "experiments.toml"

    mlflow_db = tmp_path / "mlruns" / "mlflow.db"
    tracking_uri = f"sqlite:///{mlflow_db.as_posix()}"

    model_config = {
        "SESSION": {"name": "SystemInjected"},
        "MODEL": {"name": "TestModel", "module_path": "dlkit.nn.ffnn"},
        "TRAINING": {
            "trainer": {"max_epochs": 1, "accelerator": "cpu"},
        },
        "DATASET": {"name": "FlexibleDataset"},
        "DATAMODULE": {"name": "InMemoryModule"},
    }
    with open(model_path, "wb") as fh:
        tomli_w.dump(model_config, fh)

    data_config = {
        "id": "system-injected-data",
        "source": {},
        "generation": {},
        "output": {"data_dir": str(tmp_path / "processed/system-injected-data")},
    }
    with open(data_path, "wb") as fh:
        tomli_w.dump(data_config, fh)

    experiments_config = {
        "mlflow": {
            "client": {
                "tracking_uri": tracking_uri,
            },
        },
        "names": {
            "training": "neuralls-training",
            "comparison": "neuralls-comparisons",
        },
    }
    with open(experiments_path, "wb") as fh:
        tomli_w.dump(experiments_config, fh)

    experiment = load_experiment(
        model_config_path=model_path,
        data_config_path=data_path,
        experiments_config_path=experiments_path,
    )

    assert experiment.settings.MLFLOW.enabled is True
    assert (
        experiment.settings.MLFLOW.client.tracking_uri.rstrip("/")
        == tracking_uri
    )
    assert experiment.settings.MLFLOW.client.experiment_name == "neuralls-training"
    assert experiment.settings.MLFLOW.client.run_name is None
