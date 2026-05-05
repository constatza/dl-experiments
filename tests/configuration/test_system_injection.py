"""Tests for case-config-driven MLflow injection."""

from __future__ import annotations

from pathlib import Path

import tomli_w

from neuralls.composition.experiments.assembler import load_experiment


def test_load_experiment_injects_mlflow_from_case_config(
    tmp_path: Path,
    neuralls_settings,
) -> None:
    """Model configs without [MLFLOW] get runtime-injected topology."""
    model_path = tmp_path / "model.toml"
    data_path = tmp_path / "data.toml"
    experiments_path = tmp_path / "experiments.toml"

    mlflow_db = tmp_path / "mlruns" / "mlflow.db"
    tracking_uri = f"sqlite:///{mlflow_db.as_posix()}"

    model_config = {
        "SESSION": {"name": "SystemInjected"},
        "MODEL": {"name": "TestModel", "module_path": "dlkit.nn"},
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
        "raw_dir": str(neuralls_settings.raw_dir),
        "processed_dir": str(neuralls_settings.processed_dir),
        "output_dir": str(neuralls_settings.output_dir),
        "mlflow": {
            "tracking_uri": tracking_uri,
        },
        "names": {
            "training": "Train",
            "comparison": "Comparisons",
        },
    }
    with open(experiments_path, "wb") as fh:
        tomli_w.dump(experiments_config, fh)

    experiment = load_experiment(
        model_config_path=model_path,
        data_config_path=data_path,
        neuralls_settings=neuralls_settings,
        case_config_path=experiments_path,
        dataset_registry_id=data_path.stem,
    )

    assert experiment.settings.MLFLOW is not None
    assert not hasattr(experiment.settings.MLFLOW, "tracking_uri")
    assert not hasattr(experiment.settings.MLFLOW, "artifacts_destination")
