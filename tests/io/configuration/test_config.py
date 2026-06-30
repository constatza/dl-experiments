"""Tests for the new experiment-centric configuration loader."""

from __future__ import annotations

from pathlib import Path
import importlib.util

import pytest
from neuralls.platform.config.models.workspace import ExperimentWorkspace, RunnableExperiment
from neuralls.composition.experiments.assembler import load_experiment

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("dlkit") is None, reason="dlkit circular import issue"
)


@pytest.fixture
def temp_config_structure(tmp_path: Path) -> Path:
    """Create a temporary directory structure for config files."""
    project_root = tmp_path
    (project_root / "configs").mkdir()
    (project_root / "configs" / "datasets").mkdir()
    (project_root / "configs" / "jobs").mkdir()
    (project_root / "configs" / "models").mkdir()
    (project_root / "configs" / "profiles").mkdir()

    with open(project_root / "configs" / "models" / "exp1_model.toml", "w") as f:
        f.write(
            "\n".join(
                [
                    "[model]",
                    'name = "TestModel"',
                    'module_path = "dlkit.nn"',
                    "",
                    "[data]",
                    'name = "FlexibleDataset"',
                    "batch_size = 2",
                    "",
                    "[data.module]",
                    'name = "ArrayDataModule"',
                ]
            )
        )

    with open(project_root / "configs" / "profiles" / "training.toml", "w") as f:
        f.write(
            "\n".join(
                [
                    "[training.trainer]",
                    "max_epochs = 1",
                ]
            )
        )

    with open(project_root / "configs" / "jobs" / "exp1_job.toml", "w") as f:
        f.write(
            "\n".join(
                [
                    "[run]",
                    'type = "train"',
                    "seed = 42",
                    'model = "../models/exp1_model.toml"',
                    'data = "../models/exp1_model.toml"',
                    'training = "../profiles/training.toml"',
                    "",
                    "[experiment]",
                    'name = "exp1_job"',
                ]
            )
        )

    # Dataset configs
    with open(project_root / "configs" / "datasets" / "exp1_data.toml", "w") as f:
        f.write('id="exp1_data_dataset"\n[source]\nmatrix_path="matrix.txt"\n')
    with open(project_root / "configs" / "datasets" / "default_data.toml", "w") as f:
        f.write('id="default_data_dataset"\n[source]\nmatrix_path="matrix.txt"\n')

    return project_root


def test_load_experiment_success(temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that load_experiment correctly loads a single experiment."""
    monkeypatch.chdir(temp_config_structure)

    model_path = temp_config_structure / "configs" / "jobs" / "exp1_job.toml"
    data_path = temp_config_structure / "configs" / "datasets" / "exp1_data.toml"

    experiment = load_experiment(
        job_config_path=model_path,
        data_config_path=data_path,
        output_root=temp_config_structure / "output",
        dataset_registry_id=data_path.stem,
    )

    assert isinstance(experiment, RunnableExperiment)
    assert isinstance(experiment.workspace, ExperimentWorkspace)
    assert experiment.spec.experiment_id == "exp1_job"
    assert experiment.spec.job_config_path == model_path
    assert experiment.spec.data_config_path == data_path

    assert (
        Path(experiment.settings.training.trainer.default_root_dir) == experiment.workspace.root_dir
    )


def test_load_experiment_missing_data_config(
    temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test that missing data config raises TypeError (required parameter)."""
    monkeypatch.chdir(temp_config_structure)
    model_path = temp_config_structure / "configs" / "jobs" / "exp1_job.toml"

    with pytest.raises(ValueError, match="data_config_path is required"):
        load_experiment(job_config_path=model_path)
