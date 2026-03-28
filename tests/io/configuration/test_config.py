"""Tests for the new experiment-centric configuration loader."""

from __future__ import annotations

from pathlib import Path
import importlib.util

import pytest
from neuralls.shared.workspace import ExperimentWorkspace, RunnableExperiment
from neuralls.composition.experiments.assembler import load_experiment
from neuralls.shared.constants import (
    EXP_DATA_CONFIG_NAME,
    EXP_MODEL_CONFIG_NAME,
    EXP_SOLVER_CONFIG_NAME,
)

# Skip all tests if dlkit has circular import issue
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("dlkit") is None, reason="dlkit circular import issue"
)


@pytest.fixture
def temp_config_structure(tmp_path: Path) -> Path:
    """Create a temporary directory structure for config files."""
    project_root = tmp_path
    (project_root / "configs").mkdir()
    (project_root / "configs" / "datasets").mkdir()
    (project_root / "configs" / "experiments").mkdir()
    (project_root / "configs" / "default").mkdir()
    (project_root / "configs" / "experiments" / "exp1").mkdir()

    # Exp1 configs
    with open(project_root / "configs" / "experiments" / "exp1" / EXP_MODEL_CONFIG_NAME, "w") as f:
        f.write(
            '[SESSION]\nname = "exp1_model"\n\n[MODEL]\nname = "TestModel"\nmodule_path = "test.module"'
        )
    with open(project_root / "configs" / "experiments" / "exp1" / EXP_DATA_CONFIG_NAME, "w") as f:
        f.write('dataconfig = "configs/datasets/exp1_data.toml"')
    with open(project_root / "configs" / "experiments" / "exp1" / EXP_SOLVER_CONFIG_NAME, "w") as f:
        f.write("[general]\nrtol = 1e-5")

    # Default configs
    with open(project_root / "configs" / "default" / EXP_DATA_CONFIG_NAME, "w") as f:
        f.write('dataconfig = "configs/datasets/default_data.toml"')
    with open(project_root / "configs" / "default" / EXP_SOLVER_CONFIG_NAME, "w") as f:
        f.write("[general]\nrtol = 1e-6")

    # Dataset configs
    with open(project_root / "configs" / "datasets" / "exp1_data.toml", "w") as f:
        f.write('[flow]\ndataset="exp1_data_dataset"')
    with open(project_root / "configs" / "datasets" / "default_data.toml", "w") as f:
        f.write('[flow]\ndataset="default_data_dataset"')

    return project_root


def test_load_experiment_success(temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that load_experiment correctly loads a single experiment."""
    monkeypatch.chdir(temp_config_structure)

    model_path = temp_config_structure / "configs" / "experiments" / "exp1" / EXP_MODEL_CONFIG_NAME
    data_path = temp_config_structure / "configs" / "datasets" / "exp1_data.toml"

    experiment = load_experiment(
        model_config_path=model_path,
        data_config_path=data_path,
        output_root=temp_config_structure / "output",
        dataset_registry_id=data_path.stem,
    )

    assert isinstance(experiment, RunnableExperiment)
    assert isinstance(experiment.workspace, ExperimentWorkspace)
    assert experiment.spec.experiment_id == "exp1_model"
    assert experiment.spec.model_config_path == model_path
    assert experiment.spec.data_config_path == data_path

    # Check paths injection - default_root_dir is now a Path
    assert (
        Path(experiment.settings.TRAINING.trainer.default_root_dir) == experiment.workspace.root_dir
    )


def test_load_experiment_missing_data_config(
    temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test that missing data config raises TypeError (required parameter)."""
    monkeypatch.chdir(temp_config_structure)
    model_path = temp_config_structure / "configs" / "experiments" / "exp1" / EXP_MODEL_CONFIG_NAME

    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'data_config_path'"
    ):
        load_experiment(model_config_path=model_path)  # type: ignore
