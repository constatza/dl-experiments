"""Tests for the new experiment-centric configuration loader."""

from __future__ import annotations

from pathlib import Path
import tomllib
import importlib.util

import pytest

# Skip all tests if dlkit has circular import issue
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("dlkit") is None,
    reason="dlkit circular import issue"
)

from dlkit import GeneralSettings

from neuralls.configuration.loader import load_batch
from neuralls.configuration.domain import ExperimentBatch, RunnableExperiment, ExperimentWorkspace
from neuralls.constants import (
    EXP_MODEL_CONFIG_NAME,
    EXP_DATA_CONFIG_NAME,
    EXP_SOLVER_CONFIG_NAME,
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

    # Master config
    with open(project_root / "configs" / "experiments.toml", "w") as f:
        f.write('project_root = ".."\n')
        f.write('defaults_dir = "default"\n')
        f.write('[experiments]\n')
        f.write('exp1 = "experiments/exp1"\n')
        f.write('exp2_fallback = "experiments/exp2_fallback"\n')

    # Exp1 configs
    with open(project_root / "configs" / "experiments" / "exp1" / EXP_MODEL_CONFIG_NAME, "w") as f:
        f.write('[SESSION]\nname = "exp1_model"\n\n[MODEL]\nname = "TestModel"\nmodule_path = "test.module"')
    with open(project_root / "configs" / "experiments" / "exp1" / EXP_DATA_CONFIG_NAME, "w") as f:
        f.write('dataconfig = "configs/datasets/exp1_data.toml"')
    with open(project_root / "configs" / "experiments" / "exp1" / EXP_SOLVER_CONFIG_NAME, "w") as f:
        f.write('[general]\nrtol = 1e-5')

    # Fallback experiment (exp2)
    (project_root / "configs" / "experiments" / "exp2_fallback").mkdir()
    with open(project_root / "configs" / "experiments" / "exp2_fallback" / EXP_MODEL_CONFIG_NAME, "w") as f:
        f.write('[SESSION]\nname = "exp2_model"\n\n[MODEL]\nname = "TestModel"\nmodule_path = "test.module"')

    # Default configs
    with open(project_root / "configs" / "default" / EXP_DATA_CONFIG_NAME, "w") as f:
        f.write('dataconfig = "configs/datasets/default_data.toml"')
    with open(project_root / "configs" / "default" / EXP_SOLVER_CONFIG_NAME, "w") as f:
        f.write('[general]\nrtol = 1e-6')
        
    # Dataset configs
    with open(project_root / "configs" / "datasets" / "exp1_data.toml", "w") as f:
        f.write('[flow]\ndataset="exp1_data_dataset"')
    with open(project_root / "configs" / "datasets" / "default_data.toml", "w") as f:
        f.write('[flow]\ndataset="default_data_dataset"')
        
    return project_root


def test_load_experiments_success(temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that load_batch correctly loads and resolves a mix of experiments."""
    monkeypatch.chdir(temp_config_structure)
    
    batch = load_batch(
        master_config_path=temp_config_structure / "configs" / "experiments.toml",
    )

    assert len(batch.experiments) == 2
    
    exp1 = batch.experiments[0]
    exp2 = batch.experiments[1]

    # --- Check Experiment 1 (all files present) ---
    assert exp1.spec.id == "exp1"
    assert exp1.settings is not None
    assert isinstance(exp1.workspace, ExperimentWorkspace)
    
    # Check paths are correct
    assert exp1.spec.model_config_path.name == EXP_MODEL_CONFIG_NAME
    assert "exp1" in str(exp1.spec.model_config_path)
    assert exp1.spec.data_config_path.name == "exp1_data.toml"
    assert "datasets" in str(exp1.spec.data_config_path)
    assert exp1.spec.solver_config_path.name == EXP_SOLVER_CONFIG_NAME
    assert "exp1" in str(exp1.spec.solver_config_path)

    # --- Check Experiment 2 (fallback) ---
    assert exp2.spec.id == "exp2_fallback"

    # Check paths are correct (data and solver should point to default)
    assert exp2.spec.model_config_path.name == EXP_MODEL_CONFIG_NAME
    assert "exp2_fallback" in str(exp2.spec.model_config_path)
    assert exp2.spec.data_config_path.name == "default_data.toml" # Fallback
    assert "datasets" in str(exp2.spec.data_config_path)
    assert exp2.spec.solver_config_path.name == EXP_SOLVER_CONFIG_NAME # Fallback
    assert "default" in str(exp2.spec.solver_config_path)


def test_load_experiments_file_not_found(temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that a FileNotFoundError is raised for a missing config."""
    monkeypatch.chdir(temp_config_structure)
    
    # Create a new experiments.toml for this specific test case
    with open(temp_config_structure / "configs" / "experiments.toml", "w") as f:
        f.write('project_root = ".."\n')
        f.write('defaults_dir = "default"\n')
        f.write('[experiments]\n')
        f.write('exp1 = "experiments/exp1"\n')
        f.write('exp2_fallback = "experiments/exp2_fallback"\n')
        f.write('exp3_missing = "experiments/exp3_missing"\n')
    
    (temp_config_structure / "configs" / "experiments" / "exp3_missing").mkdir()
    
    with pytest.raises(FileNotFoundError, match=EXP_MODEL_CONFIG_NAME):
        load_batch(
            master_config_path=temp_config_structure / "configs" / "experiments.toml",
        )
