"""Tests for the new experiment-centric configuration loader."""

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest
from dlkit import GeneralSettings

from src.configuration.loader import load_experiments
from src.paths.core import FlowContext


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
        f.write('run = ["exp1", "exp2_fallback"]')

    # Exp1 configs
    with open(project_root / "configs" / "experiments" / "exp1" / "model.toml", "w") as f:
        f.write('[SESSION]\nname = "exp1_model"')
    with open(project_root / "configs" / "experiments" / "exp1" / "data.toml", "w") as f:
        f.write('dataconfig = "configs/datasets/exp1_data.toml"')
    with open(project_root / "configs" / "experiments" / "exp1" / "solver.toml", "w") as f:
        f.write('[general]\nrtol = 1e-5')

    # Fallback experiment (exp2)
    (project_root / "configs" / "experiments" / "exp2_fallback").mkdir()
    with open(project_root / "configs" / "experiments" / "exp2_fallback" / "model.toml", "w") as f:
        f.write('[SESSION]\nname = "exp2_model"')

    # Default configs
    with open(project_root / "configs" / "default" / "data.toml", "w") as f:
        f.write('dataconfig = "configs/datasets/default_data.toml"')
    with open(project_root / "configs" / "default" / "solver.toml", "w") as f:
        f.write('[general]\nrtol = 1e-6')
        
    # Dataset configs
    with open(project_root / "configs" / "datasets" / "exp1_data.toml", "w") as f:
        f.write('[flow]\ndataset="exp1_data_dataset"')
    with open(project_root / "configs" / "datasets" / "default_data.toml", "w") as f:
        f.write('[flow]\ndataset="default_data_dataset"')
        
    return project_root


def test_load_experiments_success(temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that load_experiments correctly loads and resolves a mix of experiments."""
    monkeypatch.chdir(temp_config_structure)
    
    experiments = load_experiments()

    assert len(experiments) == 2
    
    exp1_name, exp1_settings, exp1_context, exp1_model, exp1_data, exp1_solver = experiments[0]
    exp2_name, exp2_settings, exp2_context, exp2_model, exp2_data, exp2_solver = experiments[1]

    # --- Check Experiment 1 (all files present) ---
    assert exp1_name == "exp1"
    assert exp1_settings is not None
    assert isinstance(exp1_context, FlowContext)
    
    # Check paths are correct
    assert exp1_model.name == "model.toml"
    assert "exp1" in str(exp1_model)
    assert exp1_data.name == "exp1_data.toml"
    assert "datasets" in str(exp1_data)
    assert exp1_solver.name == "solver.toml"
    assert "exp1" in str(exp1_solver)

    # Check that the correct solver config was loaded
    assert exp1_settings.EXTRAS.solver_config["general"]["rtol"] == 1e-5

    # --- Check Experiment 2 (fallback) ---
    assert exp2_name == "exp2_fallback"
    
    # Check paths are correct (data and solver should point to default)
    assert exp2_model.name == "model.toml"
    assert "exp2_fallback" in str(exp2_model)
    assert exp2_data.name == "default_data.toml" # Fallback
    assert "datasets" in str(exp2_data)
    assert exp2_solver.name == "solver.toml" # Fallback
    assert "default" in str(exp2_solver)

    # Check that the correct (default) solver config was loaded
    assert exp2_settings.EXTRAS.solver_config["general"]["rtol"] == 1e-6


def test_load_experiments_file_not_found(temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that a FileNotFoundError is raised for a missing config."""
    monkeypatch.chdir(temp_config_structure)
    
    # Create a new experiments.toml for this specific test case
    with open(temp_config_structure / "configs" / "experiments.toml", "w") as f:
        f.write('run = ["exp1", "exp2_fallback", "exp3_missing"]')
    
    (temp_config_structure / "configs" / "experiments" / "exp3_missing").mkdir()

    with pytest.raises(FileNotFoundError, match="model.toml"):
        load_experiments()
