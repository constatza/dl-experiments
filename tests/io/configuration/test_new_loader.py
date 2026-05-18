"""Tests for the case-centric configuration loader."""

from __future__ import annotations

from pathlib import Path
import importlib.util

import pytest
from neuralls.platform.config.models.workspace import ExperimentWorkspace
from neuralls.composition.experiments.assembler import load_batch

# Skip all tests if dlkit has circular import issue
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("dlkit") is None, reason="dlkit circular import issue"
)


@pytest.fixture
def temp_config_structure(tmp_path: Path) -> Path:
    """Create a temporary directory structure for NEW config format."""
    project_root = tmp_path
    (project_root / "configs").mkdir()
    (project_root / "configs" / "datasets").mkdir()
    (project_root / "configs" / "models").mkdir()
    matrix_path = project_root / "data" / "matrix.txt"
    matrix_path.parent.mkdir()
    matrix_path.write_text("1.0\n")
    matrix2_path = project_root / "data" / "matrix2.txt"
    matrix2_path.write_text("2.0\n")

    # Case config uses [[experiments]] registry entries.
    with open(project_root / "configs" / "experiments.toml", "w") as f:
        f.write("[[datasets]]\n")
        f.write('id = "exp1_data"\n')
        f.write('path = "datasets/exp1_data.toml"\n\n')
        f.write("[[datasets]]\n")
        f.write('id = "exp2_data"\n')
        f.write('path = "datasets/exp2_data.toml"\n\n')
        f.write("[[models]]\n")
        f.write('id = "exp1_model"\n')
        f.write('path = "models/exp1_model.toml"\n\n')
        f.write("[[models]]\n")
        f.write('id = "exp2_model"\n')
        f.write('path = "models/exp2_model.toml"\n\n')
        f.write("# Experiment 1: Full config with explicit checkpoint\n")
        f.write("[[experiments]]\n")
        f.write('id = "exp1"\n')
        f.write('dataset = "exp1_data"\n')
        f.write('model = "exp1_model"\n')
        f.write(
            f'checkpoint_path = "{(project_root / "checkpoints" / "exp1.ckpt").as_posix()}"\n\n'
        )
        f.write("# Experiment 2: Config without checkpoint (will warn)\n")
        f.write("[[experiments]]\n")
        f.write('id = "exp2"\n')
        f.write('dataset = "exp2_data"\n')
        f.write('model = "exp2_model"\n')

    # Dataset configs
    with open(project_root / "configs" / "datasets" / "exp1_data.toml", "w") as f:
        f.write('id = "exp1_data"\n\n')
        f.write("[source]\n")
        f.write(f'matrix_path = "{matrix_path.as_posix()}"\n\n')
        f.write("[generation]\n")
        f.write('normalize = "matrix"\n')
        f.write("shuffle = false\n")
        f.write("seed = 42\n\n")
        f.write("[output]\n")
        f.write(f'data_dir = "{(project_root / "data" / "processed").as_posix()}"\n')

    with open(project_root / "configs" / "datasets" / "exp2_data.toml", "w") as f:
        f.write('id = "exp2_data"\n\n')
        f.write("[source]\n")
        f.write(f'matrix_path = "{matrix2_path.as_posix()}"\n\n')
        f.write("[generation]\n")
        f.write('normalize = "matrix"\n')
        f.write("shuffle = false\n")
        f.write("seed = 42\n\n")
        f.write("[output]\n")
        f.write(f'data_dir = "{(project_root / "data" / "processed").as_posix()}"\n')

    # Model configs
    with open(project_root / "configs" / "models" / "exp1_model.toml", "w") as f:
        f.write("[SESSION]\n")
        f.write('name = "exp1_model"\n\n')
        f.write("[MODEL]\n")
        f.write('name = "TestModel"\n')
        f.write('module_path = "dlkit.nn"\n\n')
        f.write("[TRAINING]\n")
        f.write("[TRAINING.trainer]\n")
        f.write("max_epochs = 1\n")

    with open(project_root / "configs" / "models" / "exp2_model.toml", "w") as f:
        f.write("[SESSION]\n")
        f.write('name = "exp2_model"\n\n')
        f.write("[MODEL]\n")
        f.write('name = "TestModel2"\n')
        f.write('module_path = "dlkit.domain.nn.graph"\n\n')
        f.write("[TRAINING]\n")
        f.write("[TRAINING.trainer]\n")
        f.write("max_epochs = 1\n")

    return project_root


def test_load_experiments_success(temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that load_batch correctly loads experiments with NEW format."""
    monkeypatch.chdir(temp_config_structure)

    batch = load_batch(
        case_config_path=temp_config_structure / "configs" / "experiments.toml",
    )

    assert len(batch.experiments) == 2

    exp1 = batch.experiments[0]
    exp2 = batch.experiments[1]

    # --- Check Experiment 1 (with checkpoint) ---
    # spec.experiment_id preserves the master registry experiment id.
    assert exp1.spec.experiment_id == "exp1"
    assert exp1.settings is not None
    assert isinstance(exp1.workspace, ExperimentWorkspace)

    # Check paths resolve to shared directories
    assert exp1.spec.model_config_path.name == "exp1_model.toml"
    assert "models" in str(exp1.spec.model_config_path)
    assert exp1.spec.data_config_path.name == "exp1_data.toml"
    assert "datasets" in str(exp1.spec.data_config_path)

    # Check checkpoint path
    assert exp1.spec.checkpoint_path is not None
    assert "exp1.ckpt" in str(exp1.spec.checkpoint_path)

    # --- Check Experiment 2 (without checkpoint) ---
    assert exp2.spec.experiment_id == "exp2"

    # Check paths resolve to shared directories
    assert exp2.spec.model_config_path.name == "exp2_model.toml"
    assert "models" in str(exp2.spec.model_config_path)
    assert exp2.spec.data_config_path.name == "exp2_data.toml"
    assert "datasets" in str(exp2.spec.data_config_path)

    # No checkpoint for exp2
    assert exp2.spec.checkpoint_path is None


def test_load_experiments_missing_registry_id(
    temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch
):
    """Missing registry ids fail before any guessed-path lookup."""
    monkeypatch.chdir(temp_config_structure)

    # Create experiments.toml with reference to non-existent dataset
    with open(temp_config_structure / "configs" / "experiments.toml", "w") as f:
        f.write("[[models]]\n")
        f.write('id = "exp1_model"\n')
        f.write('path = "models/exp1_model.toml"\n\n')
        f.write("[[experiments]]\n")
        f.write('id = "exp_missing"\n')
        f.write('dataset = "missing_dataset"\n')
        f.write('model = "exp1_model"\n')

    with pytest.raises(
        ValueError, match="Experiment 'exp_missing' references dataset id 'missing_dataset'"
    ):
        load_batch(
            case_config_path=temp_config_structure / "configs" / "experiments.toml",
        )


def test_load_experiments_rejects_unknown_comparison_experiment_filter(
    temp_config_structure: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comparison experiments filter cannot reference ids missing from [[experiments]]."""
    monkeypatch.chdir(temp_config_structure)
    with open(temp_config_structure / "configs" / "experiments.toml", "a") as f:
        f.write("\n[[datasets]]\n")
        f.write('id = "solutions"\n')
        f.write('path = "datasets/exp1_data.toml"\n\n')
        f.write("[[datasets]]\n")
        f.write('id = "gaussian-rhs"\n')
        f.write('path = "datasets/exp2_data.toml"\n\n')
        f.write("[[comparisons]]\n")
        f.write('id = "gaussian"\n')
        f.write('matrix_dataset = "solutions"\n')
        f.write('rhs_dataset = "gaussian-rhs"\n')
        f.write('experiments = ["missing-exp"]\n')

    with pytest.raises(
        ValueError,
        match="Comparison 'gaussian' experiments filter references unknown experiment ids: missing-exp",
    ):
        load_batch(
            case_config_path=temp_config_structure / "configs" / "experiments.toml",
        )


def test_load_experiments_no_experiments(
    temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test that ValueError is raised when no experiments defined."""
    monkeypatch.chdir(temp_config_structure)

    # Create empty experiments.toml
    with open(temp_config_structure / "configs" / "experiments.toml", "w") as f:
        f.write("")

    with pytest.raises(ValueError, match="No experiments defined"):
        load_batch(
            case_config_path=temp_config_structure / "configs" / "experiments.toml",
        )
