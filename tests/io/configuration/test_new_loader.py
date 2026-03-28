"""Tests for the new experiment-centric configuration loader."""

from __future__ import annotations

from pathlib import Path
import importlib.util

import pytest
from neuralls.shared.workspace import ExperimentWorkspace
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

    # Master config (NEW FORMAT: [[experiment]] entries)
    with open(project_root / "configs" / "experiments.toml", "w") as f:
        f.write('project_root = ".."\n')
        f.write(f'output_dir = "{project_root / "output"}"\n\n')
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
        f.write("[[experiment]]\n")
        f.write('id = "exp1"\n')
        f.write('dataset = "exp1_data"\n')
        f.write('model = "exp1_model"\n')
        f.write(f'checkpoint_path = "{project_root / "checkpoints" / "exp1.ckpt"}"\n\n')
        f.write("# Experiment 2: Config without checkpoint (will warn)\n")
        f.write("[[experiment]]\n")
        f.write('id = "exp2"\n')
        f.write('dataset = "exp2_data"\n')
        f.write('model = "exp2_model"\n')

    # Dataset configs
    with open(project_root / "configs" / "datasets" / "exp1_data.toml", "w") as f:
        f.write("[flow]\n\n")
        f.write("[source]\n")
        f.write(f'matrix_path = "{matrix_path}"\n\n')
        f.write("[generation]\n")
        f.write('normalize = "matrix"\n')
        f.write("shuffle = false\n")
        f.write("seed = 42\n\n")
        f.write("[output]\n")
        f.write(f'data_dir = "{project_root / "data" / "processed"}"\n')

    with open(project_root / "configs" / "datasets" / "exp2_data.toml", "w") as f:
        f.write("[flow]\n\n")
        f.write("[source]\n")
        f.write(f'matrix_path = "{matrix2_path}"\n\n')
        f.write("[generation]\n")
        f.write('normalize = "matrix"\n')
        f.write("shuffle = false\n")
        f.write("seed = 42\n\n")
        f.write("[output]\n")
        f.write(f'data_dir = "{project_root / "data" / "processed"}"\n')

    # Model configs
    with open(project_root / "configs" / "models" / "exp1_model.toml", "w") as f:
        f.write("[SESSION]\n")
        f.write('name = "exp1_model"\n\n')
        f.write("[MODEL]\n")
        f.write('name = "TestModel"\n')
        f.write('module_path = "test.module"\n')

    with open(project_root / "configs" / "models" / "exp2_model.toml", "w") as f:
        f.write("[SESSION]\n")
        f.write('name = "exp2_model"\n\n')
        f.write("[MODEL]\n")
        f.write('name = "TestModel2"\n')
        f.write('module_path = "test.module2"\n')

    return project_root


def test_load_experiments_success(temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that load_batch correctly loads experiments with NEW format."""
    monkeypatch.chdir(temp_config_structure)

    batch = load_batch(
        master_config_path=temp_config_structure / "configs" / "experiments.toml",
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
        f.write('project_root = ".."\n')
        f.write(f'output_dir = "{temp_config_structure / "output"}"\n\n')
        f.write("[[models]]\n")
        f.write('id = "exp1_model"\n')
        f.write('path = "models/exp1_model.toml"\n\n')
        f.write("[[experiment]]\n")
        f.write('id = "exp_missing"\n')
        f.write('dataset = "missing_dataset"\n')
        f.write('model = "exp1_model"\n')

    with pytest.raises(
        ValueError, match="Experiment 'exp_missing' references dataset id 'missing_dataset'"
    ):
        load_batch(
            master_config_path=temp_config_structure / "configs" / "experiments.toml",
        )


def test_load_experiments_rejects_unknown_comparison_experiment_reference(
    temp_config_structure: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comparison configs cannot reference experiment ids missing from [[experiments]]."""
    monkeypatch.chdir(temp_config_structure)
    comparison_dir = temp_config_structure / "configs" / "comparison"
    comparison_dir.mkdir()
    (comparison_dir / "linear.toml").write_text(
        "\n".join(
            [
                "[general]",
                "",
                "[general.params]",
                "rtol = 1.0e-6",
                "atol = 1.0e-14",
                "max_iterations = 10",
                'stopping_criterion = "residual_norm"',
                "m_max = 20",
                "",
                "[general.data]",
                f'matrix_path = "{temp_config_structure / "data" / "matrix.txt"}"',
                f'rhs_path = "{temp_config_structure / "data" / "matrix2.txt"}"',
                "",
                "[[preconditioners]]",
                'name = "bound-neural"',
                'type = "neural"',
                'experiment = "missing-exp"',
                'model_ref = { source = "registered", name = "NormScaledLinearFFNN", alias = "solutions" }',
            ]
        )
    )
    with open(temp_config_structure / "configs" / "experiments.toml", "a") as f:
        f.write("\n[[comparisons]]\n")
        f.write('id = "linear"\n')
        f.write('path = "comparison/linear.toml"\n')

    with pytest.raises(
        ValueError,
        match="Comparison 'linear' neural preconditioner 'bound-neural' references experiment id 'missing-exp'",
    ):
        load_batch(
            master_config_path=temp_config_structure / "configs" / "experiments.toml",
        )


def test_load_experiments_no_experiments(
    temp_config_structure: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test that ValueError is raised when no experiments defined."""
    monkeypatch.chdir(temp_config_structure)

    # Create empty experiments.toml
    with open(temp_config_structure / "configs" / "experiments.toml", "w") as f:
        f.write('project_root = ".."\n')
        f.write(f'output_dir = "{temp_config_structure / "output"}"\n')

    with pytest.raises(ValueError, match="No experiments defined"):
        load_batch(
            master_config_path=temp_config_structure / "configs" / "experiments.toml",
        )
