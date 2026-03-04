"""Unit tests for run naming logic (no timestamps — temp dir guarantees uniqueness)."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.configuration.loader import load_experiment


@pytest.fixture
def model_config_with_session(tmp_path: Path) -> Path:
    """Create a model config with SESSION.name set."""
    config_path = tmp_path / "model_with_session.toml"
    config_content = """
[SESSION]
name = "MyCustomSession"
seed = 42

[MODEL]
name = "FFNNModel"
module_path = "dlkit.nn.ffnn"

[TRAINING]
[TRAINING.trainer]
max_epochs = 1

[DATASET]
name = "FlexibleDataset"

[MLFLOW]
enabled = false
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def model_config_without_session(tmp_path: Path) -> Path:
    """Create a model config without SESSION.name (uses MODEL.name)."""
    config_path = tmp_path / "model_no_session.toml"
    config_content = """
[SESSION]
seed = 42

[MODEL]
name = "NormScaledLinearFFNN"
module_path = "dlkit.nn.ffnn"

[TRAINING]
[TRAINING.trainer]
max_epochs = 1

[DATASET]
name = "FlexibleDataset"

[MLFLOW]
enabled = false
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def model_config_with_dlkit_default_session(tmp_path: Path) -> Path:
    """Create a model config with dlkit's default SESSION.name."""
    config_path = tmp_path / "model_dlkit_default.toml"
    config_content = """
[SESSION]
name = "dlkit-session"
seed = 42

[MODEL]
name = "GNNModel"
module_path = "dlkit.nn.gnn"

[TRAINING]
[TRAINING.trainer]
max_epochs = 1

[DATASET]
name = "FlexibleDataset"

[MLFLOW]
enabled = false
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def sample_data_config(tmp_path: Path) -> Path:
    """Create a minimal data config TOML."""
    config_path = tmp_path / "test-dataset.toml"
    config_content = """
[flow]
dataset = "test-data"

[source]
matrix_path = "/tmp/test_matrix.txt"

[generation]
normalize = "matrix"
"""
    config_path.write_text(config_content)
    return config_path


class TestRunNaming:
    """Tests for run_id generation — plain model name, no timestamp."""

    def test_run_id_equals_model_name(
        self,
        model_config_without_session: Path,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Verify run_id is exactly the model name with no timestamp suffix."""
        experiment = load_experiment(
            model_config_without_session,
            sample_data_config,
            output_root=tmp_path / "output",
        )

        assert experiment.workspace.run_id == "NormScaledLinearFFNN"

    def test_run_id_uses_session_name_when_set(
        self,
        model_config_with_session: Path,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Verify SESSION.name takes precedence over MODEL.name."""
        experiment = load_experiment(
            model_config_with_session,
            sample_data_config,
            output_root=tmp_path / "output",
        )

        assert experiment.workspace.run_id == "MyCustomSession"

    def test_run_id_uses_model_name_when_session_is_dlkit_default(
        self,
        model_config_with_dlkit_default_session: Path,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Verify dlkit-session default is treated as unset, uses MODEL.name."""
        experiment = load_experiment(
            model_config_with_dlkit_default_session,
            sample_data_config,
            output_root=tmp_path / "output",
        )

        assert experiment.workspace.run_id == "GNNModel"

    def test_run_id_is_stable_across_loads(
        self,
        model_config_without_session: Path,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Verify the same config always produces the same run_id."""
        exp1 = load_experiment(
            model_config_without_session,
            sample_data_config,
            output_root=tmp_path / "output1",
        )
        exp2 = load_experiment(
            model_config_without_session,
            sample_data_config,
            output_root=tmp_path / "output2",
        )

        assert exp1.workspace.run_id == exp2.workspace.run_id == "NormScaledLinearFFNN"

    def test_experiment_spec_id_matches_run_id(
        self,
        model_config_without_session: Path,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Verify ExperimentSpec.id and workspace.run_id are both the base name."""
        experiment = load_experiment(
            model_config_without_session,
            sample_data_config,
            output_root=tmp_path / "output",
        )

        assert experiment.spec.id == "NormScaledLinearFFNN"
        assert experiment.workspace.run_id == "NormScaledLinearFFNN"
        assert experiment.spec.id == experiment.workspace.run_id

    def test_workspace_directories_use_run_id(
        self,
        model_config_without_session: Path,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Verify workspace directories incorporate the run_id."""
        output_root = tmp_path / "output"

        experiment = load_experiment(
            model_config_without_session,
            sample_data_config,
            output_root=output_root,
        )

        run_id = experiment.workspace.run_id
        dataset_id = "test-dataset"  # From sample_data_config filename

        expected_root = output_root / dataset_id / run_id
        assert experiment.workspace.root_dir == expected_root
        assert str(run_id) in str(experiment.workspace.checkpoint_dir)


class TestRunNamingEdgeCases:
    """Edge cases and error scenarios for run naming."""

    def test_missing_model_name_raises_error(
        self,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Verify error when MODEL.name is empty string."""
        bad_config = tmp_path / "bad_model.toml"
        bad_config_content = """
[SESSION]
seed = 42

[MODEL]
name = ""
module_path = "dlkit.nn.ffnn"

[TRAINING]
[TRAINING.trainer]
max_epochs = 1

[DATASET]
name = "FlexibleDataset"

[MLFLOW]
enabled = false
"""
        bad_config.write_text(bad_config_content)

        with pytest.raises(ValueError, match="Model name missing"):
            load_experiment(
                bad_config,
                sample_data_config,
                output_root=tmp_path / "output",
            )

    def test_run_id_handles_special_characters_in_model_name(
        self,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Verify run_id preserves special characters from model name."""
        special_config = tmp_path / "special_model.toml"
        special_config_content = """
[SESSION]
name = "Model_release-alpha"
seed = 42

[MODEL]
name = "TestModel"
module_path = "dlkit.nn.ffnn"

[TRAINING]
[TRAINING.trainer]
max_epochs = 1

[DATASET]
name = "FlexibleDataset"

[MLFLOW]
enabled = false
"""
        special_config.write_text(special_config_content)

        experiment = load_experiment(
            special_config,
            sample_data_config,
            output_root=tmp_path / "output",
        )

        assert experiment.workspace.run_id == "Model_release-alpha"
