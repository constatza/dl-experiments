"""Unit tests for run naming logic with ISO 8601 timestamps."""

from __future__ import annotations

import re
import time
from datetime import datetime
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
    """Tests for run_id generation with timestamps."""

    def test_run_id_includes_timestamp(
        self,
        model_config_without_session: Path,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Verify run_id format: {model_name}-{ISO_8601}."""
        experiment = load_experiment(
            model_config_without_session,
            sample_data_config,
            output_root=tmp_path / "output",
        )

        run_id = experiment.workspace.run_id

        # Check format: base_name-YYYY-MM-DDTHH:MM:SS
        pattern = r"^NormScaledLinearFFNN-\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"
        assert re.match(pattern, run_id), f"run_id '{run_id}' doesn't match expected pattern"

        # Verify base name is correct
        assert run_id.startswith("NormScaledLinearFFNN-")

    def test_run_id_timestamp_is_iso8601(
        self,
        model_config_without_session: Path,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Verify timestamp is valid ISO 8601 format."""
        experiment = load_experiment(
            model_config_without_session,
            sample_data_config,
            output_root=tmp_path / "output",
        )

        run_id = experiment.workspace.run_id

        # Extract timestamp part
        timestamp_str = run_id.split("-", maxsplit=1)[1]  # After first hyphen
        # NormScaledLinearFFNN-2025-12-30T14:30:45 -> 2025-12-30T14:30:45

        # Verify it can be parsed as ISO 8601
        try:
            parsed = datetime.fromisoformat(timestamp_str)
            assert isinstance(parsed, datetime)
        except ValueError as e:
            pytest.fail(f"Timestamp '{timestamp_str}' is not valid ISO 8601: {e}")

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

        run_id = experiment.workspace.run_id

        # Should use SESSION.name = "MyCustomSession" not MODEL.name = "FFNNModel"
        assert run_id.startswith("MyCustomSession-")
        assert not run_id.startswith("FFNNModel-")

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

        run_id = experiment.workspace.run_id

        # Should use MODEL.name = "GNNModel" not SESSION.name = "dlkit-session"
        assert run_id.startswith("GNNModel-")
        assert not run_id.startswith("dlkit-session-")

    def test_run_id_unique_across_multiple_loads(
        self,
        model_config_without_session: Path,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Verify each load_experiment() call generates unique run_id."""
        # Load first experiment
        exp1 = load_experiment(
            model_config_without_session,
            sample_data_config,
            output_root=tmp_path / "output1",
        )

        # Small delay to ensure timestamp differs
        time.sleep(1.1)

        # Load second experiment
        exp2 = load_experiment(
            model_config_without_session,
            sample_data_config,
            output_root=tmp_path / "output2",
        )

        # run_ids should be different due to timestamp
        assert exp1.workspace.run_id != exp2.workspace.run_id

        # Both should start with same base name
        assert exp1.workspace.run_id.startswith("NormScaledLinearFFNN-")
        assert exp2.workspace.run_id.startswith("NormScaledLinearFFNN-")

    def test_run_id_timestamp_is_recent(
        self,
        model_config_without_session: Path,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Verify timestamp reflects current time (within reasonable margin)."""
        before = datetime.now().replace(microsecond=0)

        experiment = load_experiment(
            model_config_without_session,
            sample_data_config,
            output_root=tmp_path / "output",
        )

        after = datetime.now().replace(microsecond=0)

        # Extract and parse timestamp from run_id
        run_id = experiment.workspace.run_id
        timestamp_str = run_id.split("-", maxsplit=1)[1]
        timestamp = datetime.fromisoformat(timestamp_str)

        # Timestamp should be between before and after (allowing 1 second margin)
        assert (before - timestamp).total_seconds() <= 1, (
            f"Timestamp {timestamp} is too far before {before}"
        )
        assert (timestamp - after).total_seconds() <= 1, (
            f"Timestamp {timestamp} is too far after {after}"
        )

    def test_experiment_spec_id_matches_run_id(
        self,
        model_config_without_session: Path,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Verify ExperimentSpec.id is base name, workspace.run_id has timestamp."""
        experiment = load_experiment(
            model_config_without_session,
            sample_data_config,
            output_root=tmp_path / "output",
        )

        # spec.id should be base name (logical identifier)
        assert experiment.spec.id == "NormScaledLinearFFNN"

        # workspace.run_id should have timestamp for uniqueness
        assert experiment.workspace.run_id.startswith("NormScaledLinearFFNN-")
        assert experiment.workspace.run_id != experiment.spec.id

    def test_workspace_directories_use_timestamped_run_id(
        self,
        model_config_without_session: Path,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Verify workspace directories incorporate timestamped run_id."""
        output_root = tmp_path / "output"

        experiment = load_experiment(
            model_config_without_session,
            sample_data_config,
            output_root=output_root,
        )

        run_id = experiment.workspace.run_id
        dataset_id = "test-dataset"  # From sample_data_config filename

        # Workspace root should be: output_root/dataset_id/run_id
        expected_root = output_root / dataset_id / run_id
        assert experiment.workspace.root_dir == expected_root

        # Checkpoint dir should include timestamped run_id in path
        assert str(run_id) in str(experiment.workspace.checkpoint_dir)


class TestRunNamingEdgeCases:
    """Edge cases and error scenarios for run naming."""

    def test_missing_model_name_raises_error(
        self,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Verify error when MODEL.name is empty string."""
        # Create config with empty model name
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
        """Verify run_id handles special characters in model name."""
        # Create config with special characters in name
        special_config = tmp_path / "special_model.toml"
        special_config_content = """
[SESSION]
name = "Model_v2.0-alpha"
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

        run_id = experiment.workspace.run_id

        # Should preserve special characters from model name
        assert run_id.startswith("Model_v2.0-alpha-")
        assert ":" in run_id  # ISO timestamp has colons
