"""Integration tests for configuration loading (settings + load_experiment)."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.configuration.loader import load_experiment
from neuralls.configuration.domain import (
    ExperimentSpec,
    ExperimentWorkspace,
    RunnableExperiment,
)
from neuralls.configuration.paths import PathContext
from neuralls.configuration.settings import build_settings


@pytest.fixture
def sample_model_config(tmp_path: Path) -> Path:
    """Create a minimal model config TOML."""
    config_path = tmp_path / "model.toml"
    config_content = """
[SESSION]
name = "test-model"
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
enabled = true

[MLFLOW.client]
tracking_uri = "http://localhost:5000"
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def sample_data_config(tmp_path: Path) -> Path:
    """Create a minimal data config TOML."""
    config_path = tmp_path / "data.toml"
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


class TestBuildSettings:
    """Tests for build_settings function."""

    def test_build_settings_with_workspace(
        self,
        sample_model_config: Path,
        tmp_path: Path,
    ):
        """Test building settings with workspace paths injected."""
        from neuralls.configuration.domain import ExperimentWorkspace
        from neuralls.configuration.paths import PathContext

        # Create directories for dlkit validation
        root_dir = tmp_path / "root"
        root_dir.mkdir(parents=True)
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)

        workspace = ExperimentWorkspace(
            dataset_id="test-dataset",
            run_id="test-model",
            root_dir=root_dir,
            data_dir=data_dir,
        )

        path_ctx = PathContext(
            project_root=tmp_path / "project",
            output_root=tmp_path / "output",
            processed_root=tmp_path / "processed",
        )

        settings = build_settings(
            model_config_path=sample_model_config,
            workspace=workspace,
            path_context=path_ctx,
        )

        # Check workspace root injected
        assert Path(settings.TRAINING.trainer.default_root_dir) == workspace.root_dir

        # Check MLflow server paths injected from path_context
        assert settings.MLFLOW.server.backend_store_uri == path_ctx.mlflow_tracking_uri
        assert (
            settings.MLFLOW.server.artifacts_destination
            == path_ctx.mlflow_artifact_location
        )

        # Check MLflow client experiment/run names injected
        assert settings.MLFLOW.client.experiment_name == workspace.dataset_id
        assert settings.MLFLOW.client.run_name == workspace.run_id

        # Check PATHS injected
        assert settings.PATHS.project_root == str(path_ctx.project_root)
        assert settings.PATHS.processed_dir == str(path_ctx.processed_root)
        assert settings.PATHS.output_dir == str(path_ctx.output_root)

    def test_mlflow_tracking_uri_not_overridden(
        self,
        sample_model_config: Path,
        tmp_path: Path,
    ):
        """Test that client.tracking_uri from config is preserved."""
        from neuralls.configuration.domain import ExperimentWorkspace
        from neuralls.configuration.paths import PathContext

        workspace = ExperimentWorkspace(
            dataset_id="test",
            run_id="test",
            root_dir=tmp_path,
            data_dir=tmp_path,
        )

        path_ctx = PathContext(
            project_root=tmp_path,
            output_root=tmp_path,
            processed_root=tmp_path,
        )

        settings = build_settings(
            model_config_path=sample_model_config,
            workspace=workspace,
            path_context=path_ctx,
        )

        # Client tracking_uri should come from config (http URL)
        assert settings.MLFLOW.client.tracking_uri.startswith("http://")

        # Server backend_store_uri should come from path_ctx (sqlite URI)
        assert settings.MLFLOW.server.backend_store_uri.startswith("sqlite:///")


class TestLoadExperiment:
    """Integration tests for load_experiment function."""

    def test_load_experiment_success(
        self,
        sample_model_config: Path,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Test successful experiment loading."""
        output_root = tmp_path / "output"
        output_root.mkdir()

        experiment = load_experiment(
            model_config_path=sample_model_config,
            data_config_path=sample_data_config,
            output_root=output_root,
        )

        # Check types
        assert isinstance(experiment, RunnableExperiment)
        assert isinstance(experiment.spec, ExperimentSpec)
        assert isinstance(experiment.workspace, ExperimentWorkspace)

    def test_experiment_spec_fields(
        self,
        sample_model_config: Path,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Test experiment spec has correct fields."""
        experiment = load_experiment(
            sample_model_config,
            sample_data_config,
            output_root=tmp_path,
        )

        assert experiment.spec.id == "test-model"  # From SESSION.name
        assert experiment.spec.model_config_path == sample_model_config
        assert experiment.spec.data_config_path == sample_data_config
        assert experiment.spec.checkpoint_path is None

    def test_workspace_fields(
        self,
        sample_model_config: Path,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Test workspace has correct fields and paths."""
        output_root = tmp_path / "output"
        output_root.mkdir()

        experiment = load_experiment(
            sample_model_config,
            sample_data_config,
            output_root=output_root,
        )

        # Check identifiers
        assert experiment.workspace.dataset_id == "data"  # From config filename
        # workspace.run_id should have timestamp for uniqueness
        assert experiment.workspace.run_id.startswith("test-model-")
        assert len(experiment.workspace.run_id) > len("test-model-")

        # Check path structure (includes timestamped run_id)
        assert experiment.workspace.root_dir.parent == output_root / "data"
        assert experiment.workspace.root_dir.name.startswith("test-model-")

        # Check subdirectories created
        assert experiment.workspace.checkpoint_dir.exists()
        assert experiment.workspace.figures_dir.exists()
        assert experiment.workspace.predictions_dir.exists()

    def test_settings_integration(
        self,
        sample_model_config: Path,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Test settings are correctly configured."""
        output_root = tmp_path / "output"
        output_root.mkdir()

        experiment = load_experiment(
            sample_model_config,
            sample_data_config,
            output_root=output_root,
        )

        # Check workspace paths injected
        assert Path(experiment.settings.TRAINING.trainer.default_root_dir) == experiment.workspace.root_dir

        # Check MLflow configuration
        assert experiment.settings.MLFLOW.client.experiment_name == "data"
        # MLflow run_name should have timestamp for uniqueness
        assert experiment.settings.MLFLOW.client.run_name.startswith("test-model-")
        assert len(experiment.settings.MLFLOW.client.run_name) > len("test-model-")

        # MLflow server URIs derived from output_root
        expected_tracking = f"sqlite:///{(output_root / 'mlruns' / 'mlflow.db').as_posix()}"
        assert experiment.settings.MLFLOW.server.backend_store_uri == expected_tracking

        expected_artifacts = str((output_root / "mlartifacts").as_posix())
        assert (
            experiment.settings.MLFLOW.server.artifacts_destination == expected_artifacts
        )

    def test_load_with_default_output_root(
        self,
        sample_model_config: Path,
        sample_data_config: Path,
    ):
        """Test loading without output_root override uses default."""
        experiment = load_experiment(
            sample_model_config,
            sample_data_config,
        )

        # Should use DEFAULT_OUTPUT_DIR from constants
        from neuralls.constants import DEFAULT_OUTPUT_DIR

        assert str(DEFAULT_OUTPUT_DIR) in str(experiment.workspace.root_dir)

    def test_load_with_model_name_from_model_section(
        self,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Test using MODEL.name when SESSION.name not present."""
        # Create config without SESSION.name
        model_config = tmp_path / "model_only.toml"
        model_config.write_text("""
[MODEL]
name = "OnlyModelName"
module_path = "test.module"

[TRAINING]
[TRAINING.trainer]
max_epochs = 1

[DATASET]
name = "FlexibleDataset"

[MLFLOW]
enabled = true
[MLFLOW.client]
tracking_uri = "http://localhost:5000"
""")

        experiment = load_experiment(
            model_config,
            sample_data_config,
            output_root=tmp_path,
        )

        # Should use MODEL.name (spec.id = base, workspace.run_id = base + timestamp)
        assert experiment.spec.id == "OnlyModelName"
        assert experiment.workspace.run_id.startswith("OnlyModelName-")

    def test_load_missing_session_name_uses_model_name(
        self,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Test that MODEL.name is used when SESSION.name is missing."""
        # Create config without SESSION.name (dlkit requires MODEL.name)
        model_config = tmp_path / "no_session.toml"
        model_config.write_text("""
[MODEL]
name = "JustModelName"
module_path = "test.module"

[TRAINING]
[TRAINING.trainer]
max_epochs = 1

[DATASET]
name = "FlexibleDataset"

[MLFLOW]
enabled = true
[MLFLOW.client]
tracking_uri = "http://localhost:5000"
""")

        experiment = load_experiment(
            model_config,
            sample_data_config,
            output_root=tmp_path,
        )

        # Should use MODEL.name when SESSION.name is missing
        assert experiment.spec.id == "JustModelName"

    def test_path_context_single_source_of_truth(
        self,
        sample_model_config: Path,
        sample_data_config: Path,
        tmp_path: Path,
    ):
        """Test that output_root is the single source of truth for MLflow paths."""
        output_root = tmp_path / "master_output"
        output_root.mkdir()

        experiment = load_experiment(
            sample_model_config,
            sample_data_config,
            output_root=output_root,
        )

        # All MLflow paths should derive from output_root
        mlflow_db = output_root / "mlruns" / "mlflow.db"
        mlflow_artifacts = output_root / "mlartifacts"

        assert mlflow_db.as_posix() in experiment.settings.MLFLOW.server.backend_store_uri
        assert (
            mlflow_artifacts.as_posix()
            in experiment.settings.MLFLOW.server.artifacts_destination
        )

    def test_different_experiments_different_paths(
        self,
        sample_model_config: Path,
        tmp_path: Path,
    ):
        """Test that different datasets create different paths."""
        # Create two different data configs
        data_config_1 = tmp_path / "data1.toml"
        data_config_1.write_text("""
[flow]
dataset = "dataset-1"

[source]
matrix_path = "/tmp/test.txt"

[generation]
normalize = "matrix"
""")

        data_config_2 = tmp_path / "data2.toml"
        data_config_2.write_text("""
[flow]
dataset = "dataset-2"

[source]
matrix_path = "/tmp/test.txt"

[generation]
normalize = "matrix"
""")

        output_root = tmp_path / "output"
        output_root.mkdir()

        exp1 = load_experiment(sample_model_config, data_config_1, output_root)
        exp2 = load_experiment(sample_model_config, data_config_2, output_root)

        # Different data directories
        assert exp1.workspace.data_dir != exp2.workspace.data_dir

        # Different experiment roots
        assert exp1.workspace.root_dir != exp2.workspace.root_dir

        # But same output_root parent
        assert exp1.workspace.root_dir.parent.parent == output_root
        assert exp2.workspace.root_dir.parent.parent == output_root
