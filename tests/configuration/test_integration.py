"""Integration tests for configuration loading (settings + load_experiment)."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.composition.experiments.assembler import load_experiment
from neuralls.platform.config.models.data_models import DataConfigFile, OutputConfig
from neuralls.platform.config.models.workspace import (
    ExperimentSpec,
    ExperimentWorkspace,
    RunnableExperiment,
)
from neuralls.platform.config.dlkit_bridge import build_settings, load_model_config
from neuralls.platform.config.resolution import PathContext
from tests.path_assertions import assert_local_path_eq


@pytest.fixture
def sample_model_config(tmp_path: Path) -> Path:
    """Create a minimal model config TOML."""
    config_path = tmp_path / "model.toml"
    config_content = """
[SESSION]
name = "test-model"
seed = 42
workflow = "train"

[MODEL]
name = "TestModel"
module_path = "dlkit.nn"

[TRAINING]
[TRAINING.trainer]
max_epochs = 1

[[TRAINING.optimizer.stages]]

[TRAINING.optimizer.stages.optimizer]
name = "AdamW"
lr = 1e-3

[TRAINING.optimizer.stages.trigger]
at_epoch = 200

[[TRAINING.optimizer.stages]]

[TRAINING.optimizer.stages.optimizer]
name = "LBFGS"
lr = 1.0

[DATASET]
name = "FlexibleDataset"
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def sample_data_config(tmp_path: Path) -> Path:
    """Create a minimal data config TOML."""
    config_path = tmp_path / "data.toml"
    matrix_path = tmp_path / "test_matrix.txt"
    config_content = f"""
id = "test-data"

[source]
matrix_path = "{matrix_path.as_posix()}"

[generation]
normalize = "matrix"
"""
    config_path.write_text(config_content)
    return config_path


class TestBuildSettings:
    """Tests for build_settings function."""

    def test_load_model_config_loads_explicit_optimization_workflows(
        self,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Optimization workflows must declare SESSION.workflow explicitly."""
        from dlkit.infrastructure.config.workflow_configs import OptimizationWorkflowConfig

        config_path = tmp_path / "optuna-model.toml"
        config_path.write_text(
            """
[SESSION]
name = "optuna-model"
seed = 42
workflow = "optimize"

[MODEL]
name = "TestModel"
module_path = "dlkit.nn"

[TRAINING]
[TRAINING.trainer]
max_epochs = 1
[TRAINING.optimizer.default_optimizer]
name = "AdamW"
lr = 1e-3

[DATASET]
name = "FlexibleDataset"

[OPTUNA]
enabled = true
n_trials = 2
"""
        )

        settings = load_model_config(config_path, neuralls_settings)
        assert isinstance(settings, OptimizationWorkflowConfig)

    def test_load_model_config_does_not_promote_optuna_enabled_configs(
        self,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Optuna-enabled training configs stay training without explicit optimize workflow."""
        from dlkit.infrastructure.config.workflow_configs import TrainingWorkflowConfig

        config_path = tmp_path / "train-optuna-model.toml"
        config_path.write_text(
            """
[SESSION]
name = "train-optuna-model"
seed = 42
workflow = "train"

[MODEL]
name = "TestModel"
module_path = "dlkit.nn"

[TRAINING]
[TRAINING.trainer]
max_epochs = 1
[TRAINING.optimizer.default_optimizer]
name = "AdamW"
lr = 1e-3

[DATASET]
name = "FlexibleDataset"

[OPTUNA]
enabled = true
n_trials = 2
"""
        )

        settings = load_model_config(config_path, neuralls_settings)
        assert isinstance(settings, TrainingWorkflowConfig)

    def test_load_model_config_rejects_legacy_optimization_section(
        self,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Top-level OPTIMIZATION sections are rejected after the hard cutover."""
        config_path = tmp_path / "legacy-optimization-model.toml"
        config_path.write_text(
            """
[SESSION]
name = "legacy-optimization-model"
seed = 42
workflow = "train"

[MODEL]
name = "TestModel"
module_path = "dlkit.nn"

[TRAINING]
[TRAINING.trainer]
max_epochs = 1

[OPTIMIZATION]

[[OPTIMIZATION.stages]]
[OPTIMIZATION.stages.optimizer]
name = "AdamW"
lr = 1e-3
"""
        )

        with pytest.raises(Exception, match="OPTIMIZATION"):
            load_model_config(config_path, neuralls_settings)

    def test_build_settings_with_workspace(
        self,
        sample_model_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Test building settings with workspace paths injected."""
        from neuralls.platform.config.models.workspace import ExperimentWorkspace

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
            data_cfg=DataConfigFile(
                id="test-dataset",
                output=OutputConfig(data_dir=path_ctx.processed_root),
            ),
            settings=neuralls_settings,
            output_override=path_ctx.output_root,
        )

        # Check workspace root injected
        assert Path(settings.TRAINING.trainer.default_root_dir) == workspace.root_dir

        # Check MLflow remains enabled without embedding runtime naming.
        assert settings.MLFLOW is not None
        assert not hasattr(settings.MLFLOW, "client")
        assert not hasattr(settings.MLFLOW, "server")

        # Check PATHS injected
        assert settings.PATHS is not None
        assert getattr(settings.PATHS, "project_root") != ""
        assert_local_path_eq(getattr(settings.PATHS, "processed_dir"), path_ctx.processed_root)
        assert_local_path_eq(settings.PATHS.output_dir, path_ctx.output_root)

    def test_mlflow_runtime_fields_are_env_only(
        self,
        sample_model_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Runtime MLflow infrastructure should not be stored in settings."""
        from neuralls.platform.config.models.workspace import ExperimentWorkspace

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
            data_cfg=DataConfigFile(
                id="test",
                output=OutputConfig(data_dir=path_ctx.processed_root),
            ),
            settings=neuralls_settings,
            output_override=path_ctx.output_root,
        )

        assert settings.MLFLOW is not None
        assert not hasattr(settings.MLFLOW, "tracking_uri")
        assert not hasattr(settings.MLFLOW, "artifacts_destination")


class TestLoadExperiment:
    """Integration tests for load_experiment function."""

    def test_load_experiment_success(
        self,
        sample_model_config: Path,
        sample_data_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Test successful experiment loading."""
        output_root = tmp_path / "output"
        output_root.mkdir()

        experiment = load_experiment(
            model_config_path=sample_model_config,
            data_config_path=sample_data_config,
            neuralls_settings=neuralls_settings,
            output_root=output_root,
            dataset_registry_id=sample_data_config.stem,
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
        neuralls_settings,
    ) -> None:
        """Test experiment spec has correct fields."""
        experiment = load_experiment(
            sample_model_config,
            sample_data_config,
            neuralls_settings=neuralls_settings,
            output_root=tmp_path,
            dataset_registry_id=sample_data_config.stem,
        )

        assert experiment.spec.experiment_id == "test-model"  # From SESSION.name
        assert experiment.spec.model_config_path == sample_model_config
        assert experiment.spec.data_config_path == sample_data_config
        assert experiment.spec.checkpoint_path is None

    def test_workspace_fields(
        self,
        sample_model_config: Path,
        sample_data_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Test workspace has correct fields and paths."""
        output_root = tmp_path / "output"
        output_root.mkdir()

        experiment = load_experiment(
            sample_model_config,
            sample_data_config,
            neuralls_settings=neuralls_settings,
            output_root=output_root,
            dataset_registry_id=sample_data_config.stem,
        )

        # Check identifiers
        assert experiment.workspace.dataset_id == "data"  # From config filename
        assert experiment.workspace.run_id == "test-model"

        # Check path structure
        assert experiment.workspace.root_dir.parent == output_root / "data"
        assert experiment.workspace.root_dir.name == "test-model"

        # Check subdirectories created
        assert experiment.workspace.checkpoint_dir.exists()
        assert experiment.workspace.figures_dir.exists()
        assert experiment.workspace.predictions_dir.exists()

    def test_settings_integration(
        self,
        sample_model_config: Path,
        sample_data_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Test settings are correctly configured."""
        output_root = tmp_path / "output"
        output_root.mkdir()

        experiment = load_experiment(
            sample_model_config,
            sample_data_config,
            neuralls_settings=neuralls_settings,
            output_root=output_root,
            dataset_registry_id=sample_data_config.stem,
        )

        # Check workspace paths injected
        assert (
            Path(experiment.settings.TRAINING.trainer.default_root_dir)
            == experiment.workspace.root_dir
        )

        # Check MLflow remains enabled without embedding runtime naming.
        assert experiment.settings.MLFLOW is not None
        assert not hasattr(experiment.settings.MLFLOW, "tracking_uri")
        assert not hasattr(experiment.settings.MLFLOW, "artifacts_destination")

    def test_load_with_default_output_root(
        self,
        sample_model_config: Path,
        sample_data_config: Path,
        neuralls_settings,
    ) -> None:
        """Test loading without output_root override uses default."""
        experiment = load_experiment(
            sample_model_config,
            sample_data_config,
            neuralls_settings=neuralls_settings,
            dataset_registry_id=sample_data_config.stem,
        )

        assert str(neuralls_settings.output_dir) in str(experiment.workspace.root_dir)

    def test_load_with_model_name_from_model_section(
        self,
        sample_data_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Test using MODEL.name when SESSION.name not present."""
        # Create config without SESSION.name
        model_config = tmp_path / "model_only.toml"
        model_config.write_text("""
[SESSION]
seed = 42
workflow = "train"

[MODEL]
name = "OnlyModelName"
module_path = "dlkit.nn"

[TRAINING]
[TRAINING.trainer]
max_epochs = 1

[DATASET]
name = "FlexibleDataset"
""")

        experiment = load_experiment(
            model_config,
            sample_data_config,
            neuralls_settings=neuralls_settings,
            output_root=tmp_path,
            dataset_registry_id=sample_data_config.stem,
        )

        assert experiment.spec.experiment_id == "OnlyModelName"
        assert experiment.workspace.run_id == "OnlyModelName"

    def test_load_missing_session_name_uses_model_name(
        self,
        sample_data_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Test that MODEL.name is used when SESSION.name is missing."""
        # Create config without SESSION.name (dlkit requires MODEL.name)
        model_config = tmp_path / "no_session.toml"
        model_config.write_text("""
[SESSION]
seed = 42
workflow = "train"

[MODEL]
name = "JustModelName"
module_path = "dlkit.nn"

[TRAINING]
[TRAINING.trainer]
max_epochs = 1

[DATASET]
name = "FlexibleDataset"
""")

        experiment = load_experiment(
            model_config,
            sample_data_config,
            neuralls_settings=neuralls_settings,
            output_root=tmp_path,
            dataset_registry_id=sample_data_config.stem,
        )

        # Should use MODEL.name when SESSION.name is missing
        assert experiment.spec.experiment_id == "JustModelName"

    def test_path_context_single_source_of_truth(
        self,
        sample_model_config: Path,
        sample_data_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Test that output_root remains the single source of truth for runtime paths."""
        output_root = tmp_path / "master_output"
        output_root.mkdir()

        experiment = load_experiment(
            sample_model_config,
            sample_data_config,
            neuralls_settings=neuralls_settings,
            output_root=output_root,
            dataset_registry_id=sample_data_config.stem,
        )

        assert_local_path_eq(experiment.settings.PATHS.output_dir, output_root)
        assert experiment.workspace.root_dir.parent.parent == output_root

    def test_different_experiments_different_paths(
        self,
        sample_model_config: Path,
        tmp_path: Path,
        neuralls_settings,
    ) -> None:
        """Test that different datasets create different paths."""
        # Create two different data configs
        matrix_path = tmp_path / "test.txt"
        data_config_1 = tmp_path / "data1.toml"
        data_config_1.write_text(f"""
id = "dataset-1"

[source]
matrix_path = "{matrix_path.as_posix()}"

[generation]
normalize = "matrix"
""")

        data_config_2 = tmp_path / "data2.toml"
        data_config_2.write_text(f"""
id = "dataset-2"

[source]
matrix_path = "{matrix_path.as_posix()}"

[generation]
normalize = "matrix"
""")

        output_root = tmp_path / "output"
        output_root.mkdir()

        exp1 = load_experiment(
            sample_model_config,
            data_config_1,
            neuralls_settings=neuralls_settings,
            output_root=output_root,
            dataset_registry_id=data_config_1.stem,
        )
        exp2 = load_experiment(
            sample_model_config,
            data_config_2,
            neuralls_settings=neuralls_settings,
            output_root=output_root,
            dataset_registry_id=data_config_2.stem,
        )

        # Different data directories
        assert exp1.workspace.data_dir != exp2.workspace.data_dir

        # Different experiment roots
        assert exp1.workspace.root_dir != exp2.workspace.root_dir

        # But same output_root parent
        assert exp1.workspace.root_dir.parent.parent == output_root
        assert exp2.workspace.root_dir.parent.parent == output_root
