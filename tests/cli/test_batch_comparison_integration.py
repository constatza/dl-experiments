"""Integration tests for batch comparison workflow.

Tests the end-to-end batch comparison flow including configuration loading,
experiment building, and result aggregation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add root to path for script imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_methods import (
    ExperimentConfig,
    ExperimentResult,
    load_experiments_config,
    extract_experiments_list,
    extract_output_root,
    build_experiment_config,
)


@pytest.fixture
def sample_experiments_config(tmp_path: Path) -> Path:
    """Create a sample experiments.toml for testing."""
    config_path = tmp_path / "experiments.toml"
    config_path.write_text("""
[paths]
output_root = "/tmp/test-output"
mlruns_dir = "/tmp/mlruns"
mlartifacts_dir = "/tmp/mlartifacts"

[[experiments]]
model_template = "configs/ffnn.toml"
data_config = "data-configs/test-solutions.toml"
solver_config = "solver-configs/default.toml"

[[experiments]]
model_template = "configs/linear.toml"
data_config = "data-configs/test-eigenvector.toml"
solver_config = "solver-configs/cg.toml"

[[experiments]]
model_template = "configs/gnn.toml"
data_config = "data-configs/collect-504.toml"
""")
    return config_path


@pytest.fixture
def model_configs(tmp_path: Path) -> Path:
    """Create sample model configs."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()

    (configs_dir / "ffnn.toml").write_text("""
[SESSION]
name = "ffnn"
seed = 42
""")

    (configs_dir / "linear.toml").write_text("""
[SESSION]
name = "linear"
seed = 42
""")

    (configs_dir / "gnn.toml").write_text("""
[SESSION]
name = "gnn"
seed = 42
""")

    return tmp_path


class TestBatchConfigurationLoading:
    """Test batch configuration loading workflow."""

    def test_load_full_experiment_matrix(
        self, sample_experiments_config: Path
    ) -> None:
        """Test loading complete experiment matrix."""
        config = load_experiments_config(sample_experiments_config)
        experiments = extract_experiments_list(config)

        assert len(experiments) == 3
        assert all("model_template" in exp for exp in experiments)
        assert all("data_config" in exp for exp in experiments)

    def test_extract_all_experiment_components(
        self, sample_experiments_config: Path
    ) -> None:
        """Test extracting all components from config."""
        config = load_experiments_config(sample_experiments_config)
        experiments = extract_experiments_list(config)
        output_root = extract_output_root(config)

        assert output_root == Path("/tmp/test-output")
        assert experiments[0]["model_template"] == "configs/ffnn.toml"
        assert experiments[0]["solver_config"] == "solver-configs/default.toml"

    def test_default_solver_config_applied(
        self, sample_experiments_config: Path
    ) -> None:
        """Test that missing solver_config defaults to default.toml."""
        config = load_experiments_config(sample_experiments_config)
        experiments = extract_experiments_list(config)

        # Third experiment doesn't specify solver_config
        assert "solver_config" not in experiments[2]


class TestBatchExperimentBuilding:
    """Test building experiment configs in batch."""

    def test_build_all_experiment_configs(
        self, sample_experiments_config: Path, model_configs: Path
    ) -> None:
        """Test building experiment configs for all experiments."""
        config = load_experiments_config(sample_experiments_config)
        experiments = extract_experiments_list(config)
        output_root = extract_output_root(config)

        experiment_configs = [
            build_experiment_config(exp, model_configs, output_root)
            for exp in experiments
        ]

        assert len(experiment_configs) == 3
        assert all(isinstance(exp, ExperimentConfig) for exp in experiment_configs)

    def test_checkpoint_paths_unique_per_experiment(
        self, sample_experiments_config: Path, model_configs: Path
    ) -> None:
        """Test that each experiment has unique checkpoint path."""
        config = load_experiments_config(sample_experiments_config)
        experiments = extract_experiments_list(config)
        output_root = extract_output_root(config)

        experiment_configs = [
            build_experiment_config(exp, model_configs, output_root)
            for exp in experiments
        ]

        checkpoint_paths = [exp.checkpoint_path for exp in experiment_configs]

        # All checkpoint paths should be unique
        assert len(checkpoint_paths) == len(set(checkpoint_paths))

        # Verify path structure
        assert checkpoint_paths[0] == Path(
            "/tmp/test-output/test-solutions/ffnn/checkpoints/ffnn.ckpt"
        )
        assert checkpoint_paths[1] == Path(
            "/tmp/test-output/test-eigenvector/linear/checkpoints/linear.ckpt"
        )
        assert checkpoint_paths[2] == Path(
            "/tmp/test-output/collect-504/gnn/checkpoints/gnn.ckpt"
        )

    def test_solver_configs_properly_assigned(
        self, sample_experiments_config: Path, model_configs: Path
    ) -> None:
        """Test that solver configs are properly assigned to experiments."""
        config = load_experiments_config(sample_experiments_config)
        experiments = extract_experiments_list(config)
        output_root = extract_output_root(config)

        experiment_configs = [
            build_experiment_config(exp, model_configs, output_root)
            for exp in experiments
        ]

        # First experiment: explicit solver config
        assert experiment_configs[0].solver_config.name == "default.toml"

        # Second experiment: different explicit solver config
        assert experiment_configs[1].solver_config.name == "cg.toml"

        # Third experiment: defaults to default.toml
        assert experiment_configs[2].solver_config.name == "default.toml"


class TestExperimentResultAggregation:
    """Test aggregating results from multiple experiments."""

    def test_aggregate_mixed_results(self) -> None:
        """Test aggregating results with successes and failures."""
        results = [
            ExperimentResult(
                experiment=ExperimentConfig(
                    model_template=Path("m1.toml"),
                    data_config=Path("d1.toml"),
                    solver_config=Path("s1.toml"),
                    checkpoint_path=Path("c1.ckpt"),
                ),
                success=True,
            ),
            ExperimentResult(
                experiment=ExperimentConfig(
                    model_template=Path("m2.toml"),
                    data_config=Path("d2.toml"),
                    solver_config=Path("s2.toml"),
                    checkpoint_path=Path("c2.ckpt"),
                ),
                success=False,
                error_message="Checkpoint not found",
            ),
            ExperimentResult(
                experiment=ExperimentConfig(
                    model_template=Path("m3.toml"),
                    data_config=Path("d3.toml"),
                    solver_config=Path("s3.toml"),
                    checkpoint_path=Path("c3.ckpt"),
                ),
                success=True,
            ),
        ]

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        assert len(successful) == 2
        assert len(failed) == 1
        assert failed[0].error_message == "Checkpoint not found"

    def test_all_experiments_successful(self) -> None:
        """Test aggregation when all experiments succeed."""
        results = [
            ExperimentResult(
                experiment=ExperimentConfig(
                    model_template=Path(f"m{i}.toml"),
                    data_config=Path(f"d{i}.toml"),
                    solver_config=Path(f"s{i}.toml"),
                    checkpoint_path=Path(f"c{i}.ckpt"),
                ),
                success=True,
            )
            for i in range(5)
        ]

        assert all(r.success for r in results)
        assert not any(r.error_message for r in results)


class TestConfigurationValidation:
    """Test configuration validation in batch processing."""

    def test_invalid_experiments_config_format(self, tmp_path: Path) -> None:
        """Test handling of invalid experiment config format."""
        config_path = tmp_path / "bad.toml"
        config_path.write_text("""
[[experiments]]
# Missing required fields
solver_config = "solver-configs/default.toml"
""")

        config = load_experiments_config(config_path)
        experiments = extract_experiments_list(config)

        # Should have experiment but missing required fields
        assert len(experiments) == 1
        assert "model_template" not in experiments[0]

        # build_experiment_config will raise KeyError for missing fields
        output_root = Path("/tmp")
        with pytest.raises(KeyError):
            build_experiment_config(experiments[0], tmp_path, output_root)


class TestEndToEndBatchPipeline:
    """Test complete end-to-end batch processing pipeline."""

    def test_full_pipeline_execution(
        self, sample_experiments_config: Path, model_configs: Path
    ) -> None:
        """Test full pipeline from config load to experiment building."""
        # Step 1: Load configuration
        config = load_experiments_config(sample_experiments_config)

        # Step 2: Extract components
        experiments = extract_experiments_list(config)
        output_root = extract_output_root(config)

        # Step 3: Build experiment configs
        experiment_configs = [
            build_experiment_config(exp, model_configs, output_root)
            for exp in experiments
        ]

        # Verify complete pipeline
        assert len(experiment_configs) == 3

        # Verify each experiment has all required components
        for exp_config in experiment_configs:
            assert exp_config.model_template.exists()
            assert exp_config.data_config.name.endswith(".toml")
            assert exp_config.solver_config.name.endswith(".toml")
            assert exp_config.checkpoint_path.suffix == ".ckpt"

        # Verify paths follow convention
        for exp_config in experiment_configs:
            # Checkpoint path should contain model name twice:
            # once in directory, once in filename
            model_name = exp_config.model_template.stem
            # Path parts: [..., data_config_stem, model_name, "checkpoints", "model_name.ckpt"]
            assert exp_config.checkpoint_path.parent.name == "checkpoints"
            assert exp_config.checkpoint_path.parent.parent.name in [
                "ffnn",
                "linear",
                "gnn",
            ]
