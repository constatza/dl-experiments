"""Tests for pure functions in compare_methods.py.

Tests the configuration loading, extraction, and building logic
used by the batch comparison script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add root to path for script imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_methods import (
    ExperimentConfig,
    load_experiments_config,
    extract_experiments_list,
    extract_output_root,
    build_experiment_config,
    validate_checkpoint_exists,
)


class TestLoadExperimentsConfig:
    """Tests for load_experiments_config function."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Test loading a valid experiments config."""
        config_path = tmp_path / "experiments.toml"
        config_path.write_text("""
[paths]
output_root = "/tmp/output"

[[experiments]]
model_template = "configs/ffnn.toml"
data_config = "data-configs/test.toml"
solver_config = "solver-configs/default.toml"
""")

        config = load_experiments_config(config_path)

        assert isinstance(config, dict)
        assert "paths" in config
        assert "experiments" in config
        assert config["paths"]["output_root"] == "/tmp/output"

    def test_load_missing_config(self, tmp_path: Path) -> None:
        """Test loading nonexistent config raises FileNotFoundError."""
        config_path = tmp_path / "missing.toml"

        with pytest.raises(FileNotFoundError, match="Experiments config not found"):
            load_experiments_config(config_path)

    def test_load_invalid_toml(self, tmp_path: Path) -> None:
        """Test loading invalid TOML raises ValueError."""
        config_path = tmp_path / "invalid.toml"
        config_path.write_text("invalid toml content [[[")

        with pytest.raises(ValueError, match="Error loading experiments config"):
            load_experiments_config(config_path)


class TestExtractExperimentsList:
    """Tests for extract_experiments_list function."""

    def test_extract_experiments_list(self) -> None:
        """Test extracting experiments list from config dict."""
        config = {
            "paths": {"output_root": "/tmp"},
            "experiments": [
                {"model_template": "m1.toml", "data_config": "d1.toml"},
                {"model_template": "m2.toml", "data_config": "d2.toml"},
            ],
        }

        experiments = extract_experiments_list(config)

        assert len(experiments) == 2
        assert experiments[0]["model_template"] == "m1.toml"
        assert experiments[1]["data_config"] == "d2.toml"

    def test_extract_empty_experiments_list(self) -> None:
        """Test extracting empty experiments list raises ValueError."""
        config = {"paths": {"output_root": "/tmp"}, "experiments": []}

        with pytest.raises(ValueError, match="No experiments defined"):
            extract_experiments_list(config)

    def test_extract_missing_experiments_key(self) -> None:
        """Test missing experiments key raises ValueError."""
        config = {"paths": {"output_root": "/tmp"}}

        with pytest.raises(ValueError, match="No experiments defined"):
            extract_experiments_list(config)


class TestExtractOutputRoot:
    """Tests for extract_output_root function."""

    def test_extract_output_root_from_config(self) -> None:
        """Test extracting output root from config."""
        config = {
            "paths": {"output_root": "/custom/output/path"},
            "experiments": [],
        }

        output_root = extract_output_root(config)

        assert output_root == Path("/custom/output/path")

    def test_extract_output_root_uses_default(self) -> None:
        """Test default output root when not specified."""
        config = {"experiments": []}

        output_root = extract_output_root(config)

        assert output_root == Path("/data/projects/graph-cg/data/output")

    def test_extract_output_root_with_empty_paths(self) -> None:
        """Test default when paths section is empty."""
        config = {"paths": {}, "experiments": []}

        output_root = extract_output_root(config)

        assert output_root == Path("/data/projects/graph-cg/data/output")


class TestBuildExperimentConfig:
    """Tests for build_experiment_config function."""

    def test_build_experiment_config_basic(self, tmp_path: Path) -> None:
        """Test building experiment config from dict."""
        model_config = tmp_path / "configs" / "model.toml"
        model_config.parent.mkdir(parents=True)
        model_config.write_text("""
[SESSION]
name = "test_model"
""")

        experiment_dict = {
            "model_template": "configs/model.toml",
            "data_config": "data-configs/data.toml",
            "solver_config": "solver-configs/solver.toml",
        }

        output_root = Path("/tmp/output")

        exp_config = build_experiment_config(experiment_dict, tmp_path, output_root)

        assert isinstance(exp_config, ExperimentConfig)
        assert exp_config.model_template == tmp_path / "configs/model.toml"
        assert exp_config.data_config == tmp_path / "data-configs/data.toml"
        assert exp_config.solver_config == tmp_path / "solver-configs/solver.toml"
        assert exp_config.checkpoint_path == Path(
            "/tmp/output/data/test_model/checkpoints/test_model.ckpt"
        )

    def test_build_experiment_config_with_default_solver(self, tmp_path: Path) -> None:
        """Test building config uses default solver config when not specified."""
        model_config = tmp_path / "configs" / "model.toml"
        model_config.parent.mkdir(parents=True)
        model_config.write_text("""
[SESSION]
name = "ffnn"
""")

        experiment_dict = {
            "model_template": "configs/model.toml",
            "data_config": "data-configs/test.toml",
        }

        output_root = Path("/tmp/output")

        exp_config = build_experiment_config(experiment_dict, tmp_path, output_root)

        # Should default to solver-configs/default.toml
        assert exp_config.solver_config == tmp_path / "solver-configs/default.toml"

    def test_build_experiment_config_derives_checkpoint_path(self, tmp_path: Path) -> None:
        """Test that checkpoint path follows convention."""
        model_config = tmp_path / "configs" / "linear.toml"
        model_config.parent.mkdir(parents=True)
        model_config.write_text("""
[SESSION]
name = "linear"
""")

        experiment_dict = {
            "model_template": "configs/linear.toml",
            "data_config": "data-configs/collect-504.toml",
            "solver_config": "solver-configs/default.toml",
        }

        output_root = Path("/data/output")

        exp_config = build_experiment_config(experiment_dict, tmp_path, output_root)

        expected_checkpoint = Path(
            "/data/output/collect-504/linear/checkpoints/linear.ckpt"
        )
        assert exp_config.checkpoint_path == expected_checkpoint


class TestValidateCheckpointExists:
    """Tests for validate_checkpoint_exists function."""

    def test_validate_existing_checkpoint(self, tmp_path: Path) -> None:
        """Test validation passes for existing checkpoint."""
        checkpoint_path = tmp_path / "checkpoints" / "model.ckpt"
        checkpoint_path.parent.mkdir(parents=True)
        checkpoint_path.touch()

        exp_config = ExperimentConfig(
            model_template=Path("model.toml"),
            data_config=Path("data.toml"),
            solver_config=Path("solver.toml"),
            checkpoint_path=checkpoint_path,
        )

        error = validate_checkpoint_exists(exp_config)

        assert error is None

    def test_validate_missing_checkpoint(self, tmp_path: Path) -> None:
        """Test validation returns error for missing checkpoint."""
        checkpoint_path = tmp_path / "nonexistent" / "model.ckpt"

        exp_config = ExperimentConfig(
            model_template=Path("model.toml"),
            data_config=Path("data.toml"),
            solver_config=Path("solver.toml"),
            checkpoint_path=checkpoint_path,
        )

        error = validate_checkpoint_exists(exp_config)

        assert error is not None
        assert "Checkpoint not found" in error
        assert str(checkpoint_path) in error


class TestExperimentConfigDataclass:
    """Tests for ExperimentConfig dataclass."""

    def test_experiment_config_is_frozen(self) -> None:
        """Test that ExperimentConfig is immutable."""
        exp_config = ExperimentConfig(
            model_template=Path("model.toml"),
            data_config=Path("data.toml"),
            solver_config=Path("solver.toml"),
            checkpoint_path=Path("checkpoint.ckpt"),
        )

        with pytest.raises(AttributeError):
            exp_config.model_template = Path("other.toml")  # type: ignore[misc]

    def test_experiment_config_all_fields(self) -> None:
        """Test ExperimentConfig contains all required fields."""
        exp_config = ExperimentConfig(
            model_template=Path("configs/ffnn.toml"),
            data_config=Path("data-configs/test.toml"),
            solver_config=Path("solver-configs/default.toml"),
            checkpoint_path=Path("/output/test/ffnn/checkpoints/ffnn.ckpt"),
        )

        assert exp_config.model_template == Path("configs/ffnn.toml")
        assert exp_config.data_config == Path("data-configs/test.toml")
        assert exp_config.solver_config == Path("solver-configs/default.toml")
        assert exp_config.checkpoint_path == Path(
            "/output/test/ffnn/checkpoints/ffnn.ckpt"
        )


class TestIntegrationWithRealConfig:
    """Integration tests using realistic configuration."""

    def test_load_and_extract_real_config(self, tmp_path: Path) -> None:
        """Test full pipeline with realistic config."""
        config_path = tmp_path / "experiments.toml"
        config_path.write_text("""
[paths]
output_root = "/data/projects/graph-cg/data/output"
mlruns_dir = "/data/projects/graph-cg/data/mlruns"

[[experiments]]
model_template = "configs/ffnn.toml"
data_config = "data-configs/test-solutions.toml"
solver_config = "solver-configs/default.toml"

[[experiments]]
model_template = "configs/linear.toml"
data_config = "data-configs/test-eigenvector.toml"
solver_config = "solver-configs/cg.toml"
""")

        # Load config
        config = load_experiments_config(config_path)
        assert config["paths"]["output_root"] == "/data/projects/graph-cg/data/output"

        # Extract experiments
        experiments = extract_experiments_list(config)
        assert len(experiments) == 2

        # Extract output root
        output_root = extract_output_root(config)
        assert output_root == Path("/data/projects/graph-cg/data/output")

        # Verify experiment structure
        assert experiments[0]["model_template"] == "configs/ffnn.toml"
        assert experiments[0]["data_config"] == "data-configs/test-solutions.toml"
        assert experiments[1]["model_template"] == "configs/linear.toml"
