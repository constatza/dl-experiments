"""Tests for TOML loader module.

Tests cover TOML parsing and Pydantic validation of configuration files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.configuration.loaders.toml_loader import (
    ConfigLoadError,
    load_data_config,
    load_model_config,
    load_raw_toml,
    load_solver_config,
)


class TestLoadModelConfig:
    """Tests for load_model_config function."""

    def test_load_valid_model_config(self, tmp_path: Path):
        """Test loading a valid model config."""
        config_file = tmp_path / "linear.toml"
        config_file.write_text(
            """
[SESSION]
seed = 123
precision = "float32"

[MODEL]
name = "TestModel"
module_path = "test.module"
hidden_size = 256
"""
        )

        config = load_model_config(config_file)
        assert config.SESSION.seed == 123
        assert config.SESSION.precision == "float32"
        assert config.MODEL.name == "TestModel"
        assert config.MODEL.hidden_size == 256

    def test_load_model_config_missing_required_field(self, tmp_path: Path):
        """Test loading model config with missing required field."""
        config_file = tmp_path / "linear.toml"
        config_file.write_text(
            """
[SESSION]
seed = 123

[MODEL]
name = "TestModel"
# Missing module_path - should fail
"""
        )

        with pytest.raises(ConfigLoadError) as exc_info:
            load_model_config(config_file)
        assert "Invalid model config" in str(exc_info.value)

    def test_load_model_config_invalid_toml(self, tmp_path: Path):
        """Test loading invalid TOML syntax."""
        config_file = tmp_path / "linear.toml"
        config_file.write_text("invalid [ toml")

        with pytest.raises(ConfigLoadError) as exc_info:
            load_model_config(config_file)
        assert "Failed to load model config" in str(exc_info.value)

    def test_load_model_config_nonexistent_file(self, tmp_path: Path):
        """Test loading nonexistent file."""
        config_file = tmp_path / "nonexistent.toml"

        with pytest.raises(ConfigLoadError):
            load_model_config(config_file)


class TestLoadDataConfig:
    """Tests for load_data_config function."""

    def test_load_valid_data_config(self, tmp_path: Path):
        """Test loading a valid data config."""
        config_file = tmp_path / "data.toml"
        config_file.write_text(
            """
[flow]
id = "test_flow"

[source]
matrix_path = "/path/to/matrix.txt"

[generation]
normalize = "matrix"
shuffle = true
seed = 42

[[generation.strategy]]
name = "solution_archive"
samples = 1000

[output]
processed_dir = "/data/processed"

[test]
solutions_path = "/data/test/solutions.txt"
"""
        )

        config = load_data_config(config_file)
        assert config.flow.id == "test_flow"
        assert config.source.matrix_path == "/path/to/matrix.txt"
        assert config.generation.normalize == "matrix"
        assert len(config.generation.strategy) == 1
        assert config.generation.strategy[0].name == "solution_archive"
        assert config.output.processed_dir == "/data/processed"
        assert config.test.solutions_path == "/data/test/solutions.txt"

    def test_load_data_config_with_defaults(self, tmp_path: Path):
        """Test loading data config with mostly defaults."""
        config_file = tmp_path / "data.toml"
        config_file.write_text(
            """
[source]
# Minimal config

[generation]
# Use defaults
"""
        )

        config = load_data_config(config_file)
        assert config.generation.normalize == "matrix"  # Default
        assert config.generation.shuffle is True  # Default
        assert config.generation.seed == 42  # Default

    def test_load_data_config_invalid_field(self, tmp_path: Path):
        """Test loading data config with invalid field value."""
        config_file = tmp_path / "data.toml"
        config_file.write_text(
            """
[generation]
seed = -1

[[generation.strategy]]
name = "test"
residual_iters = 0  # Should be >= 1
"""
        )

        with pytest.raises(ConfigLoadError) as exc_info:
            load_data_config(config_file)
        assert "Invalid data config" in str(exc_info.value)


class TestLoadSolverConfig:
    """Tests for load_solver_config function."""

    def test_load_valid_solver_config(self, tmp_path: Path):
        """Test loading a valid solver config."""
        config_file = tmp_path / "solver.toml"
        output_root = tmp_path / "output"
        config_file.write_text(
            f"""
[general]
rtol = 1e-6
atol = 1e-12
max_iterations = 100
output_root = "{output_root}"

[[solvers]]
name = "test_solver"
type = "jacobi"
"""
        )

        config = load_solver_config(config_file)
        assert config.general.rtol == 1e-6
        assert config.general.atol == 1e-12
        assert len(config.solvers) == 1
        assert config.solvers[0].name == "test_solver"
        assert config.solvers[0].type == "jacobi"

    def test_load_solver_config_with_defaults(self, tmp_path: Path):
        """Test loading solver config with defaults."""
        config_file = tmp_path / "solver.toml"
        output_root = tmp_path / "output"
        config_file.write_text(
            f"""
[general]
# Use all defaults
output_root = "{output_root}"

[[solvers]]
name = "default_solver"
type = "none"
"""
        )

        config = load_solver_config(config_file)
        # Check defaults from constants
        assert config.general.max_iterations == 100  # Default
        assert config.general.stopping_criterion == "residual_norm"  # Default


class TestLoadRawToml:
    """Tests for load_raw_toml function."""

    def test_load_raw_toml_success(self, tmp_path: Path):
        """Test loading raw TOML successfully."""
        config_file = tmp_path / "test.toml"
        config_file.write_text(
            """
[section]
key = "value"
number = 42
"""
        )

        data = load_raw_toml(config_file)
        assert isinstance(data, dict)
        assert data["section"]["key"] == "value"
        assert data["section"]["number"] == 42

    def test_load_raw_toml_invalid_syntax(self, tmp_path: Path):
        """Test loading raw TOML with invalid syntax."""
        config_file = tmp_path / "test.toml"
        config_file.write_text("invalid [ syntax")

        with pytest.raises(ConfigLoadError):
            load_raw_toml(config_file)
