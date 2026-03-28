"""Tests for TOML loader module.

Tests cover TOML parsing and Pydantic validation of configuration files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.platform.config.loaders import (
    load_data_config,
    load_comparison_config,
    load_raw_toml,
)

import tomllib
from pydantic import ValidationError


class TestLoadDataConfig:
    """Tests for load_data_config function."""

    def test_load_valid_data_config(self, tmp_path: Path):
        """Test loading a valid data config."""
        config_file = tmp_path / "data.toml"
        matrix_path = tmp_path / "matrix.txt"
        processed_dir = tmp_path / "processed"
        solutions_path = tmp_path / "test" / "solutions.txt"
        config_file.write_text(
            f"""
[flow]
dataset = "test_dataset"

[source]
matrix_path = "{matrix_path}"

[generation]
normalize = "matrix"
shuffle = true
seed = 42

[[generation.strategy]]
name = "solution_archive"
samples = 1000

[output]
data_dir = "{processed_dir}"

[test]
solutions_path = "{solutions_path}"
"""
        )

        config = load_data_config(config_file)
        assert config.flow.dataset == "test_dataset"
        assert config.source.matrix_path == str(matrix_path)
        assert config.generation.normalize == "matrix"
        assert len(config.generation.strategy) == 1
        assert config.generation.strategy[0].name == "solution_archive"
        assert config.output.data_dir == processed_dir
        assert config.test.solutions_path == str(solutions_path)

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

        with pytest.raises(ValidationError) as exc_info:
            load_data_config(config_file)
        assert "validation error for DataConfigFile" in str(exc_info.value)


class TestLoadComparisonConfig:
    """Tests for load_comparison_config function."""

    def test_load_valid_comparison_config(self, tmp_path: Path):
        """Test loading a valid comparison config."""
        config_file = tmp_path / "comparison.toml"
        config_file.write_text(
            f"""
[general]

[general.params]
rtol = 1e-6
atol = 1e-12
max_iterations = 100

[general.data]
matrix_path = "{tmp_path / "matrix.npy"}"
rhs_path = "{tmp_path / "rhs.npy"}"
normalize_system = "matrix"

[[preconditioners]]
name = "test_solver"
type = "jacobi"
"""
        )

        config = load_comparison_config(config_file)
        assert config.general.params.rtol == 1e-6
        assert config.general.params.atol == 1e-12
        assert len(config.preconditioners) == 1
        assert config.preconditioners[0].name == "test_solver"
        assert config.preconditioners[0].type == "jacobi"

    def test_legacy_schema_rejected(self, tmp_path: Path):
        """Legacy solver schema must fail."""
        config_file = tmp_path / "comparison.toml"
        config_file.write_text(
            """
[general]
rtol = 1e-6

[[solvers]]
name = "legacy_solver"
type = "none"
"""
        )

        with pytest.raises(ValidationError):
            load_comparison_config(config_file)

    def test_neural_model_ref_loads_without_embedded_model_store(self, tmp_path: Path) -> None:
        """Neural model_ref configs no longer embed MLflow topology."""
        config_file = tmp_path / "comparison.toml"
        config_file.write_text(
            f"""
[general]

[general.params]
rtol = 1e-6
atol = 1e-12
max_iterations = 100

[general.data]
matrix_path = "{tmp_path / "matrix.npy"}"
rhs_path = "{tmp_path / "rhs.npy"}"

[[preconditioners]]
name = "neural"
type = "neural"
model_ref = {{ source = "registered", name = "NormScaledLinearFFNN", alias = "solutions" }}
"""
        )
        config = load_comparison_config(config_file)
        assert config.preconditioners[0].name == "neural"

    def test_dataset_alias_required_for_at_dataset(self, tmp_path: Path) -> None:
        """@dataset requires general.data.dataset_alias."""
        config_file = tmp_path / "comparison.toml"
        config_file.write_text(
            f"""
[general]

[general.params]
rtol = 1e-6
atol = 1e-12
max_iterations = 100

[general.data]
matrix_path = "{tmp_path / "matrix.npy"}"
rhs_path = "{tmp_path / "rhs.npy"}"

[[preconditioners]]
name = "neural"
type = "neural"
model_ref = {{ source = "registered", name = "NormScaledLinearFFNN", alias = "@dataset" }}
"""
        )
        with pytest.raises(ValidationError, match="dataset_alias"):
            load_comparison_config(config_file)

    def test_project_comparison_configs_load(self) -> None:
        """Test-scoped comparison configs (linear/ffnn) load successfully."""
        comparison_dir = Path(__file__).resolve().parents[2] / "fixtures" / "configs" / "comparison"
        linear_cfg = comparison_dir / "linear.toml"
        ffnn_cfg = comparison_dir / "ffnn.toml"

        linear = load_comparison_config(linear_cfg)
        ffnn = load_comparison_config(ffnn_cfg)

        assert len(linear.preconditioners) > 0
        assert len(ffnn.preconditioners) > 0

    def test_comparison_profile_rejects_embedded_tracking(self, tmp_path: Path) -> None:
        """Comparison profiles must not define runtime MLflow topology."""
        config_file = tmp_path / "comparison.toml"
        config_file.write_text(
            f"""
[general]

[general.params]
rtol = 1e-6
atol = 1e-12
max_iterations = 100

[general.data]
matrix_path = "{tmp_path / "matrix.npy"}"
rhs_path = "{tmp_path / "rhs.npy"}"

[general.tracking]
tracking_uri = "sqlite:///{(tmp_path / "comparisons.db").as_posix()}"

[[preconditioners]]
name = "test_solver"
type = "jacobi"
"""
        )

        with pytest.raises(ValidationError, match="tracking"):
            load_comparison_config(config_file)

    def test_comparison_config_rejects_schema_marker(self, tmp_path: Path) -> None:
        """Comparison configs must not declare a schema marker."""
        config_file = tmp_path / "comparison.toml"
        config_file.write_text(
            f"""
schema_version = 3

[general]

[general.params]
rtol = 1e-6
atol = 1e-12
max_iterations = 100

[general.data]
matrix_path = "{tmp_path / "matrix.npy"}"
rhs_path = "{tmp_path / "rhs.npy"}"

[[preconditioners]]
name = "test_solver"
type = "jacobi"
"""
        )

        with pytest.raises(ValidationError, match="schema_version"):
            load_comparison_config(config_file)


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

        with pytest.raises(tomllib.TOMLDecodeError):
            load_raw_toml(config_file)
