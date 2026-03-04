"""Tests for TOML loader module.

Tests cover TOML parsing and Pydantic validation of configuration files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.io.toml_loader import (
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
data_dir = "/data/processed"

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
        assert config.output.data_dir == Path("/data/processed")
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
schema_version = 3

[general]

[general.params]
rtol = 1e-6
atol = 1e-12
max_iterations = 100

[general.data]
matrix_path = "{tmp_path / "matrix.npy"}"
rhs_path = "{tmp_path / "rhs.npy"}"
normalize_system = "matrix"

[general.tracking]
tracking_uri = "sqlite:///{(tmp_path / "comparisons.db").as_posix()}"
artifact_location = "{(tmp_path / "mlartifacts").as_posix()}"

[general.model_store]
tracking_uri = "sqlite:///{(tmp_path / "models.db").as_posix()}"

[[preconditioners]]
name = "test_solver"
type = "jacobi"
"""
        )

        config = load_comparison_config(config_file)
        assert config.schema_version == 3
        assert config.general.params.rtol == 1e-6
        assert config.general.params.atol == 1e-12
        assert len(config.preconditioners) == 1
        assert config.preconditioners[0].name == "test_solver"
        assert config.preconditioners[0].type == "jacobi"

    def test_legacy_schema_rejected(self, tmp_path: Path):
        """Legacy solver schema must fail."""
        config_file = tmp_path / "comparison.toml"
        config_file.write_text(
            f"""
[general]
rtol = 1e-6

[[solvers]]
name = "legacy_solver"
type = "none"
"""
        )

        with pytest.raises(ValidationError):
            load_comparison_config(config_file)

    def test_neural_model_ref_requires_model_store(self, tmp_path: Path) -> None:
        """Neural model_ref configs must define general.model_store."""
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

[general.tracking]
tracking_uri = "sqlite:///{(tmp_path / "comparisons.db").as_posix()}"
artifact_location = "{(tmp_path / "mlartifacts").as_posix()}"

[[preconditioners]]
name = "neural"
type = "neural"
model_ref = {{ source = "registered", name = "NormScaledLinearFFNN", alias = "solutions" }}
"""
        )
        with pytest.raises(ValidationError, match="general.model_store"):
            load_comparison_config(config_file)

    def test_dataset_alias_required_for_at_dataset(self, tmp_path: Path) -> None:
        """@dataset requires general.data.dataset_alias."""
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

[general.tracking]
tracking_uri = "sqlite:///{(tmp_path / "comparisons.db").as_posix()}"
artifact_location = "{(tmp_path / "mlartifacts").as_posix()}"

[general.model_store]
tracking_uri = "sqlite:///{(tmp_path / "models.db").as_posix()}"

[[preconditioners]]
name = "neural"
type = "neural"
model_ref = {{ source = "registered", name = "NormScaledLinearFFNN", alias = "@dataset" }}
"""
        )
        with pytest.raises(ValidationError, match="dataset_alias"):
            load_comparison_config(config_file)

    def test_project_comparison_configs_load(self) -> None:
        """Project comparison configs (linear/ffnn) load successfully."""
        project_root = Path(__file__).resolve().parents[3]
        comparison_dir = project_root / "configs" / "comparison"
        linear_cfg = comparison_dir / "linear.toml"
        ffnn_cfg = comparison_dir / "ffnn.toml"

        linear = load_comparison_config(linear_cfg)
        ffnn = load_comparison_config(ffnn_cfg)

        assert linear.schema_version == 3
        assert ffnn.schema_version == 3
        assert len(linear.preconditioners) > 0
        assert len(ffnn.preconditioners) > 0


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
