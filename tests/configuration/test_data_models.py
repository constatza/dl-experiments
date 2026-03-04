"""Tests for Pydantic data configuration models.

Tests cover validation of data config TOML structure using Pydantic models.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neuralls.configuration.data_models import (
    DataConfigFile,
    FlowConfig,
    GenerationConfig,
    OutputConfig,
    SourceConfig,
    StrategyConfig,
    DataTestConfig,
)


class TestSourceConfig:
    """Tests for SourceConfig validation."""

    def test_source_config_defaults(self):
        """Test SourceConfig with defaults."""
        config = SourceConfig()
        assert config.type is None
        assert config.matrix_path is None
        assert config.solutions_path is None

    def test_source_config_with_values(self):
        """Test SourceConfig with custom values."""
        config = SourceConfig(
            type="solution_archive",
            matrix_path="/path/to/matrix.txt",
            solutions_path="/path/to/solutions/*.txt",
        )
        assert config.type == "solution_archive"
        assert config.matrix_path == "/path/to/matrix.txt"
        assert config.solutions_path == "/path/to/solutions/*.txt"


class TestStrategyConfig:
    """Tests for StrategyConfig validation."""

    def test_strategy_config_required_fields(self):
        """Test StrategyConfig with required name field."""
        config = StrategyConfig(name="cg_residual")
        assert config.name == "cg_residual"
        assert config.samples == 0

    def test_strategy_config_with_all_fields(self):
        """Test StrategyConfig with all fields."""
        config = StrategyConfig(
            name="cg_residual_error",
            samples=10000,
            residual_iters=50,
            solutions_glob="/path/to/solutions/*.txt",
        )
        assert config.name == "cg_residual_error"
        assert config.samples == 10000
        assert config.residual_iters == 50
        assert config.solutions_glob == "/path/to/solutions/*.txt"

    def test_strategy_config_validation(self):
        """Test StrategyConfig field validation."""
        # krylov_iters must be >= 1
        with pytest.raises(ValidationError):
            StrategyConfig(name="test", krylov_iters=0)

        # residual_iters must be >= 1
        with pytest.raises(ValidationError):
            StrategyConfig(name="test", residual_iters=-1)


class TestGenerationConfig:
    """Tests for GenerationConfig validation."""

    def test_generation_config_defaults(self):
        """Test GenerationConfig with defaults."""
        config = GenerationConfig()
        assert config.normalize == "matrix"
        assert config.shuffle is True
        assert config.seed == 42
        assert config.strategy == []

    def test_generation_config_with_strategies(self):
        """Test GenerationConfig with strategy list."""
        config = GenerationConfig(
            normalize="rhs",
            shuffle=False,
            seed=123,
            strategy=[
                StrategyConfig(name="strategy1", samples=1000),
                StrategyConfig(name="strategy2", samples=2000),
            ],
        )
        assert config.normalize == "rhs"
        assert config.shuffle is False
        assert config.seed == 123
        assert len(config.strategy) == 2
        assert config.strategy[0].name == "strategy1"
        assert config.strategy[1].samples == 2000


class TestDataTestConfig:
    """Tests for DataTestConfig validation."""

    def test_test_config_defaults(self):
        """Test DataTestConfig with defaults."""
        config = DataTestConfig()
        assert config.solutions_path is None
        assert config.rhs_path is None

    def test_test_config_with_values(self):
        """Test DataTestConfig with custom values."""
        config = DataTestConfig(
            solutions_path="/path/to/test/solutions.txt",
            rhs_path="/path/to/test/rhs.txt",
        )
        assert config.solutions_path == "/path/to/test/solutions.txt"
        assert config.rhs_path == "/path/to/test/rhs.txt"


class TestDataConfigFile:
    """Tests for complete DataConfigFile validation."""

    def test_data_config_file_minimal(self):
        """Test DataConfigFile with minimal fields (all defaults)."""
        config = DataConfigFile()
        assert config.flow.id is None
        assert config.source.matrix_path is None
        assert config.generation.normalize == "matrix"
        assert config.output.data_dir is None
        assert config.test.solutions_path is None

    def test_data_config_file_full(self):
        """Test DataConfigFile with all sections."""
        config = DataConfigFile(
            flow=FlowConfig(id="test_flow", dataset="test_dataset"),
            source=SourceConfig(
                type="solution_archive",
                matrix_path="/path/to/matrix.txt",
                solutions_path="/path/to/solutions/*.txt",
            ),
            generation=GenerationConfig(
                normalize="matrix",
                shuffle=True,
                seed=42,
                strategy=[
                    StrategyConfig(
                        name="solution_archive", samples=-1, solutions_glob="*.txt"
                    )
                ],
            ),
            output=OutputConfig(data_dir="/data/processed"),
            test=DataTestConfig(solutions_path="/data/test/solutions.txt"),
        )
        assert config.flow.id == "test_flow"
        assert config.source.type == "solution_archive"
        assert config.generation.strategy[0].name == "solution_archive"
        assert config.output.data_dir == Path("/data/processed")
        assert config.test.solutions_path == "/data/test/solutions.txt"

    def test_data_config_file_immutable(self):
        """Test that DataConfigFile is frozen."""
        config = DataConfigFile()
        with pytest.raises((ValidationError, AttributeError)):
            config.generation.seed = 999  # type: ignore

    def test_data_config_file_allows_extra_fields(self):
        """Test DataConfigFile allows extra fields for extensibility."""
        config = DataConfigFile(custom_field="custom_value")  # type: ignore
        # Should not raise ValidationError
        assert config.flow.id is None  # Defaults still work
