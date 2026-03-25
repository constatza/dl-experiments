"""Tests for path resolution module (configuration/paths.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.configuration.data_models import DataConfigFile, OutputConfig
from neuralls.configuration.paths import (
    PathContext,
    build_path_context,
    resolve_output_root,
    resolve_processed_root,
    resolve_project_root,
)
from neuralls.constants import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROCESSED_DATA_DIR,
    DEFAULT_PROJECT_ROOT,
)


class TestResolveProjectRoot:
    """Tests for resolve_project_root function."""

    def test_default_project_root(self):
        """Test that default project root is returned when no override."""
        result = resolve_project_root()
        assert result == DEFAULT_PROJECT_ROOT
        assert result.is_absolute()

    def test_override_with_absolute_path(self, tmp_path: Path):
        """Test override with absolute path."""
        custom_root = tmp_path / "custom_project"
        custom_root.mkdir()

        result = resolve_project_root(custom_root)
        assert result == custom_root.resolve()
        assert result.is_absolute()

    def test_override_with_relative_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test override with relative path gets resolved to absolute."""
        monkeypatch.chdir(tmp_path)
        relative_path = "relative/project"
        (tmp_path / "relative" / "project").mkdir(parents=True)

        result = resolve_project_root(relative_path)
        assert result.is_absolute()
        assert result == (tmp_path / "relative" / "project").resolve()

    def test_override_with_string(self, tmp_path: Path):
        """Test override with string path."""
        custom_root = tmp_path / "string_project"
        custom_root.mkdir()

        result = resolve_project_root(str(custom_root))
        assert result == custom_root.resolve()
        assert isinstance(result, Path)


class TestResolveOutputRoot:
    """Tests for resolve_output_root function."""

    def test_default_output_root(self):
        """Test that default output root is returned when no override."""
        result = resolve_output_root()
        assert result == DEFAULT_OUTPUT_DIR
        assert result.is_absolute()

    def test_override_with_path(self, tmp_path: Path):
        """Test override with Path object."""
        custom_output = tmp_path / "custom_output"
        custom_output.mkdir()

        result = resolve_output_root(custom_output)
        assert result == custom_output.resolve()
        assert result.is_absolute()

    def test_override_with_string(self, tmp_path: Path):
        """Test override with string path."""
        custom_output = tmp_path / "output"
        custom_output.mkdir()

        result = resolve_output_root(str(custom_output))
        assert result == custom_output.resolve()


class TestResolveProcessedRoot:
    """Tests for resolve_processed_root function."""

    def test_default_processed_root(self, tmp_path: Path):
        """Test default processed root when not specified in config."""
        data_cfg = DataConfigFile(output=OutputConfig())

        result = resolve_processed_root(data_cfg, tmp_path)
        assert result == DEFAULT_PROCESSED_DATA_DIR

    def test_absolute_path_from_config(self, tmp_path: Path):
        """Test absolute path from data config."""
        absolute_processed = tmp_path / "absolute_processed"
        absolute_processed.mkdir()

        data_cfg = DataConfigFile(output=OutputConfig(data_dir=absolute_processed))

        result = resolve_processed_root(data_cfg, tmp_path)
        assert result == absolute_processed

    def test_relative_path_from_config(self, tmp_path: Path):
        """Test relative path from config gets resolved against project_root."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        (project_root / "relative" / "processed").mkdir(parents=True)

        data_cfg = DataConfigFile(output=OutputConfig(data_dir=Path("relative/processed")))

        result = resolve_processed_root(data_cfg, project_root)
        expected = (project_root / "relative" / "processed").resolve()
        assert result == expected
        assert result.is_absolute()


class TestPathContext:
    """Tests for PathContext dataclass."""

    def test_path_context_creation(self, tmp_path: Path):
        """Test creating a PathContext."""
        project_root = tmp_path / "project"
        output_root = tmp_path / "output"
        processed_root = tmp_path / "processed"

        ctx = PathContext(
            project_root=project_root,
            output_root=output_root,
            processed_root=processed_root,
        )

        assert ctx.project_root == project_root
        assert ctx.output_root == output_root
        assert ctx.processed_root == processed_root

    def test_path_context_frozen(self, tmp_path: Path):
        """Test that PathContext is immutable (frozen)."""
        ctx = PathContext(
            project_root=tmp_path,
            output_root=tmp_path,
            processed_root=tmp_path,
        )

        with pytest.raises(AttributeError):
            ctx.project_root = tmp_path / "new"  # type: ignore

    def test_mlflow_tracking_uri_property(self, tmp_path: Path):
        """Test MLflow tracking URI property derivation."""
        output_root = tmp_path / "output"
        ctx = PathContext(
            project_root=tmp_path,
            output_root=output_root,
            processed_root=tmp_path,
        )

        expected_db_path = output_root / "mlruns" / "mlflow.db"
        expected_uri = f"sqlite:///{expected_db_path.as_posix()}"

        assert ctx.mlflow_tracking_uri == expected_uri

    def test_mlflow_artifact_location_property(self, tmp_path: Path):
        """Test MLflow artifact location property derivation."""
        output_root = tmp_path / "output"
        ctx = PathContext(
            project_root=tmp_path,
            output_root=output_root,
            processed_root=tmp_path,
        )

        expected_artifacts = output_root / "mlartifacts"
        assert ctx.mlflow_artifact_location == str(expected_artifacts.as_posix())


class TestBuildPathContext:
    """Tests for build_path_context function."""

    def test_build_with_defaults(self, tmp_path: Path):
        """Test building path context with all defaults."""
        data_cfg = DataConfigFile(output=OutputConfig())

        ctx = build_path_context(data_cfg)

        assert ctx.project_root == DEFAULT_PROJECT_ROOT
        assert ctx.output_root == DEFAULT_OUTPUT_DIR
        assert ctx.processed_root == DEFAULT_PROCESSED_DATA_DIR

    def test_build_with_output_override(self, tmp_path: Path):
        """Test building with output_root override."""
        data_cfg = DataConfigFile(output=OutputConfig())
        custom_output = tmp_path / "custom_output"
        custom_output.mkdir()

        ctx = build_path_context(data_cfg, output_override=custom_output)

        assert ctx.output_root == custom_output.resolve()
        # Others should be defaults
        assert ctx.project_root == DEFAULT_PROJECT_ROOT
        assert ctx.processed_root == DEFAULT_PROCESSED_DATA_DIR

    def test_build_with_project_override(self, tmp_path: Path):
        """Test building with project_root override."""
        data_cfg = DataConfigFile(output=OutputConfig())
        custom_project = tmp_path / "custom_project"
        custom_project.mkdir()

        ctx = build_path_context(data_cfg, project_override=custom_project)

        assert ctx.project_root == custom_project.resolve()
        # Others should be defaults
        assert ctx.output_root == DEFAULT_OUTPUT_DIR
        assert ctx.processed_root == DEFAULT_PROCESSED_DATA_DIR

    def test_build_with_config_processed_dir(self, tmp_path: Path):
        """Test building with processed_dir from config."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        (project_root / "data" / "processed").mkdir(parents=True)

        data_cfg = DataConfigFile(output=OutputConfig(data_dir=Path("data/processed")))

        ctx = build_path_context(data_cfg, project_override=project_root)

        expected_processed = (project_root / "data" / "processed").resolve()
        assert ctx.processed_root == expected_processed

    def test_build_with_all_overrides(self, tmp_path: Path):
        """Test building with all parameters overridden."""
        project_root = tmp_path / "project"
        output_root = tmp_path / "output"
        processed_dir = tmp_path / "processed"

        for path in [project_root, output_root, processed_dir]:
            path.mkdir()

        data_cfg = DataConfigFile(output=OutputConfig(data_dir=processed_dir))

        ctx = build_path_context(
            data_cfg,
            output_override=output_root,
            project_override=project_root,
        )

        assert ctx.project_root == project_root.resolve()
        assert ctx.output_root == output_root.resolve()
        assert ctx.processed_root == processed_dir.resolve()

    def test_mlflow_paths_derived_from_output_root(self, tmp_path: Path):
        """Test that MLflow paths are correctly derived from output_root."""
        output_root = tmp_path / "experiments_output"
        output_root.mkdir()

        data_cfg = DataConfigFile(output=OutputConfig())

        ctx = build_path_context(data_cfg, output_override=output_root)

        # MLflow tracking DB should be under output_root
        expected_tracking = f"sqlite:///{(output_root / 'mlruns' / 'mlflow.db').as_posix()}"
        assert ctx.mlflow_tracking_uri == expected_tracking

        # MLflow artifacts should be under output_root
        expected_artifacts = str((output_root / "mlartifacts").as_posix())
        assert ctx.mlflow_artifact_location == expected_artifacts
