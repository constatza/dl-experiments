"""Tests for dataset_builder module.

Tests cover dataset construction from arrays and injection into
GeneralSettings objects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.configuration.dataset import (
    create_features_from_array,
    create_matrix_feature,
    create_targets_from_array,
    with_dataset_arrays,
)


# Get project root (repo root directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.fixture
def linear_config_path() -> Path:
    """Path to linear model config file.

    Returns:
        Path to graph-cg/configs/linear.toml.
    """
    return PROJECT_ROOT / "configs" / "linear.toml"


@pytest.fixture
def sample_rhs_array() -> np.ndarray:
    """Sample RHS array for testing.

    Returns:
        1D numpy array with RHS values.
    """
    return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)


@pytest.fixture
def sample_solutions_array() -> np.ndarray:
    """Sample solutions array for testing.

    Returns:
        1D numpy array with solution values.
    """
    return np.array([[0.5, 0.3], [0.2, 0.8]], dtype=np.float64)


@pytest.fixture
def sample_matrix_array() -> np.ndarray:
    """Sample matrix array for testing.

    Returns:
        2D numpy array with matrix data.
    """
    return np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)


class TestCreateFeaturesFromArray:
    """Tests for create_features_from_array function."""

    def test_create_features_default_name(
        self, sample_rhs_array: np.ndarray
    ) -> None:
        """Test feature creation with default name.

        Args:
            sample_rhs_array: Sample RHS array.
        """
        features = create_features_from_array(sample_rhs_array)

        assert isinstance(features, tuple)
        assert len(features) == 1
        assert features[0].name == "rhs"

    def test_create_features_custom_name(
        self, sample_rhs_array: np.ndarray
    ) -> None:
        """Test feature creation with custom name.

        Args:
            sample_rhs_array: Sample RHS array.
        """
        features = create_features_from_array(sample_rhs_array, name="custom_feature")

        assert len(features) == 1
        assert features[0].name == "custom_feature"

    def test_create_features_preserves_array_data(
        self, sample_rhs_array: np.ndarray
    ) -> None:
        """Test that array data is preserved.

        Args:
            sample_rhs_array: Sample RHS array.
        """
        features = create_features_from_array(sample_rhs_array)

        np.testing.assert_array_equal(features[0].value, sample_rhs_array)

    def test_create_features_returns_tuple(
        self, sample_rhs_array: np.ndarray
    ) -> None:
        """Test that features are returned as tuple.

        Args:
            sample_rhs_array: Sample RHS array.
        """
        features = create_features_from_array(sample_rhs_array)

        assert isinstance(features, tuple)
        assert all(hasattr(f, "name") and hasattr(f, "value") for f in features)


class TestCreateMatrixFeature:
    """Tests for create_matrix_feature function."""

    def test_create_matrix_feature(self, sample_matrix_array: np.ndarray) -> None:
        """Test matrix feature creation.

        Args:
            sample_matrix_array: Sample matrix array.
        """
        feature = create_matrix_feature(sample_matrix_array)

        assert feature.name == "matrix"
        assert feature is not None

    def test_create_matrix_feature_data_preserved(
        self, sample_matrix_array: np.ndarray
    ) -> None:
        """Test that matrix data is preserved.

        Args:
            sample_matrix_array: Sample matrix array.
        """
        feature = create_matrix_feature(sample_matrix_array)

        np.testing.assert_array_equal(feature.value, sample_matrix_array)

    def test_create_matrix_feature_2d_array(self) -> None:
        """Test matrix feature with 2D array."""
        matrix = np.eye(3, dtype=np.float64)
        feature = create_matrix_feature(matrix)

        np.testing.assert_array_equal(feature.value, matrix)

    def test_create_matrix_feature_sparse_like_array(self) -> None:
        """Test matrix feature with sparse-like array."""
        matrix = np.zeros((5, 5), dtype=np.float64)
        matrix[0, 0] = 1.0
        matrix[1, 1] = 2.0

        feature = create_matrix_feature(matrix)

        np.testing.assert_array_equal(feature.value, matrix)


class TestCreateTargetsFromArray:
    """Tests for create_targets_from_array function."""

    def test_create_targets_default_name(
        self, sample_solutions_array: np.ndarray
    ) -> None:
        """Test target creation with default name.

        Args:
            sample_solutions_array: Sample solutions array.
        """
        targets = create_targets_from_array(sample_solutions_array)

        assert isinstance(targets, tuple)
        assert len(targets) == 1
        assert targets[0].name == "solutions"

    def test_create_targets_custom_name(
        self, sample_solutions_array: np.ndarray
    ) -> None:
        """Test target creation with custom name.

        Args:
            sample_solutions_array: Sample solutions array.
        """
        targets = create_targets_from_array(sample_solutions_array, name="custom_target")

        assert len(targets) == 1
        assert targets[0].name == "custom_target"

    def test_create_targets_preserves_array_data(
        self, sample_solutions_array: np.ndarray
    ) -> None:
        """Test that array data is preserved.

        Args:
            sample_solutions_array: Sample solutions array.
        """
        targets = create_targets_from_array(sample_solutions_array)

        np.testing.assert_array_equal(targets[0].value, sample_solutions_array)

    def test_create_targets_returns_tuple(
        self, sample_solutions_array: np.ndarray
    ) -> None:
        """Test that targets are returned as tuple.

        Args:
            sample_solutions_array: Sample solutions array.
        """
        targets = create_targets_from_array(sample_solutions_array)

        assert isinstance(targets, tuple)
        assert all(hasattr(t, "name") and hasattr(t, "value") for t in targets)


class TestWithDatasetArrays:
    """Tests for with_dataset_arrays function."""

    def test_with_dataset_arrays_standard_dataset(
        self,
        linear_config_path: Path,
        sample_rhs_array: np.ndarray,
        sample_solutions_array: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test dataset injection for standard dataset.

        Args:
            linear_config_path: Path to linear config.
            sample_rhs_array: Sample RHS array.
            sample_solutions_array: Sample solutions array.
            monkeypatch: Pytest monkeypatch fixture.
        """
        from dlkit.tools.config import load_training_settings

        # Mock mkdir
        original_mkdir = Path.mkdir

        def _safe_mkdir(
            self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
        ) -> None:
            if str(self).startswith("/data/"):
                return None
            original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

        monkeypatch.setattr(Path, "mkdir", _safe_mkdir, raising=False)

        # Load settings
        settings = load_training_settings(str(linear_config_path))

        # Inject dataset
        updated = with_dataset_arrays(
            settings, sample_rhs_array, sample_solutions_array
        )

        # Verify DATASET was injected
        assert updated.DATASET is not None
        assert hasattr(updated.DATASET, "features")
        assert hasattr(updated.DATASET, "targets")

    def test_with_dataset_arrays_includes_rhs_feature(
        self,
        linear_config_path: Path,
        sample_rhs_array: np.ndarray,
        sample_solutions_array: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that RHS is included in features.

        Args:
            linear_config_path: Path to linear config.
            sample_rhs_array: Sample RHS array.
            sample_solutions_array: Sample solutions array.
            monkeypatch: Pytest monkeypatch fixture.
        """
        from dlkit.tools.config import load_training_settings

        # Mock mkdir
        original_mkdir = Path.mkdir

        def _safe_mkdir(
            self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
        ) -> None:
            if str(self).startswith("/data/"):
                return None
            original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

        monkeypatch.setattr(Path, "mkdir", _safe_mkdir, raising=False)

        # Load settings
        settings = load_training_settings(str(linear_config_path))

        # Inject dataset
        updated = with_dataset_arrays(
            settings, sample_rhs_array, sample_solutions_array
        )

        # Check features
        features = updated.DATASET.features
        assert len(features) >= 1
        assert any(f.name == "rhs" for f in features)

    def test_with_dataset_arrays_includes_targets(
        self,
        linear_config_path: Path,
        sample_rhs_array: np.ndarray,
        sample_solutions_array: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that targets are included.

        Args:
            linear_config_path: Path to linear config.
            sample_rhs_array: Sample RHS array.
            sample_solutions_array: Sample solutions array.
            monkeypatch: Pytest monkeypatch fixture.
        """
        from dlkit.tools.config import load_training_settings

        # Mock mkdir
        original_mkdir = Path.mkdir

        def _safe_mkdir(
            self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
        ) -> None:
            if str(self).startswith("/data/"):
                return None
            original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

        monkeypatch.setattr(Path, "mkdir", _safe_mkdir, raising=False)

        # Load settings
        settings = load_training_settings(str(linear_config_path))

        # Inject dataset
        updated = with_dataset_arrays(
            settings, sample_rhs_array, sample_solutions_array
        )

        # Check targets
        targets = updated.DATASET.targets
        assert len(targets) >= 1
        assert any(t.name == "solutions" for t in targets)

    def test_with_dataset_arrays_graph_dataset_includes_matrix(
        self,
        linear_config_path: Path,
        sample_rhs_array: np.ndarray,
        sample_solutions_array: np.ndarray,
        sample_matrix_array: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test dataset injection includes matrix for GraphDataset.

        Args:
            linear_config_path: Path to linear config.
            sample_rhs_array: Sample RHS array.
            sample_solutions_array: Sample solutions array.
            sample_matrix_array: Sample matrix array.
            monkeypatch: Pytest monkeypatch fixture.
        """
        from dlkit.tools.config import load_training_settings

        # Mock mkdir
        original_mkdir = Path.mkdir

        def _safe_mkdir(
            self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
        ) -> None:
            if str(self).startswith("/data/"):
                return None
            original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

        monkeypatch.setattr(Path, "mkdir", _safe_mkdir, raising=False)

        # Load settings and patch DATASET.name to "GraphDataset"
        settings = load_training_settings(str(linear_config_path))

        # Create a copy with GraphDataset
        graph_settings = settings.model_copy(
            update={"DATASET": settings.DATASET.model_copy(update={"name": "GraphDataset"})}
        )

        # Inject dataset with matrix
        updated = with_dataset_arrays(
            graph_settings, sample_rhs_array, sample_solutions_array, sample_matrix_array
        )

        # For GraphDataset, matrix should be in features
        features = updated.DATASET.features
        assert any(f.name == "matrix" for f in features)

    def test_with_dataset_arrays_non_graph_excludes_matrix_if_provided(
        self,
        linear_config_path: Path,
        sample_rhs_array: np.ndarray,
        sample_solutions_array: np.ndarray,
        sample_matrix_array: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that matrix is excluded for non-GraphDataset even if provided.

        Args:
            linear_config_path: Path to linear config.
            sample_rhs_array: Sample RHS array.
            sample_solutions_array: Sample solutions array.
            sample_matrix_array: Sample matrix array.
            monkeypatch: Pytest monkeypatch fixture.
        """
        from dlkit.tools.config import load_training_settings

        # Mock mkdir
        original_mkdir = Path.mkdir

        def _safe_mkdir(
            self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
        ) -> None:
            if str(self).startswith("/data/"):
                return None
            original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

        monkeypatch.setattr(Path, "mkdir", _safe_mkdir, raising=False)

        # Load settings (FlexibleDataset, not GraphDataset)
        settings = load_training_settings(str(linear_config_path))

        # Inject dataset with matrix
        updated = with_dataset_arrays(
            settings, sample_rhs_array, sample_solutions_array, sample_matrix_array
        )

        # Matrix should NOT be in features for non-GraphDataset
        features = updated.DATASET.features
        assert not any(f.name == "matrix" for f in features)

    def test_with_dataset_arrays_preserves_other_settings(
        self,
        linear_config_path: Path,
        sample_rhs_array: np.ndarray,
        sample_solutions_array: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that update_settings doesn't overwrite unrelated fields.

        Args:
            linear_config_path: Path to linear config.
            sample_rhs_array: Sample RHS array.
            sample_solutions_array: Sample solutions array.
            monkeypatch: Pytest monkeypatch fixture.
        """
        from dlkit.tools.config import load_training_settings

        # Mock mkdir
        original_mkdir = Path.mkdir

        def _safe_mkdir(
            self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
        ) -> None:
            if str(self).startswith("/data/"):
                return None
            original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

        monkeypatch.setattr(Path, "mkdir", _safe_mkdir, raising=False)

        # Load settings
        settings = load_training_settings(str(linear_config_path))

        # Store original values
        original_model = settings.MODEL
        original_training = settings.TRAINING

        # Inject dataset
        updated = with_dataset_arrays(
            settings, sample_rhs_array, sample_solutions_array
        )

        # Verify other sections are unchanged
        assert updated.MODEL == original_model
        assert updated.TRAINING == original_training

    def test_with_dataset_arrays_without_matrix(
        self,
        linear_config_path: Path,
        sample_rhs_array: np.ndarray,
        sample_solutions_array: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test injection without matrix (matrix=None).

        Args:
            linear_config_path: Path to linear config.
            sample_rhs_array: Sample RHS array.
            sample_solutions_array: Sample solutions array.
            monkeypatch: Pytest monkeypatch fixture.
        """
        from dlkit.tools.config import load_training_settings

        # Mock mkdir
        original_mkdir = Path.mkdir

        def _safe_mkdir(
            self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
        ) -> None:
            if str(self).startswith("/data/"):
                return None
            original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

        monkeypatch.setattr(Path, "mkdir", _safe_mkdir, raising=False)

        # Load settings
        settings = load_training_settings(str(linear_config_path))

        # Inject without matrix
        updated = with_dataset_arrays(
            settings, sample_rhs_array, sample_solutions_array, matrix=None
        )

        # Should only have rhs in features
        features = updated.DATASET.features
        feature_names = [f.name for f in features]
        assert "rhs" in feature_names
        assert "matrix" not in feature_names
