"""Tests for dataset_builder module.

Tests cover dataset construction from arrays and injection into
GeneralSettings objects.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pytest

# Skip all tests if dlkit has circular import issue
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("dlkit") is None,
    reason="dlkit circular import issue"
)

from dlkit import GeneralSettings
from dlkit.tools.config import (
    TrainingSettings,
    ModelComponentSettings as ModelSettings,
    DatasetSettings,
    SessionSettings,
)
from dlkit.tools.config.trainer_settings import TrainerSettings

from src.configuration.dataset import (
    create_features_from_array,
    create_matrix_feature,
    create_targets_from_array,
    with_dataset_arrays,
)


@pytest.fixture
def sample_rhs_array() -> np.ndarray:
    """Sample RHS array for testing."""
    return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)


@pytest.fixture
def sample_solutions_array() -> np.ndarray:
    """Sample solutions array for testing."""
    return np.array([[0.5, 0.3], [0.2, 0.8]], dtype=np.float64)


@pytest.fixture
def sample_matrix_array() -> np.ndarray:
    """Sample matrix array for testing."""
    return np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)


@pytest.fixture
def mock_settings(tmp_path: Path) -> GeneralSettings:
    """Create a mock GeneralSettings object for testing."""
    trainer_root = tmp_path / "trainer_root"
    trainer_root.mkdir()
    
    return GeneralSettings(
        SESSION=SessionSettings(seed=42),
        MODEL=ModelSettings(name="LinearModel"),
        DATASET=DatasetSettings(name="FlexibleDataset"),
        TRAINING=TrainingSettings(
            trainer=TrainerSettings(max_epochs=1, default_root_dir=str(trainer_root))
        ),
    )


class TestCreateFeaturesFromArray:
    """Tests for create_features_from_array function."""

    def test_create_features_default_name(self, sample_rhs_array: np.ndarray) -> None:
        features = create_features_from_array(sample_rhs_array)
        assert isinstance(features, tuple)
        assert len(features) == 1
        assert features[0].name == "rhs"

    def test_create_features_custom_name(self, sample_rhs_array: np.ndarray) -> None:
        features = create_features_from_array(sample_rhs_array, name="custom_feature")
        assert len(features) == 1
        assert features[0].name == "custom_feature"

class TestCreateMatrixFeature:
    """Tests for create_matrix_feature function."""

    def test_create_matrix_feature(self, sample_matrix_array: np.ndarray) -> None:
        feature = create_matrix_feature(sample_matrix_array)
        assert feature.name == "matrix"
        assert feature is not None

class TestCreateTargetsFromArray:
    """Tests for create_targets_from_array function."""

    def test_create_targets_default_name(
        self, sample_solutions_array: np.ndarray
    ) -> None:
        targets = create_targets_from_array(sample_solutions_array)
        assert isinstance(targets, tuple)
        assert len(targets) == 1
        assert targets[0].name == "solutions"

class TestWithDatasetArrays:
    """Tests for with_dataset_arrays function."""

    def test_with_dataset_arrays_standard_dataset(
        self,
        mock_settings: GeneralSettings,
        sample_rhs_array: np.ndarray,
        sample_solutions_array: np.ndarray,
    ) -> None:
        """Test dataset injection for standard dataset."""
        updated = with_dataset_arrays(
            mock_settings, sample_rhs_array, sample_solutions_array
        )
        assert updated.DATASET is not None
        assert hasattr(updated.DATASET, "features")
        assert hasattr(updated.DATASET, "targets")

    def test_with_dataset_arrays_graph_dataset_includes_matrix(
        self,
        mock_settings: GeneralSettings,
        sample_rhs_array: np.ndarray,
        sample_solutions_array: np.ndarray,
        sample_matrix_array: np.ndarray,
    ) -> None:
        """Test dataset injection includes matrix for GraphDataset."""
        mock_settings.DATASET.name = "GraphDataset"
        updated = with_dataset_arrays(
            mock_settings,
            sample_rhs_array,
            sample_solutions_array,
            sample_matrix_array,
        )
        assert any(f.name == "matrix" for f in updated.DATASET.features)