"""Tests for dataset_builder module.

Tests cover dataset construction from arrays and injection into
GeneralSettings objects.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pytest
from dlkit.infrastructure.config import GeneralSettings
from dlkit.infrastructure.config import (
    DatasetSettings,
    ModelComponentSettings as ModelSettings,
    SessionSettings,
    TrainingSettings,
)
from dlkit.infrastructure.config.core.patching import patch_model
from dlkit.infrastructure.config.trainer_settings import TrainerSettings

from neuralls.platform.config.models.dataset import (
    create_features_from_array,
    create_matrix_feature,
    create_targets_from_array,
    with_dataset_arrays,
)
from neuralls.composition.experiments.runtime_dataset_contract import (
    default_training_dataset_contract,
)

# Skip all tests if dlkit has circular import issue
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("dlkit") is None, reason="dlkit circular import issue"
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
            trainer=TrainerSettings(max_epochs=1, default_root_dir=trainer_root)
        ),
    )


class TestCreateFeaturesFromArray:
    """Tests for create_features_from_array function."""

    def test_create_features_default_name(self, sample_rhs_array: np.ndarray) -> None:
        contract = default_training_dataset_contract()
        features = create_features_from_array(sample_rhs_array, name=contract.primary_input_name)
        assert isinstance(features, tuple)
        assert len(features) == 1
        assert features[0].name == "x"

    def test_create_features_custom_name(self, sample_rhs_array: np.ndarray) -> None:
        features = create_features_from_array(sample_rhs_array, name="custom_feature")
        assert len(features) == 1
        assert features[0].name == "custom_feature"

    def test_create_features_model_input_is_true(self, sample_rhs_array: np.ndarray) -> None:
        features = create_features_from_array(sample_rhs_array, name="x")
        assert features[0].model_input is True


class TestCreateMatrixFeature:
    """Tests for create_matrix_feature function."""

    def test_create_matrix_feature(self, sample_matrix_array: np.ndarray) -> None:
        feature = create_matrix_feature(sample_matrix_array)
        assert feature.name == "matrix"
        assert feature is not None

    def test_create_matrix_feature_custom_name(self, sample_matrix_array: np.ndarray) -> None:
        feature = create_matrix_feature(sample_matrix_array, name="stiffness")
        assert feature.name == "stiffness"

    def test_create_matrix_feature_excludes_model_input(
        self, sample_matrix_array: np.ndarray
    ) -> None:
        feature = create_matrix_feature(sample_matrix_array)
        assert feature.model_input is False

    def test_create_matrix_feature_custom_name_excludes_model_input(
        self, sample_matrix_array: np.ndarray
    ) -> None:
        feature = create_matrix_feature(sample_matrix_array, name="stiffness")
        assert feature.model_input is False


class TestCreateTargetsFromArray:
    """Tests for create_targets_from_array function."""

    def test_create_targets_default_name(self, sample_solutions_array: np.ndarray) -> None:
        contract = default_training_dataset_contract()
        targets = create_targets_from_array(sample_solutions_array, name=contract.target_name)
        assert isinstance(targets, tuple)
        assert len(targets) == 1
        assert targets[0].name == "y"

    def test_create_targets_write_defaults_to_false(
        self, sample_solutions_array: np.ndarray
    ) -> None:
        targets = create_targets_from_array(sample_solutions_array, name="y")
        assert targets[0].write is False


class TestWithDatasetArrays:
    """Tests for with_dataset_arrays function."""

    def test_with_dataset_arrays_standard_dataset(
        self,
        mock_settings: GeneralSettings,
        sample_rhs_array: np.ndarray,
        sample_solutions_array: np.ndarray,
    ) -> None:
        """Test dataset injection for standard dataset."""
        contract = default_training_dataset_contract()
        updated = with_dataset_arrays(
            mock_settings,
            sample_rhs_array,
            sample_solutions_array,
            primary_input_name=contract.primary_input_name,
            target_name=contract.target_name,
            matrix_input_name=contract.matrix_input_name,
        )
        assert updated.DATASET is not None
        assert hasattr(updated.DATASET, "features")
        assert hasattr(updated.DATASET, "targets")
        assert updated.DATASET.features[0].name == "x"
        assert updated.DATASET.targets[0].name == "y"

    def test_with_dataset_arrays_graph_dataset_includes_matrix(
        self,
        mock_settings: GeneralSettings,
        sample_rhs_array: np.ndarray,
        sample_solutions_array: np.ndarray,
        sample_matrix_array: np.ndarray,
    ) -> None:
        """Test dataset injection includes matrix for GraphDataset."""
        graph_settings = patch_model(mock_settings, {"DATASET": {"name": "GraphDataset"}})
        contract = default_training_dataset_contract()
        updated = with_dataset_arrays(
            graph_settings,
            sample_rhs_array,
            sample_solutions_array,
            sample_matrix_array,
            primary_input_name=contract.primary_input_name,
            target_name=contract.target_name,
            matrix_input_name=contract.matrix_input_name,
        )
        assert any(f.name == "matrix" for f in updated.DATASET.features)

    def test_with_dataset_arrays_keeps_original_settings_immutable(
        self,
        mock_settings: GeneralSettings,
        sample_rhs_array: np.ndarray,
        sample_solutions_array: np.ndarray,
    ) -> None:
        """Dataset injection returns a new settings object."""
        contract = default_training_dataset_contract()
        updated = with_dataset_arrays(
            mock_settings,
            sample_rhs_array,
            sample_solutions_array,
            primary_input_name=contract.primary_input_name,
            target_name=contract.target_name,
            matrix_input_name=contract.matrix_input_name,
        )

        assert updated is not mock_settings
        assert mock_settings.DATASET is not None
        assert mock_settings.DATASET.features == ()
        assert mock_settings.DATASET.targets == ()
        assert updated.DATASET is not None
        assert updated.DATASET.features is not None
        assert updated.DATASET.targets is not None
