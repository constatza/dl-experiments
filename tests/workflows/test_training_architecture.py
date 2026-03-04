"""Tests for the refactored training workflow architecture.

Verifies the single-responsibility helpers extracted from train_model().
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from dlkit import GeneralSettings
from dlkit.tools.io.sparse import save_sparse_pack

from neuralls.workflows.training import (
    _configure_dataloader_runtime,
    _create_feature_configs,
    _parent_run_context,
    _validate_dataset_section,
    TrainingArrays,
    GRAPH_DATASET_NAME,
    FLEXIBLE_DATASET_NAME,
)


@pytest.fixture
def sample_arrays(tmp_path: Path) -> TrainingArrays:
    """Sample training artifact paths."""
    rhs = np.random.rand(10, 5).astype(np.float64)
    solutions = np.random.rand(10, 5).astype(np.float64)
    rhs_path = tmp_path / "rhs.npy"
    solutions_path = tmp_path / "solutions.npy"
    np.save(rhs_path, rhs)
    np.save(solutions_path, solutions)

    matrix = np.eye(5, dtype=np.float64)
    rows, cols = np.nonzero(matrix)
    indices_single = np.vstack((rows, cols)).astype(np.int64)
    values_single = matrix[rows, cols].astype(np.float64)
    n = rhs.shape[0]
    indices = np.tile(indices_single, (1, n))
    values = np.tile(values_single, n)
    nnz_ptr = np.arange(0, values_single.size * n + 1, values_single.size, dtype=np.int64)
    matrix_pack = tmp_path / "matrix_coo"
    save_sparse_pack(
        path=matrix_pack,
        indices=indices,
        values=values,
        nnz_ptr=nnz_ptr,
        size=(5, 5),
    )
    return TrainingArrays(
        rhs=rhs_path,
        solutions=solutions_path,
        matrix_pack=matrix_pack,
        sample_count=10,
    )


def test_validate_dataset_section_raises_on_none() -> None:
    """_validate_dataset_section raises ValueError if DATASET is None."""
    mock_settings = MagicMock(spec=GeneralSettings)
    mock_settings.DATASET = None
    
    with pytest.raises(ValueError, match=r"missing \[DATASET\] section"):
        _validate_dataset_section(mock_settings)


def test_validate_dataset_section_passes_if_present() -> None:
    """_validate_dataset_section passes if DATASET is present."""
    mock_settings = MagicMock(spec=GeneralSettings)
    mock_settings.DATASET = MagicMock()
    
    _validate_dataset_section(mock_settings)  # Should not raise


def test_create_feature_configs_graph_dataset(sample_arrays: TrainingArrays) -> None:
    """_create_feature_configs returns x and sparse matrix entries for GraphDataset."""
    features = _create_feature_configs(sample_arrays, GRAPH_DATASET_NAME)
    
    assert len(features) == 2
    names = {f.name for f in features}
    assert names == {"x", "matrix"}
    matrix_feature = next(f for f in features if f.name == "matrix")
    assert matrix_feature.model_input is False
    assert matrix_feature.path == sample_arrays.matrix_pack


def test_create_feature_configs_flexible_dataset(sample_arrays: TrainingArrays) -> None:
    """_create_feature_configs returns x and sparse matrix entries for FlexibleDataset."""
    features = _create_feature_configs(sample_arrays, FLEXIBLE_DATASET_NAME)
    
    assert len(features) == 2
    names = {f.name for f in features}
    assert names == {"x", "matrix"}
    matrix_feature = next(f for f in features if f.name == "matrix")
    assert matrix_feature.model_input is False
    assert matrix_feature.path == sample_arrays.matrix_pack


def test_create_feature_configs_default_to_flexible(sample_arrays: TrainingArrays) -> None:
    """_create_feature_configs defaults to x + sparse matrix behavior."""
    features = _create_feature_configs(sample_arrays, None)
    
    assert len(features) == 2
    names = {f.name for f in features}
    assert names == {"x", "matrix"}
    matrix_feature = next(f for f in features if f.name == "matrix")
    assert matrix_feature.path == sample_arrays.matrix_pack


def test_parent_run_context_manager() -> None:
    """_parent_run_context correctly sets and restores MLFLOW_PARENT_RUN_ID."""
    var_name = "MLFLOW_PARENT_RUN_ID"
    original_val = os.environ.get(var_name)
    
    try:
        test_id = "test-parent-id"
        with _parent_run_context(test_id):
            assert os.environ.get(var_name) == test_id
            
        assert os.environ.get(var_name) == original_val
        
        # Test with None (should do nothing)
        with _parent_run_context(None):
            assert os.environ.get(var_name) == original_val
            
    finally:
        if original_val is not None:
            os.environ[var_name] = original_val
        elif var_name in os.environ:
            del os.environ[var_name]


def test_configure_dataloader_runtime_forces_single_process() -> None:
    """_configure_dataloader_runtime sets safe single-process dataloader flags."""
    dataloader_cfg = MagicMock()
    dataloader_cfg.update_with.return_value = "updated_dataloader"
    datamodule_cfg = MagicMock()
    datamodule_cfg.dataloader = dataloader_cfg
    datamodule_cfg.update_with.return_value = "updated_datamodule"

    settings = MagicMock(spec=GeneralSettings)
    settings.DATAMODULE = datamodule_cfg
    settings.update_with.return_value = "updated_settings"

    updated = _configure_dataloader_runtime(settings)

    dataloader_cfg.update_with.assert_called_once_with(
        {
            "num_workers": 0,
            "persistent_workers": False,
            "pin_memory": False,
        }
    )
    datamodule_cfg.update_with.assert_called_once_with(
        {"dataloader": "updated_dataloader"}
    )
    settings.update_with.assert_called_once_with({"DATAMODULE": "updated_datamodule"})
    assert updated == "updated_settings"


@patch("neuralls.workflows.training.compute_diagnostics")
@patch("neuralls.workflows.training._write_diagnostics_figure")
@patch("neuralls.workflows.training._log_diagnostics_to_mlflow")
def test_log_training_evaluation_orchestration(
    mock_mlflow_log: MagicMock,
    mock_write: MagicMock,
    mock_compute: MagicMock,
) -> None:
    """_log_training_evaluation delegates to diagnostics and figure helpers."""
    from neuralls.workflows.training import _log_training_evaluation
    
    mock_result = MagicMock()
    mock_result.to_numpy.return_value = {
        "predictions": {"output": np.zeros((10, 1))},
        "targets": {"y": np.zeros((10, 1))},
    }
    
    mock_compute.return_value = MagicMock()
    mock_write.return_value = Path("dummy.png")
    
    _log_training_evaluation(
        tracking_uri="sqlite:///test.db",
        run_id="run123",
        exp_id="exp456",
        artifacts_destination="/tmp/art",
        training_result=mock_result,
        figures_dir=Path("/tmp/figs"),
    )
    
    mock_compute.assert_called_once()
    mock_write.assert_called_once()
    mock_mlflow_log.assert_called_once()
