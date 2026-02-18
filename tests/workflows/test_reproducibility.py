"""Unit tests for reproducibility utilities.

Covers:
- _create_snapshot_config(): updates output_dir and checkpoint_paths
- capture_batch_context(): logs snapshot to MLflow (no custom dirs)
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import pytest
import tomli_w

from neuralls.workflows.comparison_run import ComparisonRun
from neuralls.workflows.multi_training import BatchResult, TrainingRunResult
from neuralls.workflows.utils.reproducibility import (
    _create_snapshot_config,
    capture_batch_context,
)


class ExperimentPaths(NamedTuple):
    """Holds paths for test experiments."""

    exp1_ckpt: Path
    exp2_ckpt: Path
    base_output: Path
    batch_output_dir: Path


@pytest.fixture
def experiment_paths(tmp_path: Path) -> ExperimentPaths:
    """Create directory structure and return paths.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        ExperimentPaths with checkpoint and output paths.
    """
    exp1_dir = tmp_path / "training" / "exp1_run"
    exp1_ckpt = exp1_dir / "checkpoints" / "model.ckpt"

    exp2_dir = tmp_path / "training" / "exp2_run"
    exp2_ckpt = exp2_dir / "checkpoints" / "model.ckpt"

    base_output = tmp_path
    batch_output_dir = base_output / "training"

    return ExperimentPaths(
        exp1_ckpt=exp1_ckpt,
        exp2_ckpt=exp2_ckpt,
        base_output=base_output,
        batch_output_dir=batch_output_dir,
    )


@pytest.fixture
def stub_comparison_run(tmp_path: Path) -> ComparisonRun:
    """Minimal ComparisonRun for tests that need a BatchResult.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Frozen ComparisonRun with stub values.
    """
    return ComparisonRun(
        mlflow_run_id="stub-run-id",
        mlflow_experiment_id="0",
        tracking_uri="sqlite:///stub.db",
        artifact_location=str(tmp_path / "comparisons" / "artifacts"),
        checkpoint_map={},
        artifact_uri="mlartifacts/0/stub-run-id/artifacts",
    )


@pytest.fixture
def batch_result_success(
    experiment_paths: ExperimentPaths, stub_comparison_run: ComparisonRun
) -> BatchResult:
    """Create a BatchResult where all runs succeeded.

    Args:
        experiment_paths: Paths for the two test experiments.
        stub_comparison_run: Minimal ComparisonRun fixture.

    Returns:
        BatchResult with two successful TrainingRunResults.
    """
    experiment_paths.exp1_ckpt.parent.mkdir(parents=True, exist_ok=True)
    experiment_paths.exp1_ckpt.touch()

    experiment_paths.exp2_ckpt.parent.mkdir(parents=True, exist_ok=True)
    experiment_paths.exp2_ckpt.touch()

    results = [
        TrainingRunResult(
            label="1",
            experiment_id="exp_1",
            checkpoint_path=experiment_paths.exp1_ckpt,
            mlflow_run_id="run1",
            metrics={},
        ),
        TrainingRunResult(
            label="2",
            experiment_id="exp_2",
            checkpoint_path=experiment_paths.exp2_ckpt,
            mlflow_run_id="run2",
            metrics={},
        ),
    ]
    return BatchResult(
        results=results,
        label_map={},
        output_dir=experiment_paths.batch_output_dir,
        comparison_run=stub_comparison_run,
    )


@pytest.fixture
def original_config_path(tmp_path: Path) -> Path:
    """Create a sample experiments.toml.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to the written experiments TOML file.
    """
    config = {
        "output_dir": "/data/output",
        "experiment": [
            {
                "id": "exp_1",
                "dataset": "d1",
                "model": "m1",
                "checkpoint_path": "/old/path/1.ckpt",
            },
            {
                "id": "exp_2",
                "dataset": "d2",
                "model": "m2",
            },
        ],
    }
    path = tmp_path / "experiments.toml"
    with open(path, "wb") as f:
        tomli_w.dump(config, f)
    return path


def test_create_snapshot_config(
    batch_result_success: BatchResult,
    original_config_path: Path,
    tmp_path: Path,
) -> None:
    """Test logic for modifying configuration.

    Args:
        batch_result_success: BatchResult with two successful runs.
        original_config_path: Path to the original experiments TOML.
        tmp_path: Pytest temporary directory.
    """
    with open(original_config_path, "rb") as f:
        original = tomllib.load(f)

    new_out_dir = tmp_path / "new_out"
    snapshot = _create_snapshot_config(original, batch_result_success, new_out_dir)

    assert snapshot["output_dir"] == str(new_out_dir)

    exps = {e["id"]: e for e in snapshot["experiment"]}
    res1 = batch_result_success.results[0]
    res2 = batch_result_success.results[1]

    assert exps["exp_1"]["checkpoint_path"] == str(res1.checkpoint_path)
    assert exps["exp_2"]["checkpoint_path"] == str(res2.checkpoint_path)


def test_capture_batch_context_logs_to_mlflow(
    batch_result_success: BatchResult,
    original_config_path: Path,
) -> None:
    """capture_batch_context logs snapshot config as an MLflow artifact.

    Args:
        batch_result_success: BatchResult with two successful runs.
        original_config_path: Path to the original experiments TOML.
    """
    mock_client = MagicMock()
    with patch(
        "neuralls.workflows.utils.reproducibility.MlflowClient",
        return_value=mock_client,
    ):
        capture_batch_context(batch_result_success, original_config_path)

    # log_artifact should have been called at least once (snapshot/experiments.toml)
    assert mock_client.log_artifact.call_count >= 1
    # First call should be the experiments.toml into the snapshot/ prefix
    first_call = mock_client.log_artifact.call_args_list[0]
    assert first_call.kwargs.get("artifact_path") == "snapshot" or first_call.args[2] == "snapshot"


def test_capture_batch_context_no_custom_dirs(
    batch_result_success: BatchResult,
    original_config_path: Path,
    tmp_path: Path,
) -> None:
    """capture_batch_context does not create any directories outside tmp.

    Only the system temp directory (used internally) should be touched; no
    UUID-based or comparison/ subdirectories should appear in tmp_path.

    Args:
        batch_result_success: BatchResult with two successful runs.
        original_config_path: Path to the original experiments TOML.
        tmp_path: Pytest temporary directory.
    """
    existing_dirs_before = {p for p in tmp_path.rglob("*") if p.is_dir()}

    mock_client = MagicMock()
    with patch(
        "neuralls.workflows.utils.reproducibility.MlflowClient",
        return_value=mock_client,
    ):
        capture_batch_context(batch_result_success, original_config_path)

    # No new subdirectories should be created under tmp_path
    existing_dirs_after = {p for p in tmp_path.rglob("*") if p.is_dir()}
    new_dirs = existing_dirs_after - existing_dirs_before
    # Any new dirs should only be pre-existing experiment dirs (checkpoints already created)
    assert not any("comparison" in str(d) for d in new_dirs)
    assert not any("snapshot" in str(d) for d in new_dirs)


def test_capture_batch_context_partial_failure(
    experiment_paths: ExperimentPaths,
    original_config_path: Path,
    stub_comparison_run: ComparisonRun,
) -> None:
    """Snapshot config is logged even when some checkpoints are missing.

    Args:
        experiment_paths: Paths for the two test experiments.
        original_config_path: Path to the original experiments TOML.
        stub_comparison_run: Minimal ComparisonRun fixture.
    """
    experiment_paths.exp1_ckpt.parent.mkdir(parents=True, exist_ok=True)
    experiment_paths.exp1_ckpt.touch()

    results = [
        TrainingRunResult(
            label="1",
            experiment_id="exp_1",
            checkpoint_path=experiment_paths.exp1_ckpt,
            mlflow_run_id="run1",
            metrics={},
        ),
        TrainingRunResult(
            label="2",
            experiment_id="exp_2",
            checkpoint_path=None,  # type: ignore
            mlflow_run_id="run2",
            metrics={},
        ),
    ]
    batch = BatchResult(
        results=results,
        label_map={},
        output_dir=experiment_paths.batch_output_dir,
        comparison_run=stub_comparison_run,
    )

    mock_client = MagicMock()
    with patch(
        "neuralls.workflows.utils.reproducibility.MlflowClient",
        return_value=mock_client,
    ):
        capture_batch_context(batch, original_config_path)

    # log_artifact should still have been called
    assert mock_client.log_artifact.call_count >= 1
