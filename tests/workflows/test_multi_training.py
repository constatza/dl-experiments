"""Unit tests for the multi_training batch training workflow.

Covers:
- _read_sidecar(): reads mlflow_run.json, returns None when absent
- _fetch_mlflow_metrics(): queries MlflowClient for run metrics
- _make_label_map(): builds correct label→identity mapping
- _resolve_config_paths(): resolves model and dataset config paths
- train_batch(): raises ValueError for empty experiments TOML
- plot_metric_comparison(): renders barplot without error
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neuralls.configuration.experiments import ExperimentsConfig
from neuralls.io.toml_loader import load_raw_toml
from neuralls.workflows.multi_training import (
    BatchResult,
    TrainingRunResult,
    _collect_batch_metrics,
    _fetch_mlflow_metrics,
    _make_label_map,
    _read_sidecar,
    _resolve_config_paths,
    _train_single,
    train_batch,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sidecar_data() -> dict[str, str]:
    """Canonical sidecar JSON payload."""
    return {
        "run_id": "abc123def456",
        "experiment_id": "1",
        "tracking_uri": "sqlite:///mlflow.db",
        "artifacts_destination": "/tmp/mlartifacts",
    }


@pytest.fixture
def sidecar_file(tmp_path: Path, sidecar_data: dict[str, str]) -> Path:
    """Write a sidecar JSON next to a fake checkpoint."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    sidecar_path = ckpt_dir / "mlflow_run.json"
    sidecar_path.write_text(json.dumps(sidecar_data))
    return ckpt_dir / "model.ckpt"  # checkpoint path (sidecar is beside it)


@pytest.fixture
def training_run_results() -> list[TrainingRunResult]:
    """Three completed training run results for label map tests."""
    return [
        TrainingRunResult(
            label="1",
            experiment_id="ffnn_solutions",
            checkpoint_path=Path("/tmp/a/model.ckpt"),
            mlflow_run_id="aaa111",
            metrics={"eval/rel_error": 0.05, "val_loss": 0.01},
        ),
        TrainingRunResult(
            label="2",
            experiment_id="ffnn_eigenvectors",
            checkpoint_path=Path("/tmp/b/model.ckpt"),
            mlflow_run_id="bbb222",
            metrics={"eval/rel_error": 0.03},
        ),
        TrainingRunResult(
            label="3",
            experiment_id="ffnn_rhs_largest",
            checkpoint_path=Path("/tmp/c/model.ckpt"),
            mlflow_run_id=None,
            metrics={},
        ),
    ]


@pytest.fixture
def model_config_file(tmp_path: Path) -> Path:
    """Create a minimal model config TOML file."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    path = models_dir / "ffnn.toml"
    path.write_text("[model]\nname = 'ffnn'\n")
    return path


@pytest.fixture
def dataset_config_file(tmp_path: Path) -> Path:
    """Create a minimal dataset config TOML file."""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    path = datasets_dir / "test-solutions.toml"
    path.write_text("[dataset]\nname = 'test-solutions'\n")
    return path


@pytest.fixture
def experiment_entry() -> dict[str, str]:
    """Single experiment TOML entry."""
    return {"id": "ffnn_test", "model": "ffnn", "dataset": "test-solutions"}


@pytest.fixture
def minimal_experiments_toml(tmp_path: Path) -> Path:
    """Experiments TOML with no [[experiment]] entries (for error testing)."""
    path = tmp_path / "experiments.toml"
    comparisons_db = tmp_path / "comparisons" / "mlflow.db"
    comparisons_artifacts = tmp_path / "comparisons" / "artifacts"
    content = (
        'project_root = ".."\n'
        f'output_dir = "{tmp_path / "output"}"\n'
        "\n"
        "[comparisons]\n"
        f'tracking_uri = "sqlite:///{comparisons_db}"\n'
        f'artifact_location = "{comparisons_artifacts}"\n'
    )
    path.write_text(content)
    return path


@pytest.fixture
def valid_experiments_toml(tmp_path: Path, model_config_file: Path, dataset_config_file: Path) -> Path:
    """Valid experiments TOML with one [[experiment]] entry pointing to real configs."""
    path = tmp_path / "experiments.toml"
    comparisons_db = tmp_path / "comparisons" / "mlflow.db"
    comparisons_artifacts = tmp_path / "comparisons" / "artifacts"
    content = (
        'project_root = ".."\n'
        f'output_dir = "{tmp_path / "output"}"\n'
        "\n"
        "[comparisons]\n"
        f'tracking_uri = "sqlite:///{comparisons_db}"\n'
        f'artifact_location = "{comparisons_artifacts}"\n'
        "\n"
        "[[experiment]]\n"
        'id = "ffnn_test"\n'
        'model = "ffnn"\n'
        'dataset = "test-solutions"\n'
    )
    path.write_text(content)
    return path


@pytest.fixture
def mock_mlflow_run() -> MagicMock:
    """Mock MlflowClient.get_run() return value with sample metrics."""
    run = MagicMock()
    run.data.metrics = {"eval/rel_error": 0.042, "val_loss": 0.009}
    return run


# ---------------------------------------------------------------------------
# _read_sidecar()
# ---------------------------------------------------------------------------


class TestReadSidecar:
    """Tests for _read_sidecar() pure helper."""

    def test_returns_dict_when_sidecar_exists(
        self, sidecar_file: Path, sidecar_data: dict[str, str]
    ) -> None:
        """Returns parsed dict when mlflow_run.json is present."""
        result = _read_sidecar(sidecar_file)
        assert result == sidecar_data

    def test_returns_none_when_sidecar_missing(self, tmp_path: Path) -> None:
        """Returns None when no sidecar file exists beside the checkpoint."""
        checkpoint = tmp_path / "checkpoints" / "model.ckpt"
        checkpoint.parent.mkdir()
        result = _read_sidecar(checkpoint)
        assert result is None

    def test_run_id_accessible(self, sidecar_file: Path) -> None:
        """run_id key is accessible from parsed sidecar."""
        result = _read_sidecar(sidecar_file)
        assert result is not None
        assert result["run_id"] == "abc123def456"

    def test_tracking_uri_accessible(self, sidecar_file: Path) -> None:
        """tracking_uri key is accessible from parsed sidecar."""
        result = _read_sidecar(sidecar_file)
        assert result is not None
        assert result["tracking_uri"] == "sqlite:///mlflow.db"


# ---------------------------------------------------------------------------
# _fetch_mlflow_metrics()
# ---------------------------------------------------------------------------


class TestFetchMlflowMetrics:
    """Tests for _fetch_mlflow_metrics() with mocked MlflowClient."""

    def test_returns_metrics_dict(self, mock_mlflow_run: MagicMock) -> None:
        """Returns flat metric dict from MLflow run data."""
        with (
            patch("neuralls.workflows.multi_training.MlflowClient") as mock_client_cls,
            patch("neuralls.workflows.multi_training.mlflow.set_tracking_uri"),
        ):
            mock_client_cls.return_value.get_run.return_value = mock_mlflow_run
            result = _fetch_mlflow_metrics("abc123", "sqlite:///mlflow.db")

        assert result == {"eval/rel_error": 0.042, "val_loss": 0.009}

    def test_calls_get_run_with_correct_run_id(self, mock_mlflow_run: MagicMock) -> None:
        """MlflowClient.get_run() is called with the provided run_id."""
        with (
            patch("neuralls.workflows.multi_training.MlflowClient") as mock_client_cls,
            patch("neuralls.workflows.multi_training.mlflow.set_tracking_uri"),
        ):
            mock_client = mock_client_cls.return_value
            mock_client.get_run.return_value = mock_mlflow_run
            _fetch_mlflow_metrics("run_xyz", "sqlite:///test.db")
            mock_client.get_run.assert_called_once_with("run_xyz")

    def test_returns_empty_dict_for_no_metrics(self) -> None:
        """Returns empty dict when run has no metrics."""
        run = MagicMock()
        run.data.metrics = {}
        with (
            patch("neuralls.workflows.multi_training.MlflowClient") as mock_client_cls,
            patch("neuralls.workflows.multi_training.mlflow.set_tracking_uri"),
        ):
            mock_client_cls.return_value.get_run.return_value = run
            result = _fetch_mlflow_metrics("run_empty", "sqlite:///test.db")
        assert result == {}


# ---------------------------------------------------------------------------
# _make_label_map()
# ---------------------------------------------------------------------------


class TestMakeLabelMap:
    """Tests for _make_label_map() pure helper."""

    def test_produces_correct_labels(
        self, training_run_results: list[TrainingRunResult]
    ) -> None:
        """All labels from results appear as keys in the map."""
        result = _make_label_map(training_run_results)
        assert set(result.keys()) == {"1", "2", "3"}

    def test_experiment_id_preserved(
        self, training_run_results: list[TrainingRunResult]
    ) -> None:
        """experiment_id matches original TrainingRunResult."""
        result = _make_label_map(training_run_results)
        assert result["1"]["experiment_id"] == "ffnn_solutions"
        assert result["2"]["experiment_id"] == "ffnn_eigenvectors"

    def test_mlflow_run_id_preserved(
        self, training_run_results: list[TrainingRunResult]
    ) -> None:
        """mlflow_run_id is correctly mapped, including None values."""
        result = _make_label_map(training_run_results)
        assert result["1"]["mlflow_run_id"] == "aaa111"
        assert result["3"]["mlflow_run_id"] is None

    def test_empty_results_gives_empty_map(self) -> None:
        """Empty input produces empty mapping."""
        assert _make_label_map([]) == {}


# ---------------------------------------------------------------------------
# _resolve_config_paths()
# ---------------------------------------------------------------------------


class TestResolveConfigPaths:
    """Tests for _resolve_config_paths() pure helper."""

    def test_resolves_existing_paths(
        self,
        tmp_path: Path,
        model_config_file: Path,
        dataset_config_file: Path,
        experiment_entry: dict[str, str],
    ) -> None:
        """Returns correct model and dataset paths when files exist."""
        model_path, data_path = _resolve_config_paths(experiment_entry, tmp_path)
        assert model_path == model_config_file
        assert data_path == dataset_config_file

    def test_raises_for_missing_model_config(
        self, tmp_path: Path, dataset_config_file: Path
    ) -> None:
        """Raises FileNotFoundError when model config does not exist."""
        entry = {"model": "nonexistent_model", "dataset": "test-solutions"}
        with pytest.raises(FileNotFoundError, match="Model config not found"):
            _resolve_config_paths(entry, tmp_path)

    def test_raises_for_missing_dataset_config(
        self, tmp_path: Path, model_config_file: Path
    ) -> None:
        """Raises FileNotFoundError when dataset config does not exist."""
        entry = {"model": "ffnn", "dataset": "nonexistent_dataset"}
        with pytest.raises(FileNotFoundError, match="Dataset config not found"):
            _resolve_config_paths(entry, tmp_path)


# ---------------------------------------------------------------------------
# _train_single()
# ---------------------------------------------------------------------------


class TestTrainSingle:
    """Tests for single-run training orchestration helper."""

    def test_assigns_dataset_alias_when_run_metadata_available(self, tmp_path: Path) -> None:
        """Alias assignment is attempted when run_id + tracking_uri are available."""
        model_cfg = tmp_path / "model.toml"
        model_cfg.write_text("[MODEL]\nname = 'NormScaledLinearFFNN'\n")
        data_cfg = tmp_path / "dataset-a.toml"
        data_cfg.write_text("id = 'dataset-a'\n")
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        ckpt = ckpt_dir / "model.ckpt"
        ckpt.touch()
        sidecar = {
            "run_id": "run-123",
            "tracking_uri": "sqlite:////tmp/mlflow.db",
        }
        (ckpt_dir / "mlflow_run.json").write_text(json.dumps(sidecar))

        with (
            patch("neuralls.workflows.multi_training.train_model", return_value=ckpt),
            patch("neuralls.workflows.multi_training.assign_dataset_alias_to_registered_model", return_value=5) as mock_assign,
            patch("neuralls.workflows.multi_training._fetch_mlflow_metrics", return_value={}),
            patch("neuralls.workflows.multi_training.MlflowClient"),
        ):
            result = _train_single(
                experiment_id="exp-1",
                model_config_path=model_cfg,
                data_config_path=data_cfg,
                label="1",
                output_root=None,
            )

        assert result.mlflow_run_id == "run-123"
        mock_assign.assert_called_once_with(
            tracking_uri="sqlite:////tmp/mlflow.db",
            registered_model_name="NormScaledLinearFFNN",
            run_id="run-123",
            dataset_alias="dataset-a",
        )

    def test_skips_alias_assignment_when_model_name_missing(self, tmp_path: Path) -> None:
        """No alias assignment when model config has no [MODEL].name."""
        model_cfg = tmp_path / "model.toml"
        model_cfg.write_text("[SESSION]\nname = 'NoModelName'\n")
        data_cfg = tmp_path / "dataset-b.toml"
        data_cfg.write_text("id = 'dataset-b'\n")
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        ckpt = ckpt_dir / "model.ckpt"
        ckpt.touch()
        sidecar = {
            "run_id": "run-456",
            "tracking_uri": "sqlite:////tmp/mlflow.db",
        }
        (ckpt_dir / "mlflow_run.json").write_text(json.dumps(sidecar))

        with (
            patch("neuralls.workflows.multi_training.train_model", return_value=ckpt),
            patch("neuralls.workflows.multi_training.assign_dataset_alias_to_registered_model") as mock_assign,
            patch("neuralls.workflows.multi_training._fetch_mlflow_metrics", return_value={}),
            patch("neuralls.workflows.multi_training.MlflowClient"),
        ):
            _train_single(
                experiment_id="exp-2",
                model_config_path=model_cfg,
                data_config_path=data_cfg,
                label="1",
                output_root=None,
            )

        mock_assign.assert_not_called()


# ---------------------------------------------------------------------------
# train_batch()
# ---------------------------------------------------------------------------


def _make_mlflow_batch_run_mock(run_id: str = "batch-run-id", experiment_id: str = "0") -> MagicMock:
    """Build a context manager mock for mlflow.start_run in train_batch().

    Args:
        run_id: Fake MLflow batch run UUID.
        experiment_id: Fake MLflow experiment ID.

    Returns:
        MagicMock configured as a context manager yielding a fake run.
    """
    mock_run = MagicMock()
    mock_run.info.run_id = run_id
    mock_run.info.experiment_id = experiment_id
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_run)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    return mock_ctx


class TestTrainBatch:
    """Tests for train_batch() orchestration."""

    def test_raises_for_empty_experiments_toml(
        self, minimal_experiments_toml: Path
    ) -> None:
        """Raises ValueError when TOML has no [[experiment]] entries."""
        with pytest.raises(ValueError, match="No .* entries found"):
            train_batch(minimal_experiments_toml)

    def test_calls_train_model_for_each_experiment(
        self,
        valid_experiments_toml: Path,
        tmp_path: Path,
    ) -> None:
        """train_model() is called once per experiment entry."""
        fake_ckpt = tmp_path / "ckpt" / "model.ckpt"
        fake_ckpt.parent.mkdir()
        fake_ckpt.touch()

        mock_ctx = _make_mlflow_batch_run_mock(run_id="run-001", experiment_id="0")
        with (
            patch("neuralls.workflows.multi_training.train_model", return_value=fake_ckpt) as mock_train,
            patch("neuralls.workflows.multi_training.setup_comparison_tracking"),
            patch("neuralls.workflows.multi_training.mlflow.start_run", return_value=mock_ctx),
            patch("neuralls.workflows.multi_training.mlflow.log_param"),
            patch("neuralls.workflows.multi_training.mlflow.log_metrics"),
            patch("neuralls.workflows.multi_training.mlflow.log_dict"),
            patch("neuralls.workflows.multi_training.mlflow.get_artifact_uri", return_value="mlartifacts/0/run-001/artifacts"),
        ):
            result = train_batch(valid_experiments_toml)

        assert mock_train.call_count == 1
        assert len(result.results) == 1
        assert result.results[0].label == "1"
        assert result.results[0].experiment_id == "ffnn_test"

    def test_output_dir_derived_from_artifact_uri(
        self,
        valid_experiments_toml: Path,
        tmp_path: Path,
    ) -> None:
        """output_dir in BatchResult is the local path from mlflow.get_artifact_uri()."""
        fake_ckpt = tmp_path / "ckpt" / "model.ckpt"
        fake_ckpt.parent.mkdir()
        fake_ckpt.touch()

        fake_artifact_uri = "mlartifacts/1/run-abc/artifacts"
        mock_ctx = _make_mlflow_batch_run_mock(run_id="run-abc", experiment_id="1")
        with (
            patch("neuralls.workflows.multi_training.train_model", return_value=fake_ckpt),
            patch("neuralls.workflows.multi_training.setup_comparison_tracking"),
            patch("neuralls.workflows.multi_training.mlflow.start_run", return_value=mock_ctx),
            patch("neuralls.workflows.multi_training.mlflow.log_param"),
            patch("neuralls.workflows.multi_training.mlflow.log_metrics"),
            patch("neuralls.workflows.multi_training.mlflow.log_dict"),
            patch("neuralls.workflows.multi_training.mlflow.get_artifact_uri", return_value=fake_artifact_uri),
        ):
            result = train_batch(valid_experiments_toml)

        # output_dir is the resolved absolute path derived from the MLflow artifact URI
        assert result.output_dir == Path(fake_artifact_uri).resolve()

    def test_label_map_populated(
        self,
        valid_experiments_toml: Path,
        tmp_path: Path,
    ) -> None:
        """BatchResult.label_map contains the experiment entry."""
        fake_ckpt = tmp_path / "ckpt" / "model.ckpt"
        fake_ckpt.parent.mkdir()
        fake_ckpt.touch()

        mock_ctx = _make_mlflow_batch_run_mock(run_id="run-lmap", experiment_id="0")
        with (
            patch("neuralls.workflows.multi_training.train_model", return_value=fake_ckpt),
            patch("neuralls.workflows.multi_training.setup_comparison_tracking"),
            patch("neuralls.workflows.multi_training.mlflow.start_run", return_value=mock_ctx),
            patch("neuralls.workflows.multi_training.mlflow.log_param"),
            patch("neuralls.workflows.multi_training.mlflow.log_metrics"),
            patch("neuralls.workflows.multi_training.mlflow.log_dict"),
            patch("neuralls.workflows.multi_training.mlflow.get_artifact_uri", return_value="mlartifacts/0/run-lmap/artifacts"),
        ):
            result = train_batch(valid_experiments_toml)

        assert "1" in result.label_map
        assert result.label_map["1"]["experiment_id"] == "ffnn_test"

    def test_comparison_run_tracking_uri_from_comparisons_config(
        self,
        valid_experiments_toml: Path,
        tmp_path: Path,
    ) -> None:
        """ComparisonRun.tracking_uri comes from [comparisons] config, not derived.

        Args:
            valid_experiments_toml: Valid experiments TOML with [comparisons] section.
            tmp_path: Pytest temporary directory.
        """
        fake_ckpt = tmp_path / "ckpt" / "model.ckpt"
        fake_ckpt.parent.mkdir()
        fake_ckpt.touch()

        cfg = ExperimentsConfig.model_validate(load_raw_toml(valid_experiments_toml))
        mock_ctx = _make_mlflow_batch_run_mock(run_id="run-uri-test", experiment_id="0")
        with (
            patch("neuralls.workflows.multi_training.train_model", return_value=fake_ckpt),
            patch("neuralls.workflows.multi_training.setup_comparison_tracking"),
            patch("neuralls.workflows.multi_training.mlflow.start_run", return_value=mock_ctx),
            patch("neuralls.workflows.multi_training.mlflow.log_param"),
            patch("neuralls.workflows.multi_training.mlflow.log_metrics"),
            patch("neuralls.workflows.multi_training.mlflow.log_dict"),
            patch("neuralls.workflows.multi_training.mlflow.get_artifact_uri", return_value="mlartifacts/0/run-uri-test/artifacts"),
        ):
            result = train_batch(valid_experiments_toml)

        assert result.comparison_run.tracking_uri == cfg.comparisons.tracking_uri
        assert result.comparison_run.artifact_location == cfg.comparisons.artifact_location


# ---------------------------------------------------------------------------
# _collect_batch_metrics()
# ---------------------------------------------------------------------------


class TestCollectBatchMetrics:
    """Tests for _collect_batch_metrics() pure helper."""

    def test_averages_metrics_across_results(
        self, training_run_results: list[TrainingRunResult]
    ) -> None:
        """Returns averaged metrics from all training results."""
        result = _collect_batch_metrics(training_run_results, None)
        # eval/rel_error appears in labels 1 and 2: avg = (0.05 + 0.03) / 2
        assert "avg_eval/rel_error" in result
        assert abs(result["avg_eval/rel_error"] - 0.04) < 1e-9

    def test_returns_empty_dict_for_empty_results(self) -> None:
        """Returns empty dict when results list is empty."""
        assert _collect_batch_metrics([], None) == {}

    def test_returns_empty_dict_when_no_metrics(self) -> None:
        """Returns empty dict when all results have empty metrics."""
        results = [
            TrainingRunResult("1", "exp", Path("/x.ckpt"), None, {}),
        ]
        assert _collect_batch_metrics(results, None) == {}


# ---------------------------------------------------------------------------
# plot_metric_comparison()
# ---------------------------------------------------------------------------


class TestPlotMetricComparison:
    """Tests for plot_metric_comparison() via plotting module."""

    def test_saves_png_file(self, tmp_path: Path) -> None:
        """Saves a PNG file to the specified path."""
        from neuralls.plotting import plot_metric_comparison

        save_path = tmp_path / "comparison.png"
        plot_metric_comparison(
            labels=["1", "2", "3"],
            values=[0.05, 0.03, 0.07],
            metric_name="eval/rel_error",
            save_path=save_path,
        )
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Creates parent directories if they do not exist."""
        from neuralls.plotting import plot_metric_comparison

        save_path = tmp_path / "nested" / "deep" / "comparison.png"
        plot_metric_comparison(
            labels=["1"],
            values=[0.1],
            metric_name="val_loss",
            save_path=save_path,
        )
        assert save_path.exists()

    def test_legend_parameter_accepted(self, tmp_path: Path) -> None:
        """Accepts legend mapping without error."""
        from neuralls.plotting import plot_metric_comparison

        save_path = tmp_path / "comparison_legend.png"
        legend = {"1": "exp_a (run: abc)", "2": "exp_b (run: def)"}
        plot_metric_comparison(
            labels=["1", "2"],
            values=[0.04, 0.02],
            metric_name="eval/rel_error",
            legend=legend,
            save_path=save_path,
        )
        assert save_path.exists()

    def test_no_save_path_does_not_raise(self) -> None:
        """Calling without save_path closes figure without error."""
        from neuralls.plotting import plot_metric_comparison

        plot_metric_comparison(
            labels=["1", "2"],
            values=[0.1, 0.2],
            metric_name="test_metric",
        )
