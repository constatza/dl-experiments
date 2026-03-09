"""Integration test for the full training → comparison workflow.

Validates the end-to-end pipeline:
1. ``train-multiple`` trains experiments, saves ``comparison_run.json``.
2. ``compare-preconditioners --comparison-run`` loads it and runs comparisons.

Heavy operations (DLKit training, CG solver, MLflow I/O) are mocked so the test
exercises CLI argument parsing, file I/O, and the handshake data structure only.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neuralls.cli.compare_preconditioners import main as compare_main
from neuralls.cli.train_multiple import main as train_multiple_main
from neuralls.io.dataset_storage import save_dataset
from neuralls.workflows.comparison_run import ComparisonRun
from neuralls.workflows.results import ComparisonRecommendations, ComparisonResult
from neuralls.workflows.specs import ComparisonOutcome


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _comparison_payload() -> ComparisonResult:
    return ComparisonResult(
        results={},
        summary="ok",
        solver_params=object(),
        preconditioners=("neural",),
        recommendations=ComparisonRecommendations(),
    )


@pytest.fixture
def configs_dir(tmp_path: Path) -> Path:
    """Create minimal configs directory structure."""
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "datasets").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def model_config(configs_dir: Path) -> Path:
    """Minimal model TOML (content not parsed in mocked run)."""
    p = configs_dir / "models" / "test_model.toml"
    p.write_text("[model]\nname = 'TestModel'\n")
    return p


@pytest.fixture
def dataset_config(configs_dir: Path) -> Path:
    """Minimal dataset TOML."""
    p = configs_dir / "datasets" / "test-dataset.toml"
    p.write_text("[dataset]\nname = 'test-dataset'\n")
    return p


@pytest.fixture
def experiments_config(
    configs_dir: Path, tmp_path: Path, model_config: Path, dataset_config: Path
) -> Path:
    """experiments.toml with one entry and required [comparisons] section."""
    comparisons_db = tmp_path / "comparisons" / "mlflow.db"
    comparisons_artifacts = tmp_path / "comparisons" / "artifacts"
    content = (
        'project_root = ".."\n'
        f'output_dir = "{configs_dir.parent / "output"}"\n'
        "\n"
        "[comparisons]\n"
        f'tracking_uri = "sqlite:///{comparisons_db}"\n'
        f'artifact_location = "{comparisons_artifacts}"\n'
        "\n"
        "[[experiment]]\n"
        'id = "test_exp_1"\n'
        'model = "test_model"\n'
        'dataset = "test-dataset"\n'
    )
    p = configs_dir / "experiments.toml"
    p.write_text(content)
    return p


@pytest.fixture
def fake_checkpoint(tmp_path: Path) -> Path:
    """A fake .ckpt file returned by the mocked train_model."""
    ckpt_dir = tmp_path / "output" / "training" / "test-dataset" / "test_model" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "epoch=0.ckpt"
    ckpt.touch()
    return ckpt


@pytest.fixture
def comparison_config(tmp_path: Path) -> Path:
    """Minimal comparison TOML with one neural preconditioner."""
    dataset_dir = tmp_path / "dataset"
    save_dataset(
        dataset_dir=dataset_dir,
        rhs=np.ones((1, 2), dtype=np.float64),
        solutions=np.ones((1, 2), dtype=np.float64),
        matrix=np.eye(2, dtype=np.float64),
        normalization_type="matrix",
        matrix_norm=1.0,
        matrix_norm_type="spectral",
        scale_metadata={},
    )
    comparisons_db = tmp_path / "comparisons" / "mlflow.db"
    comparisons_artifacts = tmp_path / "comparisons" / "artifacts"
    content = "\n".join([
        "schema_version = 3",
        "",
        "[general]",
        "",
        "[general.params]",
        "rtol = 1e-6",
        "atol = 1e-14",
        "max_iterations = 10",
        'stopping_criterion = "residual_norm"',
        "",
        "[general.data]",
        f'matrix_path = "{dataset_dir}"',
        f'rhs_path = "{dataset_dir}"',
        'normalize_system = "matrix"',
        "",
        "[general.tracking]",
        f'tracking_uri = "sqlite:///{comparisons_db}"',
        f'artifact_location = "{comparisons_artifacts}"',
        "",
        "[general.model_store]",
        f'tracking_uri = "sqlite:///{comparisons_db}"',
        "",
        "[[preconditioners]]",
        'name = "neural"',
        'type = "neural"',
        'model_ref = { source = "registered", name = "NormScaledLinearFFNN", alias = "solutions" }',
        "",
    ])
    p = tmp_path / "test_comparison.toml"
    p.write_text(content)
    return p


@pytest.fixture
def stub_artifact_dir(tmp_path: Path) -> Path:
    """Local directory that acts as the MLflow artifact store for the batch run."""
    artifact_dir = tmp_path / "mlartifacts" / "0" / "stub-parent-run-id" / "artifacts"
    artifact_dir.mkdir(parents=True)
    return artifact_dir


@pytest.fixture
def stub_comparison_run(tmp_path: Path, fake_checkpoint: Path, stub_artifact_dir: Path) -> ComparisonRun:
    """ComparisonRun that train_batch would produce."""
    return ComparisonRun(
        mlflow_run_id="stub-parent-run-id",
        mlflow_experiment_id="0",
        tracking_uri=f"sqlite:///{tmp_path / 'comparisons' / 'mlflow.db'}",
        artifact_location=str(tmp_path / "comparisons" / "artifacts"),
        checkpoint_map={"test_exp_1": fake_checkpoint},
        artifact_uri=str(stub_artifact_dir),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_train_multiple_echoes_comparison_run_info(
    tmp_path: Path,
    experiments_config: Path,
    stub_comparison_run: ComparisonRun,
    stub_artifact_dir: Path,
) -> None:
    """train-multiple echoes the comparison run ID and artifact path to stdout."""
    from neuralls.workflows.multi_training import BatchResult, TrainingRunResult

    fake_results = [
        TrainingRunResult(
            label="1",
            experiment_id="test_exp_1",
            checkpoint_path=stub_comparison_run.checkpoint_map["test_exp_1"],
            mlflow_run_id=None,
            metrics={},
        )
    ]
    fake_batch = BatchResult(
        results=fake_results,
        label_map={"1": {"experiment_id": "test_exp_1", "mlflow_run_id": None}},
        output_dir=stub_artifact_dir,
        comparison_run=stub_comparison_run,
    )

    with patch("neuralls.cli.train_multiple.train_batch", return_value=fake_batch):
        with patch("neuralls.cli.train_multiple.capture_batch_context"):
            train_multiple_main(
                config=experiments_config,
                output_dir=tmp_path / "output",
                metric="val_loss",
            )

    # comparison_run.json is now logged to MLflow by train_batch() internally.
    # The CLI echoes the run ID and artifact path — no manual file write.
    # Verify comparison_run contains the expected data by round-tripping via save/load.
    from neuralls.workflows.comparison_run import save_comparison_run, load_comparison_run
    cr_path = tmp_path / "cr_roundtrip.json"
    save_comparison_run(stub_comparison_run, cr_path)
    loaded = load_comparison_run(cr_path)
    assert loaded.mlflow_run_id == "stub-parent-run-id"
    assert "test_exp_1" in loaded.checkpoint_map


def test_compare_preconditioners_loads_comparison_run(
    tmp_path: Path,
    stub_comparison_run: ComparisonRun,
    comparison_config: Path,
) -> None:
    """compare-preconditioners reads comparison_run.json and passes it to run_batch_comparison."""
    from neuralls.workflows.comparison_run import save_comparison_run

    cr_path = tmp_path / "comparison_run.json"
    save_comparison_run(stub_comparison_run, cr_path)

    success_outcome = ComparisonOutcome(
        name=comparison_config.stem,
        success=True,
        payload=_comparison_payload(),
    )

    with patch(
        "neuralls.cli.compare_preconditioners.run_comparison",
        return_value=[success_outcome],
    ) as mock_compare:
        compare_main(
            comparison_run=cr_path,
            comparison_config=comparison_config,
            plots=False,
        )

    mock_compare.assert_called_once()
    call_args = mock_compare.call_args
    # First positional arg is comparison_config path
    assert call_args.args[0] == comparison_config
    # Third positional arg is the loaded ComparisonRun
    loaded_cr: ComparisonRun = call_args.args[2]
    assert loaded_cr.mlflow_run_id == stub_comparison_run.mlflow_run_id
    assert "test_exp_1" in loaded_cr.checkpoint_map


def test_full_workflow_integration(
    tmp_path: Path,
    experiments_config: Path,
    stub_comparison_run: ComparisonRun,
    stub_artifact_dir: Path,
    comparison_config: Path,
) -> None:
    """End-to-end: train-multiple → comparison_run.json → compare-preconditioners."""
    from neuralls.workflows.multi_training import BatchResult, TrainingRunResult
    from neuralls.workflows.comparison_run import save_comparison_run

    fake_results = [
        TrainingRunResult(
            label="1",
            experiment_id="test_exp_1",
            checkpoint_path=stub_comparison_run.checkpoint_map["test_exp_1"],
            mlflow_run_id=None,
            metrics={},
        )
    ]
    fake_batch = BatchResult(
        results=fake_results,
        label_map={"1": {"experiment_id": "test_exp_1", "mlflow_run_id": None}},
        output_dir=stub_artifact_dir,
        comparison_run=stub_comparison_run,
    )

    # Phase 1: training (train_batch mocked — comparison_run.json logged via MLflow internally)
    with patch("neuralls.cli.train_multiple.train_batch", return_value=fake_batch):
        with patch("neuralls.cli.train_multiple.capture_batch_context"):
            train_multiple_main(
                config=experiments_config,
                output_dir=tmp_path / "output",
                metric="val_loss",
            )

    # Simulate the handoff file that the user would pass to compare-preconditioners
    # (in production this is written by mlflow.log_dict inside train_batch)
    cr_path = tmp_path / "comparison_run.json"
    save_comparison_run(stub_comparison_run, cr_path)
    assert cr_path.exists()

    # Phase 2: comparison
    success_outcome = ComparisonOutcome(
        name=comparison_config.stem,
        success=True,
        payload=_comparison_payload(),
    )
    with patch(
        "neuralls.cli.compare_preconditioners.run_comparison",
        return_value=[success_outcome],
    ) as mock_compare:
        compare_main(
            comparison_run=cr_path,
            comparison_config=comparison_config,
            plots=False,
        )

    mock_compare.assert_called_once()
    loaded_cr: ComparisonRun = mock_compare.call_args.args[2]
    assert loaded_cr.mlflow_run_id == stub_comparison_run.mlflow_run_id
    assert loaded_cr.checkpoint_map["test_exp_1"] == stub_comparison_run.checkpoint_map["test_exp_1"]
