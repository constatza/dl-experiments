"""Integration test: train_batch checkpoint then comparison model loading."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pytest
import tomli_w
import torch

from neuralls.platform.config.models.experiments import CaseConfig
from neuralls.platform.config.loaders import load_raw_toml
from neuralls.composition.experiments.comparison_batch import run_comparison_batch
from neuralls.composition.generation.process_data import process_data_from_config
from neuralls.platform.storage.datasets import resolve_dataset_paths
from neuralls.composition.experiments.multi_training import BatchResult, train_batch
from neuralls.composition.comparison.models import ComparisonParams

TEST_MAX_EPOCHS = 1
TEST_LIMIT_TRAIN_BATCHES = 1
TEST_LIMIT_VAL_BATCHES = 1
TEST_CHECKPOINT_EVERY_N_EPOCHS = 1
TEST_COMPARISON_MAX_ITERATIONS = 20


@pytest.fixture
def integration_root(tmp_path: Path) -> Path:
    """Root directory for this integration scenario."""
    return tmp_path


@pytest.fixture
def config_root(integration_root: Path) -> Path:
    """Configuration root with expected subdirectories."""
    root = integration_root / "configs"
    (root / "jobs").mkdir(parents=True)
    (root / "datasets").mkdir(parents=True)
    return root


@pytest.fixture
def output_root(integration_root: Path) -> Path:
    """Output root for MLflow and workflow artifacts."""
    path = integration_root / "output"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def processed_root(integration_root: Path) -> Path:
    """Processed data root for generated normalized datasets."""
    path = integration_root / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def raw_system_files(integration_root: Path) -> tuple[Path, Path]:
    """Write a tiny SPD matrix and RHS vector for data generation."""
    raw_dir = integration_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = raw_dir / "matrix.txt"
    rhs_path = raw_dir / "rhs.txt"

    matrix = np.array(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.5, 0.2, 0.0],
            [0.0, 0.2, 2.8, 0.1],
            [0.0, 0.0, 0.1, 2.2],
        ],
        dtype=np.float64,
    )
    rhs = np.array([1.0, 0.5, -0.2, 0.8], dtype=np.float64)

    np.savetxt(matrix_path, matrix)
    np.savetxt(rhs_path, rhs)
    return matrix_path, rhs_path


@pytest.fixture
def data_config_path(
    config_root: Path,
    raw_system_files: tuple[Path, Path],
    processed_root: Path,
) -> Path:
    """Dataset config used by train_batch and later comparison."""
    matrix_path, rhs_path = raw_system_files
    path = config_root / "datasets" / "tiny-data.toml"
    payload = {
        "id": "tiny-data",
        "source": {
            "matrix_path": str(matrix_path),
            "rhs_path": str(rhs_path),
        },
        "generation": {
            "normalize": "matrix",
            "shuffle": True,
            "seed": 42,
            "strategy": [{"name": "random", "samples": 8}],
        },
        "output": {
            "data_dir": str(processed_root),
        },
    }
    with path.open("wb") as fh:
        tomli_w.dump(payload, fh)
    return path


@pytest.fixture
def job_config_path(config_root: Path) -> Path:
    """Minimal job config: single epoch with checkpoint each epoch."""
    path = config_root / "jobs" / "tiny-linear.toml"
    payload = {
        "run": {
            "type": "train",
            "seed": 42,
        },
        "experiment": {
            "name": "tiny-linear",
        },
        "model": {
            "class": "LinearNetwork",
            "module_path": "dlkit.nn",
        },
        "training": {
            "trainer": {
                "max_epochs": TEST_MAX_EPOCHS,
                "accelerator": "cpu",
                "enable_checkpointing": True,
                "num_sanity_val_steps": 0,
                "limit_train_batches": TEST_LIMIT_TRAIN_BATCHES,
                "limit_val_batches": TEST_LIMIT_VAL_BATCHES,
                "callbacks": [
                    {
                        "name": "ModelCheckpoint",
                        "filename": "tiny-epoch{epoch:02d}",
                        "monitor": "val_loss",
                        "save_top_k": 1,
                        "save_weights_only": False,
                        "every_n_epochs": TEST_CHECKPOINT_EVERY_N_EPOCHS,
                        "enable_version_counter": False,
                    }
                ],
            },
            "optimizer": {
                "default_optimizer": {
                    "name": "AdamW",
                    "lr": 1e-3,
                },
            },
        },
        "data": {
            "class": "FlexibleDataset",
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": True,
        },
    }
    with path.open("wb") as fh:
        tomli_w.dump(payload, fh)
    return path


@pytest.fixture
def runs_config_path(
    config_root: Path,
    job_config_path: Path,
    data_config_path: Path,
    output_root: Path,
    processed_root: Path,
) -> Path:
    """Case config with one job, one dataset, and one comparison entry."""
    tracking_uri = f"sqlite:///{(output_root / 'mlruns' / 'mlflow.db').as_posix()}"
    path = config_root / "experiments.toml"
    path.write_text(
        "\n".join(
            [
                "[mlflow]",
                f'tracking_uri = "{tracking_uri}"',
                "",
                "[names]",
                'training = "neuralls-training"',
                'comparison = "Comparisons"',
                "",
                "[comparison_defaults]",
                "rtol = 1e-6",
                "atol = 1e-14",
                f"max_iterations = {TEST_COMPARISON_MAX_ITERATIONS}",
                'stopping_criterion = "residual_norm"',
                "m_max = 20",
                "",
                "[[datasets]]",
                'id = "tiny-data"',
                f'path = "datasets/{data_config_path.name}"',
                "",
                "[[jobs]]",
                'id = "tiny-linear"',
                f'path = "jobs/{job_config_path.name}"',
                "",
                "[[comparisons]]",
                'id = "tiny-comparison"',
                'matrix_dataset = "tiny-data"',
                'rhs_dataset = "tiny-data"',
                'method = "comparison.toml"',
                "",
                "[[experiments]]",
                'id = "tiny-linear-run"',
                'job = "tiny-linear"',
                'dataset = "tiny-data"',
                "",
            ]
        )
    )
    return path


@pytest.fixture
def trained_batch_result(
    runs_config_path: Path,
    data_config_path: Path,
    processed_root: Path,
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    neuralls_settings,
) -> BatchResult:
    """Execute train_batch once for this integration scenario."""
    tracking_uri = f"sqlite:///{(output_root / 'mlruns' / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_ARTIFACT_URI", str((output_root / "mlartifacts").resolve()))
    monkeypatch.chdir(runs_config_path.parent.parent)
    process_data_from_config(data_config_path, neuralls_settings)
    cfg = CaseConfig.model_validate(load_raw_toml(runs_config_path))
    batch_result = train_batch(
        cfg=cfg,
        configs_dir=runs_config_path.parent,
        settings=neuralls_settings,
    )
    assert len(batch_result.results) == 1, "Integration test must train exactly one network."
    return batch_result


@pytest.fixture
def comparison_config_path(
    trained_batch_result: BatchResult,
    config_root: Path,
) -> Path:
    """Method override TOML that injects the just-trained neural preconditioner.

    Data paths are resolved by the case config via dataset registry — only the
    preconditioner list is specified here.
    """
    trained = trained_batch_result.results[0]
    run_id = trained.mlflow_run_id
    if run_id is None:
        tracking_uri = (
            f"sqlite:///{(config_root.parent / 'output' / 'mlruns' / 'mlflow.db').as_posix()}"
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("tiny-data")
        with mlflow.start_run(run_name="tiny-linear-fallback-artifact") as run:
            mlflow.log_artifact(str(trained.checkpoint_path), artifact_path="model")
            run_id = run.info.run_id
    path = config_root / "comparison.toml"
    path.write_text(
        "\n".join(
            [
                "[[preconditioners]]",
                'name = "neural_norm_scaled_linear_ffnn"',
                'type = "neural"',
                "limit_iters = 1",
                'fallback = "identity"',
                f'model_ref = {{ source = "logged", run_id = "{run_id}", artifact_path = "model" }}',
                "",
            ]
        )
    )
    return path


@pytest.mark.integration
@pytest.mark.slow
def test_train_multiple_checkpoint_then_compare_model_loading(
    trained_batch_result: BatchResult,
    comparison_config_path: Path,  # noqa: ARG001 — ensures method file is written first
    runs_config_path: Path,
) -> None:
    """Train one tiny run then load its checkpoint through comparison."""
    assert len(trained_batch_result.results) == 1
    trained = trained_batch_result.results[0]
    assert trained.checkpoint_path.exists()

    checkpoint_data = torch.load(trained.checkpoint_path, map_location="cpu")
    assert checkpoint_data.get("epoch") == 0
    assert "epoch=00" in trained.checkpoint_path.name

    dataset_paths = resolve_dataset_paths(
        runs_config_path.parent.parent / "processed" / "tiny-data"
    )
    assert dataset_paths.manifest_path.exists()
    assert dataset_paths.rhs_path.exists()
    assert dataset_paths.solutions_path.exists()
    assert dataset_paths.matrix_zarr_dir.exists()

    outcomes = run_comparison_batch(
        experiments_config_path=runs_config_path,
        params=ComparisonParams(),
    )

    assert len(outcomes) == 1
    assert outcomes[0].success is True
    assert outcomes[0].payload is not None
    assert "neural_norm_scaled_linear_ffnn" in outcomes[0].payload.preconditioners
