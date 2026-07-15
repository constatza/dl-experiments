"""Tests for eval-only workflow assembly."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from dlkit.infrastructure.config.factories import load_job
from dlkit.infrastructure.config.job_config import InferenceJobConfig

from neuralls.composition.assignments.evaluation import (
    EvaluationAssignmentContext,
    EvaluationBatchResult,
    EvaluationConfigPaths,
    EvaluationRunResult,
    _materialize_inference_settings,
    _as_inference_job,
    _with_eval_runtime_dataset,
    write_eval_metric_report,
)
from neuralls.composition.assignments.runtime_dataset_contract import (
    default_training_dataset_contract,
)
from neuralls.platform.config.dataset_entries import entry_from_path
from neuralls.platform.config.models.experiments import AssignmentEntry, CaseConfig
from neuralls.platform.tracking.evaluation_artifacts import (
    TrainingConfigArtifacts,
    TrainingEvaluationArtifacts,
)


def _write_split(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "train": [0, 1],
                "validation": [2],
                "test": [3],
                "predict": [],
            }
        ),
        encoding="utf-8",
    )


def test_eval_materialization_injects_split_file_and_array_datamodule(tmp_path: Path) -> None:
    job_path = Path(__file__).resolve().parents[2] / "configs/jobs/ffnn/ffnn.toml"
    job = load_job(job_path)
    split_file = tmp_path / "split.json"
    checkpoint = tmp_path / "model.ckpt"
    rhs = tmp_path / "rhs.npy"
    solution = tmp_path / "solutions.npy"
    matrix = tmp_path / "matrix.npy"
    _write_split(split_file)
    checkpoint.write_bytes(b"checkpoint")
    np.save(rhs, np.zeros((4, 3)))
    np.save(solution, np.ones((4, 3)))
    np.save(matrix, np.eye(3))
    contract = default_training_dataset_contract()
    features = [
        entry_from_path(
            rhs,
            name=contract.primary_input_name,
            model_input=True,
            role="feature",
        ),
        entry_from_path(
            matrix,
            name=contract.matrix_input_name,
            model_input=False,
            role="feature",
        ),
    ]
    targets = [
        entry_from_path(
            solution,
            name=contract.target_name,
            model_input=False,
            role="target",
        )
    ]

    eval_job = _with_eval_runtime_dataset(
        job,
        split_file=split_file,
        checkpoint_path=checkpoint,
        features=features,
        targets=targets,
    )
    inference_job = _as_inference_job(eval_job)

    assert isinstance(inference_job, InferenceJobConfig)
    assert inference_job.run.type == "predict"
    assert inference_job.model.checkpoint == str(checkpoint)
    assert inference_job.data is not None
    assert inference_job.data.splits.filepath == split_file
    assert inference_job.data.module.name == "ArrayDataModule"
    assert inference_job.data.module.module_path == "dlkit.engine.adapters.lightning.datamodules"
    assert [entry.name for entry in inference_job.data.features] == ["x", "matrix"]
    assert [entry.name for entry in inference_job.data.targets] == ["y"]


def test_write_eval_metric_report_accepts_plain_and_prefixed_metric(tmp_path: Path) -> None:
    batch = EvaluationBatchResult(
        results=[
            EvaluationRunResult(
                label="1",
                assignment_id="a1",
                assignment_display_name="A1",
                training_run_id="train-1",
                evaluation_run_id="eval-1",
                metrics={"mae": 0.2, "eval/rmse": 0.4},
                split_file=tmp_path / "split.json",
                figures_dir=tmp_path / "figures",
            )
        ],
        label_map={},
        output_dir=tmp_path,
    )

    report_dir = write_eval_metric_report(batch, metric="rmse")

    assert report_dir == tmp_path
    assert (tmp_path / "batch_metric_rmse.png").exists()
    assert (tmp_path / "batch_eval_labels.json").exists()


def test_materialize_inference_settings_uses_staged_config_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    job_path = Path(__file__).resolve().parents[2] / "configs/jobs/ffnn/ffnn.toml"
    job = load_job(job_path)
    staged_job = tmp_path / "staged-job.toml"
    staged_data = tmp_path / "staged-data.toml"
    current_job = tmp_path / "current-job.toml"
    current_data = tmp_path / "current-data.toml"
    split_file = tmp_path / "split.json"
    checkpoint = tmp_path / "model.ckpt"
    _write_split(split_file)
    checkpoint.write_bytes(b"checkpoint")
    for path in (staged_job, staged_data, current_job, current_data):
        path.write_text("", encoding="utf-8")
    load_assignment = MagicMock(
        return_value=SimpleNamespace(workspace=SimpleNamespace(data_dir=tmp_path))
    )
    monkeypatch.setattr(
        "neuralls.composition.assignments.evaluation.load_assignment",
        load_assignment,
    )
    monkeypatch.setattr(
        "neuralls.composition.assignments.evaluation.load_experiment_job",
        lambda path, settings: job,
    )
    np.save(tmp_path / "rhs.npy", np.zeros((4, 3)))
    np.save(tmp_path / "solutions.npy", np.ones((4, 3)))
    contract = default_training_dataset_contract()
    features = [
        entry_from_path(
            tmp_path / "rhs.npy",
            name=contract.primary_input_name,
            model_input=True,
            role="feature",
        )
    ]
    targets = [
        entry_from_path(
            tmp_path / "solutions.npy",
            name=contract.target_name,
            model_input=False,
            role="target",
        )
    ]
    monkeypatch.setattr(
        "neuralls.composition.assignments.evaluation._load_and_prepare_data",
        lambda settings, workspace, contract: (None, features, targets),
    )
    context = EvaluationAssignmentContext(
        client=MagicMock(),
        training_run_id="train-1",
        artifacts=TrainingEvaluationArtifacts(
            checkpoint_path=checkpoint,
            split_file=split_file,
            config_artifacts=TrainingConfigArtifacts(config_dir=None),
        ),
        config_paths=EvaluationConfigPaths(
            job_config_path=current_job,
            data_config_path=current_data,
            staged_job_config_path=staged_job,
            staged_data_config_path=staged_data,
        ),
        dataset_display_name="Dataset",
        job_display_name="Job",
    )

    _materialize_inference_settings(
        cfg=CaseConfig(),
        settings=MagicMock(),
        assignment=AssignmentEntry(id="a1", dataset="d1", job="j1"),
        output_root=tmp_path,
        case_config_path=tmp_path / "case.toml",
        context=context,
    )

    assert load_assignment.call_args.kwargs["job_config_path"] == staged_job
    assert load_assignment.call_args.kwargs["data_config_path"] == staged_data
