"""Tests for eval-only workflow assembly."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dlkit.infrastructure.config.job_config import InferenceJobConfig, TrainingJobConfig

from neuralls.composition.assignments.evaluation import (
    EvaluationAssignmentContext,
    EvaluationBatchResult,
    EvaluationConfigPaths,
    EvaluationRunResult,
    _materialize_inference_settings,
    _as_inference_job,
    _with_eval_runtime_dataset,
    eval_assignment,
    eval_batch,
    write_eval_metric_report,
)
from neuralls.composition.assignments.runtime_dataset_contract import (
    default_training_dataset_contract,
)
from neuralls.platform.config.dataset_entries import entry_from_path
from neuralls.platform.config.loaders import load_case_config
from neuralls.platform.config.models.experiments import AssignmentEntry, CaseConfig
from neuralls.platform.config.resolution import build_sqlite_tracking_uri
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


def test_eval_materialization_injects_split_file_and_array_datamodule(
    tmp_path: Path, minimal_training_job: TrainingJobConfig
) -> None:
    job = minimal_training_job
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
        tracking_uri=f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}",
        parent_run_id="parent-run-1",
    )

    with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
        client = mock_client_cls.return_value
        plotted = write_eval_metric_report(batch, metric="rmse")

    assert plotted is True
    logged_paths = {Path(call.args[1]).name for call in client.log_artifact.call_args_list}
    assert logged_paths == {"batch_metric_rmse.png", "batch_eval_labels.json"}
    assert all(call.args[0] == "parent-run-1" for call in client.log_artifact.call_args_list)


def test_materialize_inference_settings_uses_staged_config_paths(
    monkeypatch,
    tmp_path: Path,
    minimal_training_job: TrainingJobConfig,
) -> None:
    job = minimal_training_job
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


@pytest.fixture
def two_assignment_case_config(tmp_path: Path) -> Path:
    """Case config TOML with two eval assignments, for partial-failure tests."""
    path = tmp_path / "case.toml"
    path.write_text(
        "\n".join(
            [
                "[mlflow]",
                f'tracking_uri = "{build_sqlite_tracking_uri(tmp_path / "mlruns" / "mlflow.db")}"',
                "",
                "[[jobs]]",
                'id = "ffnn"',
                'path = "jobs/ffnn.toml"',
                "",
                "[[datasets]]",
                'id = "test-solutions"',
                'path = "datasets/test-solutions.toml"',
                "",
                "[[assignments]]",
                'id = "assign-ok"',
                'job = "ffnn"',
                'dataset = "test-solutions"',
                "",
                "[[assignments]]",
                'id = "assign-fail"',
                'job = "ffnn"',
                'dataset = "test-solutions"',
                "",
            ]
        )
    )
    (tmp_path / "jobs").mkdir()
    (tmp_path / "jobs" / "ffnn.toml").write_text("")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "test-solutions.toml").write_text('id = "test-solutions"\n')
    return path


def test_eval_batch_continues_after_one_assignment_fails(
    two_assignment_case_config: Path,
    tmp_path: Path,
    neuralls_settings,
) -> None:
    """One failing assignment must not discard the rest of the batch's results.

    Mirrors train_batch's resilience (multi_training.py): a per-assignment
    failure is logged and skipped, not allowed to abort the whole batch and
    lose already-completed reporting.
    """
    cfg = load_case_config(two_assignment_case_config, neuralls_settings)
    ok_result = EvaluationRunResult(
        label="1",
        assignment_id="assign-ok",
        assignment_display_name="assign-ok",
        training_run_id="train-1",
        evaluation_run_id="eval-1",
        metrics={"mae": 0.1},
        split_file=tmp_path / "split.json",
        figures_dir=tmp_path / "figures",
    )

    def _fake_eval_assignment(*, assignment, **kwargs):
        if assignment.id == "assign-fail":
            raise RuntimeError("checkpoint missing")
        return ok_result

    with (
        patch(
            "neuralls.composition.assignments.evaluation.eval_assignment",
            side_effect=_fake_eval_assignment,
        ),
        patch(
            "neuralls.composition.tracking.session.create_session_parent_run",
            return_value="parent-run-1",
        ),
        patch("neuralls.composition.tracking.session.finalize_session_parent_run"),
    ):
        batch = eval_batch(
            cfg=cfg,
            configs_dir=two_assignment_case_config.parent,
            settings=neuralls_settings,
            case_config_path=two_assignment_case_config,
        )

    assert [result.assignment_id for result in batch.results] == ["assign-ok"]

    with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
        client = mock_client_cls.return_value
        plotted = write_eval_metric_report(batch, metric="mae")

    assert plotted is True
    logged_paths = {Path(call.args[1]).name for call in client.log_artifact.call_args_list}
    assert logged_paths == {"batch_metric_mae.png", "batch_eval_labels.json"}


def test_eval_assignment_passes_parent_link_hooks(
    monkeypatch,
    tmp_path: Path,
    neuralls_settings,
) -> None:
    """eval_assignment wires build_parent_link_hooks into evaluate_fn — the
    same helper train_model() uses — so the child run is nested under its
    session parent atomically at creation, not via a post-hoc tag set after
    evaluate_fn() has already closed the run.
    """
    mock_client = MagicMock()
    context = EvaluationAssignmentContext(
        client=mock_client,
        training_run_id="train-1",
        artifacts=TrainingEvaluationArtifacts(
            checkpoint_path=tmp_path / "model.ckpt",
            split_file=tmp_path / "split.json",
            config_artifacts=TrainingConfigArtifacts(config_dir=None),
        ),
        config_paths=EvaluationConfigPaths(
            job_config_path=tmp_path / "job.toml",
            data_config_path=tmp_path / "data.toml",
            staged_job_config_path=tmp_path / "job.toml",
            staged_data_config_path=tmp_path / "data.toml",
        ),
        dataset_display_name=None,
        job_display_name=None,
    )
    monkeypatch.setattr(
        "neuralls.composition.assignments.evaluation._build_eval_context",
        lambda **kwargs: context,
    )
    monkeypatch.setattr(
        "neuralls.composition.assignments.evaluation._materialize_inference_settings",
        lambda **kwargs: SimpleNamespace(data=None),
    )

    sentinel_hooks = object()
    build_hooks_calls: list[str | None] = []

    def fake_build_parent_link_hooks(parent_run_id):
        build_hooks_calls.append(parent_run_id)
        return sentinel_hooks

    monkeypatch.setattr(
        "neuralls.composition.assignments.evaluation.build_parent_link_hooks",
        fake_build_parent_link_hooks,
    )

    captured_kwargs: dict[str, object] = {}

    def fake_evaluate_fn(settings, **kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(mlflow_run_id="run-99", metrics={"mae": 0.1}, figures={})

    eval_assignment(
        cfg=CaseConfig(),
        configs_dir=tmp_path,
        settings=neuralls_settings,
        assignment=AssignmentEntry(id="a1", dataset="d1", job="j1"),
        output_root=tmp_path,
        case_config_path=tmp_path / "case.toml",
        label="1",
        tracking_uri="sqlite:///" + str(tmp_path / "mlflow.db"),
        parent_run_id="parent-1",
        evaluate_fn=fake_evaluate_fn,
        mlflow_client_factory=lambda **kwargs: MagicMock(),
    )

    assert build_hooks_calls == ["parent-1"]
    assert captured_kwargs["hooks"] is sentinel_hooks
    tagged_keys = [call.args[1] for call in mock_client.set_tag.call_args_list]
    assert "mlflow.parentRunId" not in tagged_keys
