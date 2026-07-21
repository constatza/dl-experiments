"""Tests for eval-only MLflow artifact recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuralls.platform.tracking.evaluation_artifacts import (
    CheckpointArtifactError,
    CorruptCheckpointArtifactError,
    MissingCheckpointArtifactError,
    download_training_checkpoint,
    download_training_config_artifacts,
    download_training_split_file,
)


class FakeArtifactClient:
    """Minimal MLflowClient double for artifact download tests."""

    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir

    def download_artifacts(self, run_id: str, path: str, dst_path: str) -> str:
        del run_id, dst_path
        return str(self.artifact_dir / path)

    def list_artifacts(self, run_id: str, path: str):
        del run_id
        target = self.artifact_dir / path
        if not target.exists():
            return []
        return [object()]


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


def test_download_training_split_file_validates_single_json(tmp_path: Path) -> None:
    split_dir = tmp_path / "artifacts" / "splits"
    split_dir.mkdir(parents=True)
    split_file = split_dir / "run_4_split.json"
    _write_split(split_file)
    client = FakeArtifactClient(tmp_path / "artifacts")

    result = download_training_split_file(client, "run-1", tmp_path / "downloads")  # type: ignore[arg-type]

    assert result == split_file


def test_download_training_split_file_rejects_missing_json(tmp_path: Path) -> None:
    (tmp_path / "artifacts" / "splits").mkdir(parents=True)
    client = FakeArtifactClient(tmp_path / "artifacts")

    with pytest.raises(FileNotFoundError, match="no split JSON artifact"):
        download_training_split_file(client, "run-1", tmp_path / "downloads")  # type: ignore[arg-type]


def test_download_training_split_file_rejects_multiple_json_files(tmp_path: Path) -> None:
    split_dir = tmp_path / "artifacts" / "splits"
    split_dir.mkdir(parents=True)
    _write_split(split_dir / "first.json")
    _write_split(split_dir / "second.json")
    client = FakeArtifactClient(tmp_path / "artifacts")

    with pytest.raises(ValueError, match="multiple split JSON artifacts"):
        download_training_split_file(client, "run-1", tmp_path / "downloads")  # type: ignore[arg-type]


def test_download_training_checkpoint_raises_when_no_checkpoint_artifact(tmp_path: Path) -> None:
    """A FINISHED run with no 'checkpoints/' artifact (e.g. a stale pre-fix run) must
    fail with a clear, actionable error instead of MLflow's generic download exception.
    """
    (tmp_path / "artifacts").mkdir()
    client = FakeArtifactClient(tmp_path / "artifacts")

    with pytest.raises(MissingCheckpointArtifactError, match="run-1.*assignment-1"):
        download_training_checkpoint(
            client,  # type: ignore[arg-type]
            "run-1",
            tmp_path / "downloads",
            assignment_id="assignment-1",
        )


def test_download_training_checkpoint_returns_checkpoint_when_present(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "artifacts" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_file = checkpoint_dir / "best.ckpt"
    checkpoint_file.write_bytes(b"weights")
    client = FakeArtifactClient(tmp_path / "artifacts")

    result = download_training_checkpoint(
        client,  # type: ignore[arg-type]
        "run-1",
        tmp_path / "downloads",
        assignment_id="assignment-1",
    )

    assert result == checkpoint_file


def test_download_training_checkpoint_raises_corrupt_error_when_download_fails(
    tmp_path: Path,
) -> None:
    """A run whose 'checkpoints/' artifact IS listed but fails to download must raise
    CorruptCheckpointArtifactError (not the generic MLflow error), distinct from a
    genuinely missing checkpoint, with the original MLflow exception chained.
    """
    checkpoint_dir = tmp_path / "artifacts" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best.ckpt").write_bytes(b"weights")

    original_error = RuntimeError("please ensure that the path is correct")

    class BrokenDownloadClient(FakeArtifactClient):
        def download_artifacts(self, run_id: str, path: str, dst_path: str) -> str:
            del run_id, path, dst_path
            raise original_error

    client = BrokenDownloadClient(tmp_path / "artifacts")

    with pytest.raises(CorruptCheckpointArtifactError, match="run-1.*assignment-1") as excinfo:
        download_training_checkpoint(
            client,  # type: ignore[arg-type]
            "run-1",
            tmp_path / "downloads",
            assignment_id="assignment-1",
        )

    assert excinfo.value.__cause__ is original_error


def test_checkpoint_artifact_errors_share_common_base() -> None:
    """Both checkpoint-artifact exceptions must narrow to CheckpointArtifactError so
    callers can catch 'any checkpoint issue' without enumerating every subclass.
    """
    assert issubclass(MissingCheckpointArtifactError, CheckpointArtifactError)
    assert issubclass(CorruptCheckpointArtifactError, CheckpointArtifactError)


def test_download_training_config_artifacts_returns_none_when_absent(tmp_path: Path) -> None:
    client = FakeArtifactClient(tmp_path / "artifacts")

    result = download_training_config_artifacts(client, "run-1", tmp_path / "downloads")  # type: ignore[arg-type]

    assert result.config_dir is None


def test_download_training_config_artifacts_propagates_listing_errors(tmp_path: Path) -> None:
    class BrokenClient(FakeArtifactClient):
        def list_artifacts(self, run_id: str, path: str):
            del run_id, path
            raise RuntimeError("mlflow is unavailable")

    client = BrokenClient(tmp_path / "artifacts")

    with pytest.raises(RuntimeError, match="mlflow is unavailable"):
        download_training_config_artifacts(client, "run-1", tmp_path / "downloads")  # type: ignore[arg-type]
