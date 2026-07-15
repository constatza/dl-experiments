"""MLflow artifact downloads required for eval-only workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dlkit.infrastructure.io.index import load_split_indices
from mlflow.tracking import MlflowClient

from neuralls.platform.tracking.checkpoint_selection import find_single_checkpoint


@dataclass(frozen=True)
class TrainingConfigArtifacts:
    """Downloaded config artifacts from a completed training run."""

    config_dir: Path | None

    def find_named(self, fallback: Path) -> Path:
        """Return the downloaded config with the fallback basename when present."""
        if self.config_dir is None:
            return fallback
        candidate = self.config_dir / fallback.name
        return candidate if candidate.exists() else fallback


@dataclass(frozen=True)
class TrainingEvaluationArtifacts:
    """Downloaded artifacts required to evaluate one completed training run."""

    checkpoint_path: Path
    split_file: Path
    config_artifacts: TrainingConfigArtifacts


def download_training_checkpoint(
    client: MlflowClient,
    run_id: str,
    dst_dir: Path,
) -> Path:
    """Download training checkpoint artifacts and return the selected checkpoint."""
    checkpoint_root = Path(
        client.download_artifacts(
            run_id=run_id,
            path="checkpoints",
            dst_path=str(dst_dir),
        )
    )
    return find_single_checkpoint(checkpoint_root)


def download_training_split_file(
    client: MlflowClient,
    run_id: str,
    dst_dir: Path,
) -> Path:
    """Download and validate the single split JSON artifact for a training run."""
    split_dir = Path(
        client.download_artifacts(
            run_id=run_id,
            path="splits",
            dst_path=str(dst_dir),
        )
    )
    split_files = sorted(split_dir.glob("*.json"))
    if not split_files:
        raise FileNotFoundError(
            f"Training run '{run_id}' has no split JSON artifact under 'splits/'."
        )
    if len(split_files) > 1:
        joined = ", ".join(path.name for path in split_files)
        raise ValueError(
            f"Training run '{run_id}' has multiple split JSON artifacts under 'splits/': {joined}."
        )
    split_file = split_files[0]
    load_split_indices(split_file)
    return split_file


def download_training_config_artifacts(
    client: MlflowClient,
    run_id: str,
    dst_dir: Path,
) -> TrainingConfigArtifacts:
    """Download optional staged config artifacts from a training run."""
    if not client.list_artifacts(run_id, "config"):
        return TrainingConfigArtifacts(config_dir=None)
    config_dir = Path(
        client.download_artifacts(
            run_id=run_id,
            path="config",
            dst_path=str(dst_dir),
        )
    )
    return TrainingConfigArtifacts(config_dir=config_dir)


def download_training_evaluation_artifacts(
    client: MlflowClient,
    run_id: str,
    dst_dir: Path,
) -> TrainingEvaluationArtifacts:
    """Download all MLflow artifacts required for eval-only execution."""
    return TrainingEvaluationArtifacts(
        checkpoint_path=download_training_checkpoint(client, run_id, dst_dir),
        split_file=download_training_split_file(client, run_id, dst_dir),
        config_artifacts=download_training_config_artifacts(client, run_id, dst_dir),
    )
