"""Utilities for recovering train-test split artifacts from MLflow runs.

dlkit logs split indices as JSON under ``splits/`` in the run's artifact store.
These helpers support inspection and debugging of training artifacts without
implying any comparison-time sample-selection policy.
"""

from __future__ import annotations

from pathlib import Path

from dlkit.infrastructure.io.index import load_split_indices
from dlkit.infrastructure.types.split import IndexSplit
from mlflow.tracking import MlflowClient


def download_split_indices(
    run_id: str,
    *,
    client: MlflowClient,
    dst_dir: Path,
) -> IndexSplit:
    """Download and parse the train-test split artifact from an MLflow run.

    dlkit stores split files under ``artifacts/splits/{session}_{n}_split.json``
    in the run. This function downloads the ``splits/`` directory, locates the
    first JSON file, and parses it into an ``IndexSplit``.

    Args:
        run_id: MLflow run ID from the training run.
        client: Configured ``MlflowClient`` pointing at the tracking server.
        dst_dir: Local directory to download artifacts into.

    Returns:
        ``IndexSplit`` with ``train``, ``validation``, ``test`` index tuples.

    Raises:
        StopIteration: If no split JSON file is found in the downloaded artifact.
        FileNotFoundError: If the ``splits/`` artifact directory does not exist.
    """
    split_dir = Path(client.download_artifacts(run_id, "splits", str(dst_dir)))
    split_file = next(split_dir.glob("*.json"))
    return load_split_indices(split_file)


__all__ = ["download_split_indices"]
