"""Reproducibility utilities for batch training workflows.

This module logs a reproducible configuration snapshot to the batch MLflow run,
making comparisons replayable from the MLflow UI without any custom directory
management.

The snapshot consists of:
- ``snapshot/experiments.toml``: updated experiments config with resolved checkpoint paths.
- ``snapshot/models/<id>.toml``: copies of referenced model configs.
- ``snapshot/datasets/<id>.toml``: copies of referenced dataset configs.

All files are logged as MLflow artifacts on the batch run, scoped under a
``snapshot/`` prefix so they don't clutter the artifact root.
"""

from __future__ import annotations

import copy
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import tomli_w
from loguru import logger
from mlflow.tracking import MlflowClient

from neuralls.workflows.multi_training import BatchResult


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load and parse a TOML configuration file.

    Args:
        config_path: Path to the TOML file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _create_snapshot_config(
    original_config: dict[str, Any],
    batch_result: BatchResult,
    new_output_dir: Path,
) -> dict[str, Any]:
    """Create a modified configuration dictionary with updated paths.

    Args:
        original_config: The original configuration dictionary.
        batch_result: The results of the batch training run.
        new_output_dir: The explicit output directory for future runs using this config.

    Returns:
        A new configuration dictionary with:
        - ``output_dir`` set to ``new_output_dir``.
        - Updated ``checkpoint_path`` for each successful experiment.
    """
    snapshot = copy.deepcopy(original_config)

    snapshot["output_dir"] = str(new_output_dir)

    new_checkpoints = {
        res.experiment_id: str(res.checkpoint_path.resolve())
        for res in batch_result.results
        if res.checkpoint_path and res.checkpoint_path.exists()
    }

    if "experiment" in snapshot:
        for exp in snapshot["experiment"]:
            exp_id = exp.get("id")
            if exp_id in new_checkpoints:
                exp["checkpoint_path"] = new_checkpoints[exp_id]

    return snapshot


def capture_batch_context(
    batch_result: BatchResult,
    original_config_path: Path,
) -> None:
    """Log a reproducible configuration snapshot to the batch MLflow run.

    Writes the following artifacts to the batch run under a ``snapshot/`` prefix:
    - ``snapshot/experiments.toml``: config snapshot with resolved checkpoint paths.
    - ``snapshot/models/<id>.toml``: copies of referenced model configs.
    - ``snapshot/datasets/<id>.toml``: copies of referenced dataset configs.

    All I/O is done via ``MlflowClient`` so no custom directories are created.

    Args:
        batch_result: The result object from the batch training workflow.
        original_config_path: Path to the original experiments TOML file.
    """
    try:
        original_config = _load_config(original_config_path)
    except Exception as e:
        logger.error(f"Failed to load original config from {original_config_path}: {e}")
        return

    run_id = batch_result.comparison_run.mlflow_run_id
    tracking_uri = batch_result.comparison_run.tracking_uri

    client = MlflowClient(tracking_uri=tracking_uri)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)

        # Create snapshot config (output_dir placeholder; checkpoints are the key info)
        snapshot = _create_snapshot_config(
            original_config,
            batch_result,
            new_output_dir=tmp_root / "output",
        )

        # Write snapshot TOML to temp dir
        snapshot_toml = tmp_root / "experiments.toml"
        try:
            with open(snapshot_toml, "wb") as f:
                tomli_w.dump(snapshot, f)
        except Exception as e:
            logger.error(f"Failed to write snapshot config: {e}")
            return

        # Log snapshot TOML to MLflow
        try:
            client.log_artifact(run_id, str(snapshot_toml), artifact_path="snapshot")
        except Exception as e:
            logger.error(f"Failed to log snapshot config to MLflow: {e}")
            return

        # Copy and log referenced model/dataset configs
        source_root = original_config_path.parent
        _log_referenced_configs(client, run_id, original_config, source_root, tmp_root)

    logger.info(f"Logged reproducible batch context to MLflow run {run_id} (snapshot/)")


def _log_referenced_configs(
    client: Any,
    run_id: str,
    config: dict[str, Any],
    source_root: Path,
    tmp_root: Path,
) -> None:
    """Copy referenced model/dataset configs to temp dir and log to MLflow.

    Args:
        client: MlflowClient instance.
        run_id: MLflow run ID to log artifacts to.
        config: Parsed experiments configuration.
        source_root: Directory containing the original ``experiments.toml``.
        tmp_root: Temporary directory for staging files.
    """
    if "experiment" not in config:
        return

    (tmp_root / "models").mkdir(exist_ok=True)
    (tmp_root / "datasets").mkdir(exist_ok=True)

    logged_models: set[str] = set()
    logged_datasets: set[str] = set()

    for exp in config["experiment"]:
        model_id = exp.get("model")
        dataset_id = exp.get("dataset")

        if model_id and model_id not in logged_models:
            src = source_root / "models" / f"{model_id}.toml"
            if src.exists():
                dst = tmp_root / "models" / f"{model_id}.toml"
                dst.write_bytes(src.read_bytes())
                try:
                    client.log_artifact(run_id, str(dst), artifact_path="snapshot/models")
                    logged_models.add(model_id)
                except Exception as e:
                    logger.warning(f"Could not log model config {model_id}: {e}")
            else:
                logger.warning(f"Referenced model config not found: {src}")

        if dataset_id and dataset_id not in logged_datasets:
            src = source_root / "datasets" / f"{dataset_id}.toml"
            if src.exists():
                dst = tmp_root / "datasets" / f"{dataset_id}.toml"
                dst.write_bytes(src.read_bytes())
                try:
                    client.log_artifact(run_id, str(dst), artifact_path="snapshot/datasets")
                    logged_datasets.add(dataset_id)
                except Exception as e:
                    logger.warning(f"Could not log dataset config {dataset_id}: {e}")
            else:
                logger.warning(f"Referenced dataset config not found: {src}")
