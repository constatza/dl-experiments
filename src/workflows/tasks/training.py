"""Model training task."""

from __future__ import annotations

from pathlib import Path

from prefect import task, get_run_logger
from prefect.cache_policies import INPUTS

from src.cli.training import train_model
from src.system_loading import get_latest_checkpoint
from src.validation import validate_data_exists
from src.workflows.utils.paths import extract_model_name, prepare_experiment_outputs


@task(persist_result=True, cache_policy=INPUTS)
def train_model_task(
    model_config_path: str,
    data_config_path: str,
    data_dir: Path,
    data_hash: str,
    output_root: Path,
    checkpoint_state: str = "",
    model_config_version: str = "",
    data_config_version: str = "",
    src_hash: str = "",
    force: bool = False,
) -> Path:
    """Train model with resolved config paths.
    
    Args:
        model_config_path: Path to model config TOML.
        data_config_path: Path to data config TOML.
        data_dir: Directory containing processed data.
        data_hash: Hash of data content (for cache key).
        output_root: Root directory for outputs.
        checkpoint_state: Hash of checkpoint directory state (for cache key).
        model_config_version: Hash of model config content.
        data_config_version: Hash of data config content.
        src_hash: Hash of source code.
        force: If True, force re-training.
        
    Returns:
        Path to the best checkpoint.
    """
    logger = get_run_logger()
    model_name = extract_model_name(model_config_path)
    data_config_name = Path(data_config_path).stem
    logger.info("Training %s on %s", model_name, data_config_name)

    # Unused args for cache key
    _ = src_hash
    _ = data_hash
    _ = checkpoint_state

    output_dirs = prepare_experiment_outputs(model_name, data_config_name, output_root)

    if not force:
        cached_checkpoint = get_latest_checkpoint(output_dirs["checkpoints_dir"])
        if cached_checkpoint and cached_checkpoint.exists():
            logger.info("Checkpoint already exists, skipping training.")
            return cached_checkpoint

    validate_data_exists(data_dir, ["normalized.npz"])

    logger.info("Starting training...")
    checkpoint_path = train_model(
        config_path=model_config_path,
        data_config_path=data_config_path,
        output_dir=output_dirs["experiment_dir"],
        session_name=model_name,
    )

    if output_dirs["checkpoints_dir"] not in checkpoint_path.parents:
        raise RuntimeError("Checkpoint written outside checkpoints directory.")

    logger.info("Checkpoint saved to %s", checkpoint_path)
    return checkpoint_path
