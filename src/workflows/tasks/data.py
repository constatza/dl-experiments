"""Data generation and loading task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prefect import task
from prefect.cache_policies import INPUTS

from src.cli.data import load_data_config
from src.generation import process_config
from src.workflows.utils.artifacts import create_markdown_artifact
from src.workflows.utils.hashing import compute_data_files_hash


@task(
    persist_result=True,
    cache_policy=INPUTS,
    task_run_name="get-or-generate-{data_config_path}",
    retries=1,
    retry_delay_seconds=5,
)
def get_or_generate_data_task(
    data_config_path: str,
    data_config_version: str = "",
    src_hash: str = "",
    data_state: str = "",
    force: bool = False,
) -> dict[str, Any]:
    """Get existing data or generate if missing (idempotent).
    
    Args:
        data_config_path: Path to the data configuration file.
        data_config_version: Hash of the config file (for cache invalidation).
        src_hash: Hash of the source code (for cache invalidation).
        data_state: Hash of the existing data directory (for cache invalidation).
        force: If True, ignore cache and regenerate.
        
    Returns:
        Dictionary containing 'data_dir' (Path) and 'data_hash' (str).
    """
    config_path = Path(data_config_path)
    
    # Unused arguments are kept for Prefect cache key generation
    _ = data_config_version
    _ = src_hash
    _ = data_state

    cfg = load_data_config(config_path)
    output_path = process_config(cfg, config_path=config_path)
    print(f"\nData ready at: {output_path}")

    required_files = ["normalized.npz"]
    missing_files = []
    for filename in required_files:
        filepath = output_path / filename
        if not filepath.exists():
            missing_files.append(str(filepath))
        elif filepath.stat().st_size == 0:
            missing_files.append(f"{filepath} (empty)")
    if missing_files:
        files_str = "\n  - ".join(missing_files)
        raise RuntimeError(
            f"Data generation completed but required files are missing or empty:\n  - {files_str}"
        )

    data_hash = compute_data_files_hash(output_path)
    print(f"Data content hash: {data_hash[:12]}...")
    
    create_markdown_artifact(
        key=f"data-status-{config_path.stem}",
        markdown=(
            f"# Data Ready\n\n"
            f"**Config**: `{config_path.name}`\n\n"
            f"**Location**: `{output_path}`\n\n"
            f"**Data Hash**: `{data_hash[:12]}...`"
        ),
        description=f"Data status for {config_path.stem}",
    )
    return {"data_dir": output_path, "data_hash": data_hash}
