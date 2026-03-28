"""Batch dataset generation workflow: generate all datasets from an experiments config.

Key Functions:
    - ``generate_batch()``: Iterates ``[[datasets]]`` in the experiments config and
      calls ``process_data_from_config()`` for each entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from neuralls.platform.config.models.experiments import ExperimentsConfig
from neuralls.composition.generation.process_data import process_data_from_config


@dataclass(frozen=True)
class GenerationResult:
    """Immutable result for a single generated dataset.

    Attributes:
        dataset_id: Dataset registry id (from ``[[datasets]]``).
        config_path: Resolved dataset config TOML path.
        output_dir: Directory written by ``process_data_from_config``.
    """

    dataset_id: str
    config_path: Path
    output_dir: Path


def generate_batch(
    cfg: ExperimentsConfig,
    configs_dir: Path,
) -> list[GenerationResult]:
    """Generate all datasets listed in the experiments config.

    Iterates ``cfg.datasets``, resolves each entry's path relative to
    ``configs_dir``, and calls ``process_data_from_config()`` for each one.

    Args:
        cfg: Validated experiments configuration.
        configs_dir: Parent directory of the experiments TOML (for resolving
            relative dataset config paths).

    Returns:
        List of ``GenerationResult`` in the same order as ``cfg.datasets``.
    """
    results: list[GenerationResult] = []
    for entry in cfg.datasets:
        config_path = (configs_dir / entry.path).resolve()
        output_dir = process_data_from_config(config_path)
        results.append(
            GenerationResult(dataset_id=entry.id, config_path=config_path, output_dir=output_dir)
        )
        logger.info(f"[{entry.id}] generated → {output_dir}")
    return results
