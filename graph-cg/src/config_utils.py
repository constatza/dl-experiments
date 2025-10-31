"""Pure helpers for resolving data configuration paths.

These functions operate on configuration files without mutating state or
performing side effects beyond reading TOML sources. They intentionally mirror
the behaviour of :func:`src.unified_data_processing.process_config` so that
workflows, tests, and orchestration scripts can reason about dataset
locations without re-implementing parsing logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import tomllib

from .constants import ConfigKeys, ConfigSections
from .paths import ProjectRoots, FlowPaths, DataPaths, parse_flow_keys


def _coerce_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _load_raw_config(config_path: Path) -> Mapping[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "rb") as handle:
        return tomllib.load(handle)


def resolve_data_dir(data_config_path: Path | str) -> Path:
    """Return the processed data directory declared by a data config."""

    config_path = Path(data_config_path)
    raw_config = _load_raw_config(config_path)

    output_cfg = _coerce_mapping(raw_config.get(ConfigSections.OUTPUT, {}))
    flow_id, dataset_id = parse_flow_keys(raw_config, config_path=config_path)

    roots = ProjectRoots.from_overrides(
        project_root=output_cfg.get("project_root"),
        processed_root=output_cfg.get(ConfigKeys.PROCESSED_DIR),
        output_root=output_cfg.get("output_root"),
        figures_root=output_cfg.get("figures_root"),
    )
    flow_paths = FlowPaths(flow_id=flow_id, roots=roots)
    data_paths = DataPaths(flow=flow_paths, dataset_id=dataset_id)

    return data_paths.base_dir


def resolve_training_paths(data_dir: Path | str) -> dict[str, Path]:
    """Derive feature/target file paths inside a processed data directory."""

    base = Path(data_dir)
    return {
        "features_path": base / "rhs-samples.npy",
        "targets_path": base / "sol-samples.npy",
    }
