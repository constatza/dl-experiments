"""Lazy public API for neuralls I/O helpers.

This module intentionally avoids eager imports to prevent heavy import-time side
effects and circular dependencies (notably through DLKit config modules).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    # Filesystem
    "ensure_dir": ("neuralls.io.filesystem", "ensure_dir"),
    "sanitize_identifier": ("neuralls.io.filesystem", "sanitize_identifier"),
    "derive_model_identifier": ("neuralls.io.filesystem", "derive_model_identifier"),
    "parse_data_dir_name": ("neuralls.io.filesystem", "parse_data_dir_name"),
    "copy_file": ("neuralls.io.filesystem", "copy_file"),
    "get_file_size": ("neuralls.io.filesystem", "get_file_size"),
    "file_exists": ("neuralls.io.filesystem", "file_exists"),
    "get_relative_path": ("neuralls.io.filesystem", "get_relative_path"),
    "format_file_size": ("neuralls.io.filesystem", "format_file_size"),
    "save_text_file": ("neuralls.io.filesystem", "save_text_file"),
    "load_text_file": ("neuralls.io.filesystem", "load_text_file"),
    # Arrays
    "load_system_data": ("neuralls.io.arrays", "load_system_data"),
    "save_training_data": ("neuralls.io.arrays", "save_training_data"),
    "load_case_data": ("neuralls.io.arrays", "load_case_data"),
    "list_available_cases": ("neuralls.io.arrays", "list_available_cases"),
    "derive_checkpoint_path": ("neuralls.io.arrays", "derive_checkpoint_path"),
    # Datasets
    "load_dataset": ("neuralls.io.datasets", "load_dataset"),
    "load_numpy_array": ("neuralls.io.datasets", "load_numpy_array"),
    "save_numpy_array": ("neuralls.io.datasets", "save_numpy_array"),
    "has_comparison_split": ("neuralls.io.datasets", "has_comparison_split"),
    "list_available_variants": ("neuralls.io.datasets", "list_available_variants"),
    "DatasetFileTracker": ("neuralls.io.datasets", "DatasetFileTracker"),
    "file_tracker": ("neuralls.io.datasets", "file_tracker"),
    "track_files": ("neuralls.io.datasets", "track_files"),
    # Checkpoints
    "get_latest_checkpoint": ("neuralls.io.checkpoints", "get_latest_checkpoint"),
    "clean_directory": ("neuralls.io.checkpoints", "clean_directory"),
    # Base
    "load_matrix": ("neuralls.io.base", "load_matrix"),
    # Comparison
    "resolve_system_paths": ("neuralls.io.comparison", "resolve_system_paths"),
    "load_system_arrays": ("neuralls.io.comparison", "load_system_arrays"),
    # Config loading
    "load_model_config": ("neuralls.io.toml_loader", "load_model_config"),
    "load_data_config": ("neuralls.io.toml_loader", "load_data_config"),
    "load_comparison_config": ("neuralls.io.toml_loader", "load_comparison_config"),
    "load_raw_toml": ("neuralls.io.toml_loader", "load_raw_toml"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    """Resolve public attributes lazily from their source modules."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazily exported names for discovery/introspection."""
    return sorted(set(globals()) | set(__all__))
