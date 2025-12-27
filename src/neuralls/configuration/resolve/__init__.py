"""Configuration loaders package.

This package provides focused modules for configuration loading responsibilities:
- toml_loader: TOML parsing and Pydantic validation
- path_resolver: Path resolution and validation
- settings_builder: dlkit GeneralSettings construction
"""

from .path_resolver import (
    PathResolutionContext,
    build_path_context,
    resolve_config_path,
    resolve_output_root,
    resolve_processed_root,
    resolve_project_root,
)
from .settings_builder import (
    apply_datamodule_overrides,
    build_settings,
    inject_mlflow_paths,
    inject_workspace_paths,
)

__all__ = [
    # TOML Loader
    # Path Resolver
    "PathResolutionContext",
    "resolve_project_root",
    "resolve_processed_root",
    "resolve_output_root",
    "resolve_config_path",
    "build_path_context",
    # Settings Builder
    "apply_datamodule_overrides",
    "inject_workspace_paths",
    "inject_mlflow_paths",
    "build_settings",
]
