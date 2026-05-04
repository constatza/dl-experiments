"""DLKit-backed path and local URI helpers for repo-specific policy.

This module keeps neuralls-specific policy thin while delegating low-level
path and local URI handling to DLKit's path resolver and URI utilities.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dlkit.infrastructure.io import PathResolver, url_resolver


_GLOB_CHARS = frozenset("*?[]")


def expand_path_placeholders(value: str) -> str:
    """Expand environment variables and ``~`` in a path-like string."""
    return os.path.expanduser(os.path.expandvars(value))


def expand_nested_path_values(
    value: Any,
    *,
    path_keys: frozenset[str],
) -> Any:
    """Recursively expand env placeholders for known path-like config keys."""
    if isinstance(value, dict):
        return {
            key: (
                expand_path_placeholders(item)
                if isinstance(item, str) and key in path_keys
                else expand_nested_path_values(item, path_keys=path_keys)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [expand_nested_path_values(item, path_keys=path_keys) for item in value]
    return value


def resolve_local_path(
    value: Path | str,
    *,
    base_dir: Path,
) -> Path:
    """Resolve a local path or local URI against a base directory."""
    text = expand_path_placeholders(str(value))
    return url_resolver.resolve_local_uri(text, base_dir.resolve())


def resolve_optional_local_path(
    value: Path | str | None,
    *,
    base_dir: Path,
) -> Path | None:
    """Resolve an optional local path value when present."""
    if value is None:
        return None
    return resolve_local_path(value, base_dir=base_dir)


def resolve_root_path(
    *,
    default_root: Path,
    override: Path | str | None = None,
    base_dir: Path | None = None,
) -> Path:
    """Resolve a config-driven root path using DLKit's resolver semantics."""
    if override is not None:
        root_base = base_dir if base_dir is not None else Path.cwd()
        return resolve_local_path(override, base_dir=root_base)
    return PathResolver(config_root=default_root).resolve()


def resolve_path_expression(
    expr: str,
    *,
    base_dir: Path,
) -> str:
    """Resolve a path or glob expression against a config directory."""
    expanded = expand_path_placeholders(expr)
    if not any(char in expanded for char in _GLOB_CHARS):
        return str(resolve_local_path(expanded, base_dir=base_dir))

    pattern = Path(expanded)
    parent_expr = "." if str(pattern.parent) in {"", "."} else str(pattern.parent)
    parent = resolve_local_path(parent_expr, base_dir=base_dir)
    return str(parent / pattern.name)


def build_sqlite_uri(path: Path) -> str:
    """Build a normalized sqlite URI from a local database path."""
    return url_resolver.build_uri(path.resolve(), scheme="sqlite")
