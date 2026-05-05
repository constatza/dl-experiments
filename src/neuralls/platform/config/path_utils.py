"""DLKit-backed local path and sqlite URI helpers."""

from __future__ import annotations

from pathlib import Path

from dlkit.infrastructure.io import url_resolver


def resolve_local_path(
    value: Path | str,
    *,
    base_dir: Path,
) -> Path:
    """Resolve a local path or local URI against a base directory."""
    return url_resolver.resolve_local_uri(str(value), base_dir.resolve())


def resolve_optional_local_path(
    value: Path | str | None,
    *,
    base_dir: Path,
) -> Path | None:
    """Resolve an optional local path value when present."""
    if value is None:
        return None
    return resolve_local_path(value, base_dir=base_dir)


def build_sqlite_uri(path: Path) -> str:
    """Build a normalized sqlite URI from a local database path."""
    return url_resolver.build_uri(path.resolve(), scheme="sqlite")
