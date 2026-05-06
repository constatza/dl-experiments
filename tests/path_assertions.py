"""Shared test assertions for normalized local path values."""

from __future__ import annotations

from pathlib import Path


def assert_local_path_eq(actual: str, expected: Path) -> None:
    """Compare a normalized local path string against an expected path semantically."""
    assert Path(actual) == expected
