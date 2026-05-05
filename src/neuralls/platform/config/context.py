"""ConfigContext and path-expansion helpers used by all Pydantic config validators."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from neuralls.platform.config.settings import NeurallsSettings

_GRAPH_CG_RE = re.compile(r"\$\{GRAPH_CG_[^}]+\}")
_NEURALLS_RE = re.compile(r"\$\{NEURALLS_([^}]+)\}")
_GLOB_CHARS = frozenset("*?[]")


@dataclass(frozen=True)
class ConfigContext:
    """Immutable context passed to Pydantic model validators via model_validate(context=...).

    Carries the path of the config file being loaded and the resolved runtime settings
    so validators can expand ${NEURALLS_*} placeholders and anchor relative paths.

    Attributes:
        config_path: Absolute path to the TOML file being loaded.
        settings: Resolved NeurallsSettings for this invocation.
    """

    config_path: Path
    settings: NeurallsSettings

    @property
    def config_dir(self) -> Path:
        """Parent directory of config_path — used to anchor relative paths."""
        return self.config_path.parent

    def as_pydantic_context(self) -> dict[str, object]:
        """Serialize to the dict format expected by Pydantic v2 model_validate(context=...)."""
        return {
            "config_path": self.config_path,
            "config_dir": self.config_dir,
            "settings": self.settings,
        }

    @staticmethod
    def from_pydantic_context(ctx: dict[str, object]) -> ConfigContext:
        """Reconstruct from the dict produced by as_pydantic_context()."""
        return ConfigContext(
            config_path=cast(Path, ctx["config_path"]),
            settings=cast(NeurallsSettings, ctx["settings"]),
        )


def expand_config_path(value: str, ctx: ConfigContext) -> str:
    """Expand ${NEURALLS_*} placeholders and anchor relative paths to config dir."""
    if m := _GRAPH_CG_RE.search(value):
        raise ValueError(
            f"Stale placeholder '{m.group()}' in config. "
            "Rename GRAPH_CG_* → NEURALLS_* (see docs/env-reference.md)."
        )

    def _sub(m: re.Match[str]) -> str:
        attr = m.group(1).lower()
        val = getattr(ctx.settings, attr, None)
        if val is None:
            raise ValueError(
                f"Unknown placeholder '${{NEURALLS_{m.group(1)}}}'. "
                "Available: PROCESSED_DIR, OUTPUT_DIR."
            )
        return str(val)

    expanded = _NEURALLS_RE.sub(_sub, value)
    expanded = os.path.expanduser(expanded)
    path = Path(expanded)
    return str((path if path.is_absolute() else ctx.config_dir / path).resolve())


def expand_config_glob(value: str, ctx: ConfigContext) -> str:
    """Like expand_config_path but preserves glob wildcards in the filename."""
    for i, ch in enumerate(value):
        if ch in _GLOB_CHARS:
            prefix = value[:i]
            suffix = value[i:]
            if not prefix:
                return str(ctx.config_dir / suffix)
            resolved_prefix = (
                expand_config_path(prefix, ctx) if prefix.strip() else str(ctx.config_dir)
            )
            if prefix.endswith(("/", "\\")):
                return str(Path(resolved_prefix) / suffix)
            return f"{resolved_prefix}{suffix}"
    return expand_config_path(value, ctx)
