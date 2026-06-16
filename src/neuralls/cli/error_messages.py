"""Shared CLI error rendering helpers."""

from __future__ import annotations


def format_cli_error(command_context: str, exc: Exception) -> str:
    """Render a concise, typed CLI error message."""
    return f"{command_context} [{type(exc).__name__}]: {exc}"
