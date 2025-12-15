"""Artifact generation utilities with strict validation.

This module provides wrappers around Prefect artifact generation functions
to ensure compliance with key validation rules (e.g., regex constraints).
"""

from __future__ import annotations

import re
from prefect.artifacts import create_markdown_artifact as prefect_create_markdown


def sanitize_artifact_key(key: str) -> str:
    """Sanitize artifact key to match Prefect's validation rules.
    
    Prefect requires artifact keys to match: ^[a-z0-9-]+$
    
    Args:
        key: Raw key string.
        
    Returns:
        Sanitized key string (lowercase, alphanumeric, dashes only).
    """
    # Lowercase and replace non-alphanumeric chars with dashes
    sanitized = re.sub(r"[^a-z0-9]+", "-", key.lower())
    # Strip leading/trailing dashes
    return sanitized.strip("-")


def create_markdown_artifact(
    key: str, markdown: str, description: str | None = None
) -> None:
    """Wrapper for prefect.artifacts.create_markdown_artifact with key sanitization.
    
    Args:
        key: Unique identifier for the artifact. Will be sanitized.
        markdown: Markdown content.
        description: Optional description.
    """
    safe_key = sanitize_artifact_key(key)
    prefect_create_markdown(key=safe_key, markdown=markdown, description=description)
