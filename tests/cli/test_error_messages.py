"""Tests for shared CLI error rendering."""

from __future__ import annotations

from pathlib import Path

from neuralls.cli.error_messages import format_cli_error


def test_format_cli_error_renders_context_and_type() -> None:
    message = format_cli_error("Error during batch generation", ValueError("bad config"))

    assert message == "Error during batch generation [ValueError]: bad config"


def test_format_cli_error_preserves_enriched_os_error_text(tmp_path: Path) -> None:
    matrix_store = tmp_path / "matrix.zarr"
    message = format_cli_error(
        "Error during data generation",
        PermissionError(
            f"Updating matrix.zarr store at {matrix_store} failed. "
            "PermissionError: denied, winerror=5, src=a.partial, dst=zarr.json"
        ),
    )

    assert message.startswith("Error during data generation [PermissionError]: ")
    assert "winerror=5" in message
    assert "dst=zarr.json" in message
