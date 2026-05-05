"""Integration tests for explicit settings and case-root precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.platform.config.settings import get_settings, load_case_settings


def _write_case_config(path: Path, *, raw_dir: str, processed_dir: str, output_dir: str) -> None:
    path.write_text(
        "\n".join(
            [
                f'raw_dir = "{raw_dir}"',
                f'processed_dir = "{processed_dir}"',
                f'output_dir = "{output_dir}"',
            ]
        ),
        encoding="utf-8",
    )


def test_precedence_env_var_over_env_file_over_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Process env beats env file, which beats case-root values."""
    for key in ("NEURALLS_RAW_DIR", "NEURALLS_PROCESSED_DIR", "NEURALLS_OUTPUT_DIR"):
        monkeypatch.delenv(key, raising=False)
    case_config = tmp_path / "case.toml"
    _write_case_config(
        case_config,
        raw_dir="./case-raw",
        processed_dir="./case-processed",
        output_dir="./case-output",
    )
    env_file = tmp_path / "case.env"
    env_file.write_text(
        "NEURALLS_RAW_DIR=/env-file/raw\n"
        "NEURALLS_PROCESSED_DIR=/env-file/processed\n"
        "NEURALLS_OUTPUT_DIR=/env-file/output\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NEURALLS_OUTPUT_DIR", str(tmp_path / "env-output"))

    settings = load_case_settings(case_config, env_file)

    assert settings.raw_dir == Path("/env-file/raw").resolve()
    assert settings.processed_dir == Path("/env-file/processed").resolve()
    assert settings.output_dir == (tmp_path / "env-output").resolve()


def test_precedence_env_file_over_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit env file beats case-root values when process env is absent."""
    for key in ("NEURALLS_RAW_DIR", "NEURALLS_PROCESSED_DIR", "NEURALLS_OUTPUT_DIR"):
        monkeypatch.delenv(key, raising=False)
    case_config = tmp_path / "case.toml"
    _write_case_config(
        case_config,
        raw_dir="./case-raw",
        processed_dir="./case-processed",
        output_dir="./case-output",
    )
    env_file = tmp_path / "case.env"
    env_file.write_text(
        "NEURALLS_RAW_DIR=/env-file/raw\n"
        "NEURALLS_PROCESSED_DIR=/env-file/processed\n"
        "NEURALLS_OUTPUT_DIR=/env-file/output\n",
        encoding="utf-8",
    )

    settings = load_case_settings(case_config, env_file)

    assert settings.raw_dir == Path("/env-file/raw").resolve()
    assert settings.processed_dir == Path("/env-file/processed").resolve()
    assert settings.output_dir == Path("/env-file/output").resolve()


def test_no_env_files_required_env_vars_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Env-only loading still works when no env file is provided."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    output = tmp_path / "output"
    for key, value in {
        "NEURALLS_RAW_DIR": raw,
        "NEURALLS_PROCESSED_DIR": processed,
        "NEURALLS_OUTPUT_DIR": output,
    }.items():
        monkeypatch.setenv(key, str(value))

    settings = get_settings()

    assert settings.raw_dir == raw.resolve()
    assert settings.processed_dir == processed.resolve()
    assert settings.output_dir == output.resolve()


def test_get_settings_does_not_scan_cwd_env_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ambient cwd .env files are ignored by design."""
    for key in ("NEURALLS_RAW_DIR", "NEURALLS_PROCESSED_DIR", "NEURALLS_OUTPUT_DIR"):
        monkeypatch.delenv(key, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(
        "NEURALLS_RAW_DIR=/cwd/raw\n"
        "NEURALLS_PROCESSED_DIR=/cwd/processed\n"
        "NEURALLS_OUTPUT_DIR=/cwd/output\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    with pytest.raises(Exception):
        get_settings()
