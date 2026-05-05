"""Tests for profile-based user config storage."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import neuralls.platform.config.profile as profile_module
from neuralls.platform.config.profile import (
    PROFILE_ENV_VAR,
    ProfileConfig,
    delete_profile,
    list_profiles,
    load_profile,
    save_profile,
)
from neuralls.platform.config.settings import load_case_settings


def _configure_profile_path(monkeypatch: pytest.MonkeyPatch, config_file: Path) -> None:
    monkeypatch.setattr(profile_module, "USER_CONFIG_DIR", config_file.parent)
    monkeypatch.setattr(profile_module, "USER_CONFIG_FILE", config_file)


def _profile(raw_root: Path, processed_root: Path, output_root: Path) -> ProfileConfig:
    return ProfileConfig(
        raw_dir=raw_root,
        processed_dir=processed_root,
        output_dir=output_root,
    )


def test_save_profile_creates_default_and_named_profiles(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    save_profile(
        "default",
        _profile(tmp_path / "raw", tmp_path / "processed", tmp_path / "output"),
        _config_file=config_file,
    )
    save_profile(
        "laptop",
        _profile(tmp_path / "raw2", tmp_path / "processed2", tmp_path / "output2"),
        _config_file=config_file,
    )

    assert config_file.exists()
    assert list_profiles(_config_file=config_file) == ["default", "laptop"]


def test_load_profile_none_returns_default(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    expected = _profile(tmp_path / "raw", tmp_path / "processed", tmp_path / "output")
    save_profile("default", expected, _config_file=config_file)

    assert load_profile(_config_file=config_file) == expected


def test_load_profile_named_returns_named_profile(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    expected = _profile(tmp_path / "raw", tmp_path / "processed", tmp_path / "output")
    save_profile("laptop", expected, _config_file=config_file)

    assert load_profile("laptop", _config_file=config_file) == expected


def test_load_profile_missing_named_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    save_profile(
        "default",
        _profile(tmp_path / "raw", tmp_path / "processed", tmp_path / "output"),
        _config_file=config_file,
    )

    with pytest.raises(ValueError, match="Profile 'laptop' not found"):
        load_profile("laptop", _config_file=config_file)


def test_load_profile_returns_none_when_config_file_missing(tmp_path: Path) -> None:
    assert load_profile(_config_file=tmp_path / "missing.toml") is None


def test_list_profiles_returns_default_then_named_profiles(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    save_profile(
        "default",
        _profile(tmp_path / "raw", tmp_path / "processed", tmp_path / "output"),
        _config_file=config_file,
    )
    save_profile(
        "laptop",
        _profile(tmp_path / "raw2", tmp_path / "processed2", tmp_path / "output2"),
        _config_file=config_file,
    )

    assert list_profiles(_config_file=config_file) == ["default", "laptop"]


def test_delete_profile_removes_named_and_refuses_default(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    save_profile(
        "default",
        _profile(tmp_path / "raw", tmp_path / "processed", tmp_path / "output"),
        _config_file=config_file,
    )
    save_profile(
        "laptop",
        _profile(tmp_path / "raw2", tmp_path / "processed2", tmp_path / "output2"),
        _config_file=config_file,
    )

    delete_profile("laptop", _config_file=config_file)
    assert list_profiles(_config_file=config_file) == ["default"]

    with pytest.raises(ValueError, match="default profile"):
        delete_profile("default", _config_file=config_file)


def test_save_profile_duplicate_without_overwrite_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    save_profile(
        "default",
        _profile(tmp_path / "raw", tmp_path / "processed", tmp_path / "output"),
        _config_file=config_file,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        save_profile(
            "default",
            _profile(tmp_path / "raw2", tmp_path / "processed2", tmp_path / "output2"),
            _config_file=config_file,
        )


def test_save_profile_with_overwrite_replaces_existing(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    save_profile(
        "default",
        _profile(tmp_path / "raw", tmp_path / "processed", tmp_path / "output"),
        _config_file=config_file,
    )
    replacement = _profile(tmp_path / "raw2", tmp_path / "processed2", tmp_path / "output2")

    save_profile("default", replacement, overwrite=True, _config_file=config_file)

    assert load_profile(_config_file=config_file) == replacement


def test_profile_config_expands_tilde_and_resolves_absolute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    config = ProfileConfig(
        raw_dir="~/raw",
        processed_dir="~/processed",
        output_dir="~/output",
    )

    assert config.raw_dir == (home / "raw").resolve()
    assert config.processed_dir == (home / "processed").resolve()
    assert config.output_dir == (home / "output").resolve()


def test_profile_config_rejects_missing_fields() -> None:
    with pytest.raises(ValidationError):
        ProfileConfig.model_validate(
            {"processed_dir": "/tmp/processed", "output_dir": "/tmp/output"}
        )


def test_profile_config_rejects_blank_or_whitespace_paths() -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        ProfileConfig.model_validate(
            {
                "raw_dir": " ",
                "processed_dir": "/tmp/processed",
                "output_dir": "/tmp/output",
            }
        )


def test_profile_precedence_profile_then_env_file_then_process_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("NEURALLS_PROCESSED_DIR", "NEURALLS_OUTPUT_DIR"):
        monkeypatch.delenv(key, raising=False)
    case_config = tmp_path / "case.toml"
    case_config.write_text("", encoding="utf-8")
    profile_file = tmp_path / "config.toml"
    _configure_profile_path(monkeypatch, profile_file)
    save_profile(
        "default",
        _profile(
            tmp_path / "profile-raw", tmp_path / "profile-processed", tmp_path / "profile-output"
        ),
        _config_file=profile_file,
    )
    env_file = tmp_path / "case.env"
    env_file.write_text(
        "\n".join(
            [
                f"NEURALLS_PROCESSED_DIR={tmp_path / 'env-processed'}",
                f"NEURALLS_OUTPUT_DIR={tmp_path / 'env-output'}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NEURALLS_OUTPUT_DIR", str(tmp_path / "env-process-output"))

    settings = load_case_settings(case_config, env_file)

    assert settings.processed_dir == (tmp_path / "env-processed").resolve()
    assert settings.output_dir == (tmp_path / "env-process-output").resolve()


def test_neuralls_profile_env_var_selects_named_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "config.toml"
    _configure_profile_path(monkeypatch, config_file)
    save_profile(
        "default",
        _profile(tmp_path / "raw", tmp_path / "processed", tmp_path / "output"),
        _config_file=config_file,
    )
    expected = _profile(tmp_path / "raw2", tmp_path / "processed2", tmp_path / "output2")
    save_profile("laptop", expected, _config_file=config_file)
    monkeypatch.setenv(PROFILE_ENV_VAR, "laptop")

    assert load_profile(_config_file=config_file) == expected
