"""Runtime settings and strict case-based settings loading for neuralls."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CASE_CONFIG_ENV_VAR = "NEURALLS_CASE_CONFIG"
ENV_FILE_ENV_VAR = "NEURALLS_ENV_FILE"
_ENV_TO_FIELD = {
    "NEURALLS_RAW_DIR": "raw_dir",
    "NEURALLS_PROCESSED_DIR": "processed_dir",
    "NEURALLS_OUTPUT_DIR": "output_dir",
}


class NeurallsSettings(BaseSettings):
    """Resolved runtime roots for processed and output filesystem state."""

    model_config = SettingsConfigDict(
        env_prefix="NEURALLS_",
        extra="forbid",
        frozen=True,
    )

    raw_dir: Path | None = Field(default=None, description="Raw matrices and archives root")
    processed_dir: Path = Field(..., description="Generated datasets root")
    output_dir: Path = Field(..., description="MLflow artifacts, checkpoints, figures root")

    @model_validator(mode="after")
    def _resolve_absolute(self) -> NeurallsSettings:
        """Expand ~ and resolve all root dirs to absolute paths."""
        if self.raw_dir is not None:
            object.__setattr__(self, "raw_dir", self.raw_dir.expanduser().resolve())
        object.__setattr__(self, "processed_dir", self.processed_dir.expanduser().resolve())
        object.__setattr__(self, "output_dir", self.output_dir.expanduser().resolve())
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def figures_dir(self) -> Path:
        """Figures output directory, derived from output_dir."""
        return self.output_dir / "figures"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mlruns_dir(self) -> Path:
        """MLflow runs directory, derived from output_dir."""
        return self.output_dir / "mlruns"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mlartifacts_dir(self) -> Path:
        """MLflow artifacts directory, derived from output_dir."""
        return self.output_dir / "mlartifacts"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mlflow_tracking_uri(self) -> str:
        """SQLite tracking URI, derived from mlruns_dir. Uses posix slashes for cross-platform compat."""
        db = self.mlruns_dir / "mlflow.db"
        return f"sqlite:///{db.as_posix()}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mlflow_artifact_location(self) -> str:
        """MLflow artifact storage path as posix string (cross-platform)."""
        return self.mlartifacts_dir.as_posix()


def _resolve_env_file_path(env_file: Path | None) -> Path | None:
    """Resolve the explicit env file path or the compatibility env-file alias."""
    if env_file is not None:
        return env_file.expanduser().resolve()
    configured = os.getenv(ENV_FILE_ENV_VAR)
    if configured is None or not configured.strip():
        return None
    return Path(configured).expanduser().resolve()


def _strip_optional_quotes(value: str) -> str:
    """Strip one matching pair of surrounding quotes from an env value."""
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped


def _read_root_overrides_from_env_file(env_file: Path | None) -> dict[str, str]:
    """Parse only the root override variables from one explicit env file."""
    if env_file is None:
        return {}
    values: dict[str, str] = {}
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, sep, raw_value = line.partition("=")
        if not sep:
            continue
        normalized_key = key.strip()
        field_name = _ENV_TO_FIELD.get(normalized_key)
        if field_name is None:
            continue
        values[field_name] = _strip_optional_quotes(raw_value)
    return values


def _read_root_overrides_from_process_env() -> dict[str, str]:
    """Read root overrides from the current process environment."""
    values: dict[str, str] = {}
    for env_name, field_name in _ENV_TO_FIELD.items():
        env_value = os.getenv(env_name)
        if env_value is not None:
            values[field_name] = env_value
    return values


def _build_case_root_mapping(
    *,
    env_file: Path | None,
    profile_values: dict[str, str],
) -> dict[str, str]:
    """Resolve final runtime roots with strict profile/env precedence."""
    env_file_values = _read_root_overrides_from_env_file(env_file)
    process_env_values = _read_root_overrides_from_process_env()
    return {**profile_values, **env_file_values, **process_env_values}


def get_settings(env_file: Path | None = None) -> NeurallsSettings:
    """Load settings from one explicit env file or from process env only."""
    return NeurallsSettings(_env_file=_resolve_env_file_path(env_file))  # ty: ignore[unknown-argument]


def load_case_settings(
    case_config_path: Path,
    env_file: Path | None = None,
    profile: str | None = None,
) -> NeurallsSettings:
    """Resolve settings for one case config using strict case/env/profile precedence.

    Args:
        case_config_path: Path to the case TOML file.
        env_file: Optional path to an explicit .env file with root overrides.
        profile: Optional profile name to load from the user config. Falls back
            to the ``NEURALLS_PROFILE`` env var, then ``"default"``.

    Returns:
        A fully resolved and validated NeurallsSettings instance.
    """
    from neuralls.platform.config.profile import load_profile

    env_file_path = _resolve_env_file_path(env_file)
    profile_config = load_profile(profile)
    profile_values = profile_config.to_root_mapping() if profile_config else {}
    root_mapping = _build_case_root_mapping(
        env_file=env_file_path,
        profile_values=profile_values,
    )
    return NeurallsSettings.model_validate(root_mapping)


def require_settings(
    settings: NeurallsSettings | None,
    *,
    case_config_path: Path | None = None,
    env_file: Path | None = None,
    profile: str | None = None,
) -> NeurallsSettings:
    """Return explicit settings or resolve them from one explicit case config."""
    if settings is not None:
        return settings
    resolved_case = case_config_path
    if resolved_case is None:
        configured_case = os.getenv(CASE_CONFIG_ENV_VAR)
        if configured_case is not None and configured_case.strip():
            resolved_case = Path(configured_case)
    if resolved_case is None:
        raise ValueError(
            "This workflow requires a case config. Pass --case-config or set NEURALLS_CASE_CONFIG."
        )
    return load_case_settings(resolved_case, env_file=env_file, profile=profile)
