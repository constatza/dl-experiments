"""Runtime settings and strict case-based settings loading for neuralls."""

from __future__ import annotations

import os
from pathlib import Path
import tomllib

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CASE_CONFIG_ENV_VAR = "NEURALLS_CASE_CONFIG"
ENV_FILE_ENV_VAR = "NEURALLS_ENV_FILE"
_ROOT_FIELDS = ("raw_dir", "processed_dir", "output_dir")
_ENV_TO_FIELD = {
    "NEURALLS_RAW_DIR": "raw_dir",
    "NEURALLS_PROCESSED_DIR": "processed_dir",
    "NEURALLS_OUTPUT_DIR": "output_dir",
}


class NeurallsSettings(BaseSettings):
    """Resolved runtime roots for raw, processed, and output filesystem state."""

    model_config = SettingsConfigDict(
        env_prefix="NEURALLS_",
        extra="forbid",
        frozen=True,
    )

    raw_dir: Path = Field(..., description="Raw input matrices and archives")
    processed_dir: Path = Field(..., description="Generated datasets root")
    output_dir: Path = Field(..., description="MLflow artifacts, checkpoints, figures root")

    @model_validator(mode="after")
    def _resolve_absolute(self) -> NeurallsSettings:
        """Expand ~ and resolve all root dirs to absolute paths."""
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


def _read_raw_case_toml(case_config_path: Path) -> dict[str, object]:
    """Read one case TOML into a plain dictionary."""
    with case_config_path.open("rb") as fh:
        return tomllib.load(fh)


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


def _resolve_case_root_value(
    *,
    field_name: str,
    raw_value: object,
    case_config_path: Path,
    override_values: dict[str, str],
) -> str:
    """Resolve one case root from a relative path or a placeholder-backed string."""
    if not isinstance(raw_value, str):
        raise ValueError(
            f"Case config field '{field_name}' must be a string path, got "
            f"{type(raw_value).__name__}."
        )

    value = raw_value
    if "${GRAPH_CG_" in value:
        from neuralls.platform.config.context import ConfigContext, expand_config_path

        placeholder_settings = NeurallsSettings.model_construct(
            raw_dir=Path(override_values.get("raw_dir", ".")),
            processed_dir=Path(override_values.get("processed_dir", ".")),
            output_dir=Path(override_values.get("output_dir", ".")),
        )
        return expand_config_path(
            value,
            ConfigContext(config_path=case_config_path, settings=placeholder_settings),
        )

    for env_name, placeholder_field in _ENV_TO_FIELD.items():
        placeholder = f"${{{env_name}}}"
        if placeholder not in value:
            continue
        override = override_values.get(placeholder_field)
        if override is None:
            raise ValueError(
                f"Case config field '{field_name}' references {placeholder}, "
                f"but no explicit {env_name} override was provided."
            )
        value = value.replace(placeholder, override)

    path = Path(os.path.expanduser(value))
    anchored = path if path.is_absolute() else case_config_path.parent / path
    return str(anchored.resolve())


def _build_case_root_mapping(
    *,
    case_config_path: Path,
    raw_case: dict[str, object],
    env_file: Path | None,
) -> dict[str, str]:
    """Resolve final case-root values with strict precedence."""
    env_file_values = _read_root_overrides_from_env_file(env_file)
    process_env_values = _read_root_overrides_from_process_env()
    placeholder_overrides = dict(env_file_values)
    placeholder_overrides.update(process_env_values)

    case_values: dict[str, str] = {}
    for field_name in _ROOT_FIELDS:
        raw_value = raw_case.get(field_name)
        if raw_value is None:
            continue
        case_values[field_name] = _resolve_case_root_value(
            field_name=field_name,
            raw_value=raw_value,
            case_config_path=case_config_path,
            override_values=placeholder_overrides,
        )

    resolved = dict(case_values)
    resolved.update(env_file_values)
    resolved.update(process_env_values)
    return resolved


def get_settings(env_file: Path | None = None) -> NeurallsSettings:
    """Load settings from one explicit env file or from process env only."""
    return NeurallsSettings(_env_file=_resolve_env_file_path(env_file))  # ty: ignore[unknown-argument]


def load_case_settings(
    case_config_path: Path,
    env_file: Path | None = None,
) -> NeurallsSettings:
    """Resolve settings for one case config using strict case/env precedence."""
    resolved_case_path = case_config_path.expanduser().resolve()
    raw_case = _read_raw_case_toml(resolved_case_path)
    env_file_path = _resolve_env_file_path(env_file)
    root_mapping = _build_case_root_mapping(
        case_config_path=resolved_case_path,
        raw_case=raw_case,
        env_file=env_file_path,
    )
    return NeurallsSettings.model_validate(root_mapping)


def require_settings(
    settings: NeurallsSettings | None,
    *,
    case_config_path: Path | None = None,
    env_file: Path | None = None,
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
            "This workflow requires a case config. Pass --case-config or set "
            "NEURALLS_CASE_CONFIG."
        )
    return load_case_settings(resolved_case, env_file=env_file)
