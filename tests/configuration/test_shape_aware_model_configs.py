"""Regression tests: shape-aware model configs must not declare in_features/out_features.

dlkit's from_shape() passes in_features and out_features explicitly from the dataset
shape summary. If the TOML config also declares them, Python raises "multiple values for
keyword argument". These fields are redundant for shape-aware models — dlkit infers them.
"""

from __future__ import annotations

from pathlib import Path

import tomllib
import pytest


_CONFIGS_ROOT = Path(__file__).parents[2] / "configs" / "models"
_SHAPE_AWARE_PREFIX = "ScaleEquivariant"
_FORBIDDEN_KEYS = ("in_features", "out_features")


def _shape_aware_model_configs() -> list[Path]:
    """Return all model TOML configs whose MODEL.name starts with the shape-aware prefix."""
    return [
        p
        for p in _CONFIGS_ROOT.glob("*.toml")
        if tomllib.loads(p.read_text())
        .get("MODEL", {})
        .get("name", "")
        .startswith(_SHAPE_AWARE_PREFIX)
    ]


@pytest.fixture(params=_shape_aware_model_configs(), ids=lambda p: p.stem)
def shape_aware_config(request: pytest.FixtureRequest) -> dict[str, object]:
    """Parsed TOML for each shape-aware model config."""
    path: Path = request.param
    return tomllib.loads(path.read_text())


def test_shape_aware_model_config_omits_in_and_out_features(
    shape_aware_config: dict[str, object],
) -> None:
    """Shape-aware model configs must not declare in_features or out_features.

    dlkit infers these from the dataset; declaring them causes "multiple values for
    keyword argument" at runtime.
    """
    model_section: dict[str, object] = shape_aware_config.get("MODEL", {})  # type: ignore[assignment]
    for key in _FORBIDDEN_KEYS:
        assert key not in model_section, (
            f"Remove '{key}' from [MODEL] — dlkit infers it from the dataset shape. "
            "Keeping it causes 'multiple values for keyword argument' at training time."
        )
