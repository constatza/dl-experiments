"""Tests for explicit dataset identity resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.configuration.data_models import DataConfigFile
from neuralls.configuration.dataset_identity import (
    normalize_registry_alias,
    resolve_dataset_identity,
    resolve_dataset_identity_from_mapping,
)


def test_resolve_dataset_identity_requires_top_level_id_in_mapping() -> None:
    with pytest.raises(ValueError, match="top-level 'id'"):
        resolve_dataset_identity_from_mapping(
            config={
                "source": {},
                "generation": {},
                "output": {"data_dir": "tests/fixtures/data/datasets"},
            },
            config_path=Path("datasets/solutions.toml"),
        )


def test_resolve_dataset_identity_falls_back_to_path_stem() -> None:
    data_cfg = DataConfigFile.model_validate(
        {
            "source": {},
            "generation": {},
            "output": {"data_dir": "tests/fixtures/data/processed/solutions"},
            "test": {},
        }
    )
    identity = resolve_dataset_identity(
        data_cfg=data_cfg,
        config_path=Path("datasets/solutions.toml"),
    )
    assert identity.name == "solutions"
    assert identity.registry_alias == "solutions"
    assert identity.source == "id"


def test_normalize_registry_alias_rejects_invalid_chars() -> None:
    with pytest.raises(ValueError, match=r"\^\[\\w\\-\]\+\$"):
        normalize_registry_alias("alias with spaces")
