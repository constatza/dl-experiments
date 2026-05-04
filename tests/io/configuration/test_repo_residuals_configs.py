from __future__ import annotations

from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_toml(relative_path: str) -> dict:
    with (REPO_ROOT / relative_path).open("rb") as handle:
        return tomllib.load(handle)


def test_residuals_100_dataset_uses_residuals_strategy() -> None:
    """The shipped residual dataset config must generate residual-error pairs."""
    config = _load_toml("configs/datasets/residuals-100.toml")
    strategies = config["generation"]["strategy"]

    assert len(strategies) == 1
    assert strategies[0]["name"] == "residuals"
    assert strategies[0]["samples"] == 20000
    assert "solutions_glob" in strategies[0]


def test_residuals_100_gaussian_dataset_uses_gaussian_residuals_strategy() -> None:
    """The Gaussian residual dataset config must sample N(0, 1) true solutions."""
    config = _load_toml("configs/datasets/residuals-100-gaussian.toml")
    strategies = config["generation"]["strategy"]

    assert len(strategies) == 1
    assert strategies[0]["name"] == "gaussian_residuals"
    assert strategies[0]["samples"] == 20000
    assert "solutions_glob" not in strategies[0]


def test_current_experiment_registries_reference_residuals_dataset() -> None:
    """Residual registries must expose both archive and Gaussian residual datasets."""
    registry_paths = [
        "configs/experiments-ffnn.toml",
        "configs/experiments-linear.toml",
        "configs/experiments-parametrized.toml",
    ]

    for registry_path in registry_paths:
        config = _load_toml(registry_path)
        datasets = {entry["id"]: entry["path"] for entry in config["datasets"]}
        assert datasets["residuals-100"] == "datasets/residuals-100.toml"
        assert datasets["residuals-100-gaussian"] == "datasets/residuals-100-gaussian.toml"


def test_residual_experiments_use_residuals_100_dataset() -> None:
    """Every active residual-labelled experiment should use one of the residual datasets."""
    registry_paths = [
        "configs/experiments-ffnn.toml",
        "configs/experiments-linear.toml",
    ]

    for registry_path in registry_paths:
        config = _load_toml(registry_path)
        residual_experiments = [
            entry
            for entry in config["experiments"]
            if "residual" in entry["id"].lower()
            or "residual" in entry.get("display_name", "").lower()
        ]

        assert residual_experiments
        assert all(
            entry["dataset"] in {"residuals-100", "residuals-100-gaussian"}
            for entry in residual_experiments
        )
