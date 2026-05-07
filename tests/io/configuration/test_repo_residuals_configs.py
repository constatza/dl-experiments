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


def test_ffnn_registry_uses_residual_model_configs() -> None:
    """Checked-in FFNN registry entries should point at explicit residual-network configs."""
    config = _load_toml("configs/experiments-ffnn.toml")
    model_paths = {entry["id"]: entry["path"] for entry in config["models"]}

    assert model_paths["scaleequivariant-residual-ffnn"] == "models/skip-ffnn.toml"
    assert (
        model_paths["scaleequivariant-embedded-factorized-residual-ffnn"]
        == "models/embedded-factorized-skip-ffnn.toml"
    )
    assert (
        model_paths["scaleequivariant-embedded-spd-residual-ffnn"]
        == "models/embedded-spd-skip-ffnn.toml"
    )
    assert (
        model_paths["scaleequivariant-embedded-spd-factorized-residual-ffnn"]
        == "models/embedded-spd-factorized-skip-ffnn.toml"
    )


def test_ffnn_residual_model_configs_use_residual_classes() -> None:
    """Each shipped FFNN model config should use the intended residual-network API."""
    expected_names = {
        "configs/models/skip-ffnn.toml": "ScaleEquivariantConstantWidthFactorizedFFNN",
        "configs/models/embedded-factorized-skip-ffnn.toml": (
            "ScaleEquivariantEmbeddedFactorizedFFNN"
        ),
        "configs/models/embedded-spd-skip-ffnn.toml": "ScaleEquivariantEmbeddedSPDFFNN",
        "configs/models/embedded-spd-factorized-skip-ffnn.toml": (
            "ScaleEquivariantEmbeddedSPDFactorizedFFNN"
        ),
    }

    for model_path, expected_name in expected_names.items():
        config = _load_toml(model_path)
        assert config["MODEL"]["name"] == expected_name

    for model_path in expected_names:
        config = _load_toml(model_path)
        assert "residual" not in config["MODEL"]
