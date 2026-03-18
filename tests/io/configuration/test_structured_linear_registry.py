"""Tests for structured linear model configs and registry layouts."""

from __future__ import annotations

from pathlib import Path
import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("dlkit") is None,
    reason="dlkit is not importable in this environment",
)

from neuralls.configuration.loader import load_experiment
from neuralls.configuration.master_registry import (
    list_experiment_bindings,
    load_validated_master_config,
)
from neuralls.io.toml_loader import load_model_config


MODEL_CONFIG_TEMPLATE = """
[SESSION]
seed = 42
precision = "float64"

[MODEL]
name = "{model_name}"
module_path = "dlkit.nn.ffnn"

[TRAINING]
[TRAINING.trainer]
max_epochs = 1
accelerator = "cpu"
enable_checkpointing = true

[[TRAINING.trainer.callbacks]]
name = "ModelCheckpoint"
filename = "{checkpoint_name}"
monitor = "val_loss"

[TRAINING.optimizer]
lr = 1e-3

[DATASET]
name = "FlexibleDataset"
memmap_cache = true

[DATAMODULE]
name = "InMemoryModule"

[OPTUNA]
enabled = false
"""


def _write_model_config(path: Path, model_name: str, checkpoint_name: str) -> Path:
    """Write one isolated model config."""
    path.write_text(
        MODEL_CONFIG_TEMPLATE.format(
            model_name=model_name,
            checkpoint_name=checkpoint_name,
        )
    )
    return path


def _write_master_registry(
    path: Path,
    *,
    datasets: list[tuple[str, str]],
    models: list[tuple[str, str]],
    experiments: list[tuple[str, str, str]],
) -> Path:
    """Write a minimal experiments registry with explicit tables."""
    lines = [
        'project_root = ".."',
        f'output_dir = "{path.parent / "output"}"',
        "",
        "[mlflow]",
        'tracking_uri = "http://127.0.0.1:5000"',
        "",
        "[names]",
        'training = "Train"',
        'comparison = "Comparisons"',
        "",
    ]

    for dataset_id, dataset_path in datasets:
        lines.extend(
            [
                "[[datasets]]",
                f'id = "{dataset_id}"',
                f'path = "{dataset_path}"',
                "",
            ]
        )

    for model_id, model_path in models:
        lines.extend(
            [
                "[[models]]",
                f'id = "{model_id}"',
                f'path = "{model_path}"',
                "",
            ]
        )

    for experiment_id, dataset_id, model_id in experiments:
        lines.extend(
            [
                "[[experiments]]",
                f'id = "{experiment_id}"',
                f'dataset = "{dataset_id}"',
                f'model = "{model_id}"',
                "",
            ]
        )

    path.write_text("\n".join(lines))
    return path


@pytest.mark.parametrize(
    ("config_name", "model_name"),
    [
        ("symmetric.toml", "NormScaledSymmetricLinear"),
        ("spd.toml", "NormScaledSPDLinear"),
        ("factorized.toml", "NormScaledFactorizedLinear"),
    ],
)
def test_structured_linear_model_configs_load(
    tmp_path: Path,
    config_name: str,
    model_name: str,
) -> None:
    """Structured linear model configs parse through dlkit."""
    model_path = _write_model_config(
        tmp_path / config_name,
        model_name=model_name,
        checkpoint_name=config_name.removesuffix(".toml"),
    )

    settings = load_model_config(model_path)
    assert settings.MODEL.name == model_name


def test_default_structured_linear_registry_has_expected_models_and_experiments(
    tmp_path: Path,
) -> None:
    """A structured-linear registry resolves the intended 3 x 3 experiment matrix."""
    config_path = _write_master_registry(
        tmp_path / "experiments.toml",
        datasets=[
            ("residuals-100", "datasets/residuals-100.toml"),
            ("eig-rhs-smallest", "datasets/eig-rhs-smallest.toml"),
            ("eig-solutions-smallest", "datasets/eig-solutions-smallest.toml"),
        ],
        models=[
            ("symmetric", "models/symmetric.toml"),
            ("spd", "models/spd.toml"),
            ("factorized", "models/factorized.toml"),
        ],
        experiments=[
            ("eig-rhs-smallest-symmetric", "eig-rhs-smallest", "symmetric"),
            ("eig-rhs-smallest-spd", "eig-rhs-smallest", "spd"),
            ("eig-rhs-smallest-factorized", "eig-rhs-smallest", "factorized"),
            ("eig-solutions-smallest-symmetric", "eig-solutions-smallest", "symmetric"),
            ("eig-solutions-smallest-spd", "eig-solutions-smallest", "spd"),
            ("eig-solutions-smallest-factorized", "eig-solutions-smallest", "factorized"),
            ("residuals-symmetric", "residuals-100", "symmetric"),
            ("residuals-spd", "residuals-100", "spd"),
            ("residuals-factorized", "residuals-100", "factorized"),
        ],
    )

    cfg, config_dir = load_validated_master_config(config_path)
    bindings = list_experiment_bindings(cfg, config_dir)

    assert [entry.id for entry in cfg.models] == ["symmetric", "spd", "factorized"]
    assert [entry.id for entry in cfg.datasets] == [
        "residuals-100",
        "eig-rhs-smallest",
        "eig-solutions-smallest",
    ]
    assert [binding.experiment_id for binding in bindings] == [
        "eig-rhs-smallest-symmetric",
        "eig-rhs-smallest-spd",
        "eig-rhs-smallest-factorized",
        "eig-solutions-smallest-symmetric",
        "eig-solutions-smallest-spd",
        "eig-solutions-smallest-factorized",
        "residuals-symmetric",
        "residuals-spd",
        "residuals-factorized",
    ]


def test_legacy_linear_registry_is_preserved(tmp_path: Path) -> None:
    """A legacy linear-focused registry can coexist under a different filename."""
    config_path = _write_master_registry(
        tmp_path / "experiments-linear.toml",
        datasets=[
            ("spectral", "datasets/solutions.toml"),
            ("residuals-100", "datasets/residuals-100.toml"),
            ("eig-rhs-largest", "datasets/eig-rhs-largest.toml"),
            ("eig-rhs-smallest", "datasets/eig-rhs-smallest.toml"),
            ("eig-solutions-largest", "datasets/eig-solutions-largest.toml"),
            ("eig-solutions-smallest", "datasets/eig-solutions-smallest.toml"),
        ],
        models=[
            ("linear", "models/linear.toml"),
            ("linear-l2", "models/linear-l2.toml"),
        ],
        experiments=[
            ("spectral-l2", "spectral", "linear-l2"),
            ("spectral-energy", "spectral", "linear"),
            ("eig-rhs-largest", "eig-rhs-largest", "linear"),
            ("eig-rhs-smallest", "eig-rhs-smallest", "linear"),
            ("eig-solutions-largest", "eig-solutions-largest", "linear"),
            ("eig-solutions-smallest", "eig-solutions-smallest", "linear"),
            ("residuals", "residuals-100", "linear"),
        ],
    )

    cfg, config_dir = load_validated_master_config(config_path)
    bindings = list_experiment_bindings(cfg, config_dir)

    assert [entry.id for entry in cfg.models] == ["linear", "linear-l2"]
    assert [binding.experiment_id for binding in bindings] == [
        "spectral-l2",
        "spectral-energy",
        "eig-rhs-largest",
        "eig-rhs-smallest",
        "eig-solutions-largest",
        "eig-solutions-smallest",
        "residuals",
    ]


def test_load_experiment_supports_structured_linear_model(tmp_path: Path) -> None:
    """A structured model config loads through the training entry point."""
    output_root = tmp_path / "output"
    output_root.mkdir()
    data_config_path = tmp_path / "data.toml"
    data_config_path.write_text('[flow]\ndataset = "dummy_dataset"\n')
    model_config_path = _write_model_config(
        tmp_path / "symmetric.toml",
        model_name="NormScaledSymmetricLinear",
        checkpoint_name="symmetric",
    )

    experiment = load_experiment(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        output_root=output_root,
        dataset_registry_id="dummy-dataset",
    )

    assert experiment.spec.model_config_path.name == "symmetric.toml"
    assert experiment.spec.dataset_registry_id == "dummy-dataset"
    assert experiment.settings.MODEL.name == "NormScaledSymmetricLinear"
