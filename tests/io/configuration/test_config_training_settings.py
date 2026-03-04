"""Config loading smoke tests for training sections."""

from __future__ import annotations

from pathlib import Path

import tomllib

import pytest

from neuralls.configuration.loader import load_experiment


MINIMAL_FFNN_CONFIG = """
[SESSION]
seed = 42
[PATHS]
project_root = "{tmp_dir}"
results_dir = "{tmp_dir}/output"
[TRAINING]
[TRAINING.trainer]
max_epochs = 1
[[TRAINING.trainer.callbacks]]
name = "ModelCheckpoint"
[[TRAINING.metrics]]
name = "NormalizedVectorNormError"
[DATASET]
name = "FlexibleDataset"
memmap_cache = true
[MODEL]
name = "NormScaledConstantWidthFFNN"
module_path = "test.module"
"""

MINIMAL_LINEAR_CONFIG = """
[SESSION]
seed = 42
[PATHS]
project_root = "{tmp_dir}"
results_dir = "{tmp_dir}/output"
[TRAINING]
[TRAINING.trainer]
max_epochs = 1
[[TRAINING.trainer.callbacks]]
name = "EarlyStopping"
[DATASET]
name = "FlexibleDataset"
memmap_cache = true
[MODEL]
name = "LinearModel"
module_path = "test.module"
"""

MINIMAL_GNN_CONFIG = """
[SESSION]
seed = 42
[PATHS]
project_root = "{tmp_dir}"
results_dir = "{tmp_dir}/output"
[TRAINING]
[TRAINING.trainer]
max_epochs = 1
[DATASET]
name = "GraphDataset"
[MODEL]
name = "GNNModel"
module_path = "test.module"
"""


@pytest.mark.parametrize(
    "config_content_template",
    [
        MINIMAL_FFNN_CONFIG,
        MINIMAL_LINEAR_CONFIG,
        MINIMAL_GNN_CONFIG,
    ],
)
def test_training_sections_round_trip(
    tmp_path: Path, config_content_template: str
) -> None:
    """Ensure load_experiment preserves trainer callbacks/metrics from a temporary file."""
    # Create necessary directories that dlkit expects
    (tmp_path / "output").mkdir()

    # Inject the temporary path into the config content
    config_content = config_content_template.format(tmp_dir=str(tmp_path))

    # Setup temporary config file
    config_path = tmp_path / "linear.toml"
    config_path.write_text(config_content)

    with config_path.open("rb") as fh:
        raw_config = tomllib.load(fh)

    raw_training = raw_config.get("TRAINING", {})
    raw_trainer = raw_training.get("trainer", {})
    raw_callbacks = tuple(raw_trainer.get("callbacks", ()))
    expected_callback_names = tuple(cb.get("name") for cb in raw_callbacks)

    raw_metrics = tuple(raw_training.get("metrics", ()))
    expected_metric_names = tuple(m.get("name") for m in raw_metrics)

    # Create dummy data config
    data_path = tmp_path / "data.toml"
    data_path.write_text('[flow]\ndataset="dummy_dataset"')

    experiment = load_experiment(
        config_path,
        data_config_path=data_path,
        output_root=tmp_path / "output",
    )
    settings = experiment.settings
    training = settings.TRAINING
    assert training is not None, "TRAINING section missing"
    assert settings.DATASET is not None, "DATASET section missing"

    actual_callback_names = tuple(cb.name for cb in training.trainer.callbacks)
    assert actual_callback_names == expected_callback_names

    actual_metric_names = tuple(metric.name for metric in training.metrics)
    assert actual_metric_names == expected_metric_names

    dataset_name = raw_config.get("DATASET", {}).get("name")
    if dataset_name == "FlexibleDataset":
        assert settings.DATASET.memmap_cache is True
    else:
        assert settings.DATASET.memmap_cache is False
