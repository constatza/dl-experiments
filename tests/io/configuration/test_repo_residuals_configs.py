from __future__ import annotations

import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_toml(relative_path: str) -> dict:
    with (REPO_ROOT / relative_path).open("rb") as handle:
        return tomllib.load(handle)


def _load_model_toml(relative_path: str) -> dict:
    """Load one model TOML relative to configs/models."""
    return _load_toml(f"configs/models/{relative_path}")


# ---------------------------------------------------------------------------
# Embedded model config fixtures — self-contained, do not depend on configs/
# ---------------------------------------------------------------------------

_SKIP_FFNN_TOML = """\
[SESSION]
seed = 42
workflow = "train"
precision = "float64"

[MODEL]
name = "ScaleEquivariantConstantWidthFactorizedFFNN"
module_path = "dlkit.nn"
size = 504
num_layers = 1
norm = "l2"
dropout = 0.0

[TRAINING.lr_tuner]
min_lr = 1e-5
max_lr = 1e-1
num_training = 100
mode = "linear"

[TRAINING.trainer]
max_epochs = 300
accelerator = "auto"
enable_checkpointing = true

[[TRAINING.trainer.callbacks]]
name = "ModelCheckpoint"
filename = "constant-width-factorized"
monitor = "val_loss"
save_top_k = 1
save_weights_only = false
enable_version_counter = false

[[TRAINING.trainer.callbacks]]
name = "EarlyStopping"
monitor = "val_loss"
min_delta = 1e-4
patience = 30
mode = "min"
check_finite = true

[TRAINING.optimizer.default_optimizer]
name = "AdamW"
lr = 1e-3
weight_decay = 1e-2

[TRAINING.optimizer.default_scheduler]
name = "ReduceLROnPlateau"
mode = "min"
factor = 0.5
patience = 10
cooldown = 5
threshold = 1e-4
min_lr = 1e-6

[TRAINING.loss_function]
name = "normalized_vector_norm_loss"
module_path = "dlkit.domain.losses"
target_key = "targets.solutions"

[[TRAINING.metrics]]
name = "NormalizedVectorNormError"
module_path = "dlkit.domain.metrics"
norm_ord = 2
vector_dim = -1

[DATASET]
name = "FlexibleDataset"
memmap_cache = true

[DATAMODULE]
name = "InMemoryModule"

[DATAMODULE.dataloader]
num_workers = 0
batch_size = 32
pin_memory = true
shuffle = true

[OPTUNA]
enabled = false
n_trials = 40

[OPTUNA.model]
num_layers = {choices = [1, 6, 9]}
dropout = {low = 0.0, high = 0.3, step = 0.05}
size = {choices = [252, 504, 756]}
"""

_EMBEDDED_FACTORIZED_TOML = """\
[SESSION]
seed = 42
workflow = "train"
precision = "float64"

[MODEL]
name = "ScaleEquivariantEmbeddedFactorizedFFNN"
module_path = "dlkit.nn"
hidden_size = 500
num_layers = 1
bias = true
norm = "l2"
dropout = 0.0

[TRAINING.lr_tuner]
min_lr = 1e-5
max_lr = 1e-1
num_training = 100
mode = "linear"

[TRAINING.trainer]
max_epochs = 300
accelerator = "auto"
enable_checkpointing = true

[[TRAINING.trainer.callbacks]]
name = "ModelCheckpoint"
filename = "embedded-factorized"
monitor = "val_loss"
save_top_k = 1
save_weights_only = false
enable_version_counter = false

[[TRAINING.trainer.callbacks]]
name = "EarlyStopping"
monitor = "val_loss"
min_delta = 1e-4
patience = 30
mode = "min"
check_finite = true

[TRAINING.optimizer.default_optimizer]
name = "AdamW"
lr = 1e-3
weight_decay = 1e-2

[TRAINING.optimizer.default_scheduler]
name = "ReduceLROnPlateau"
mode = "min"
factor = 0.5
patience = 10
cooldown = 5
threshold = 1e-4
min_lr = 1e-6

[TRAINING.loss_function]
name = "normalized_vector_norm_loss"
module_path = "dlkit.domain.losses"
target_key = "targets.solutions"

[[TRAINING.metrics]]
name = "NormalizedVectorNormError"
module_path = "dlkit.domain.metrics"
norm_ord = 2
vector_dim = -1

[DATASET]
name = "FlexibleDataset"
memmap_cache = true

[DATAMODULE]
name = "InMemoryModule"

[DATAMODULE.dataloader]
num_workers = 0
batch_size = 32
pin_memory = true
shuffle = true

[OPTUNA]
enabled = false
n_trials = 40

[OPTUNA.model]
num_layers = {choices = [1, 6, 9]}
dropout = {low = 0.0, high = 0.3, step = 0.05}
hidden_size = {choices = [256, 512, 768]}
"""

_EMBEDDED_SPD_TOML = """\
[SESSION]
seed = 42
workflow = "train"
precision = "float64"

[MODEL]
name = "ScaleEquivariantEmbeddedSPDFFNN"
module_path = "dlkit.nn"
hidden_size = 500
num_layers = 1
bias = false
min_diag = 1e-4
norm = "l2"
dropout = 0.0

[TRAINING.lr_tuner]
min_lr = 1e-5
max_lr = 1e-1
num_training = 100
mode = "linear"

[TRAINING.trainer]
max_epochs = 300
accelerator = "auto"
enable_checkpointing = true

[[TRAINING.trainer.callbacks]]
name = "ModelCheckpoint"
filename = "embedded-spd"
monitor = "val_loss"
save_top_k = 1
save_weights_only = false
enable_version_counter = false

[[TRAINING.trainer.callbacks]]
name = "EarlyStopping"
monitor = "val_loss"
min_delta = 1e-4
patience = 30
mode = "min"
check_finite = true

[TRAINING.optimizer.default_optimizer]
name = "AdamW"
lr = 1e-3
weight_decay = 1e-2

[TRAINING.optimizer.default_scheduler]
name = "ReduceLROnPlateau"
mode = "min"
factor = 0.5
patience = 10
cooldown = 5
threshold = 1e-4
min_lr = 1e-6

[TRAINING.loss_function]
name = "normalized_vector_norm_loss"
module_path = "dlkit.domain.losses"
target_key = "targets.solutions"

[[TRAINING.metrics]]
name = "NormalizedVectorNormError"
module_path = "dlkit.domain.metrics"
norm_ord = 2
vector_dim = -1

[DATASET]
name = "FlexibleDataset"
memmap_cache = true

[DATAMODULE]
name = "InMemoryModule"

[DATAMODULE.dataloader]
num_workers = 0
batch_size = 32
pin_memory = true
shuffle = true

[OPTUNA]
enabled = false
n_trials = 40

[OPTUNA.model]
num_layers = {choices = [1, 6, 9]}
dropout = {low = 0.0, high = 0.3, step = 0.05}
hidden_size = {choices = [256, 512, 768]}
min_diag = {choices = [1e-5, 1e-4, 1e-3]}
"""

_EMBEDDED_SPD_FACTORIZED_TOML = """\
[SESSION]
seed = 42
workflow = "train"
precision = "float64"

[MODEL]
name = "ScaleEquivariantEmbeddedSPDFactorizedFFNN"
module_path = "dlkit.nn"
hidden_size = 500
num_layers = 1
bias = false
min_diag = 1e-4
norm = "l2"
dropout = 0.0

[TRAINING.lr_tuner]
min_lr = 1e-5
max_lr = 1e-1
num_training = 100
mode = "linear"

[TRAINING.trainer]
max_epochs = 300
accelerator = "auto"
enable_checkpointing = true

[[TRAINING.trainer.callbacks]]
name = "ModelCheckpoint"
filename = "embedded-spd-factorized"
monitor = "val_loss"
save_top_k = 1
save_weights_only = false
enable_version_counter = false

[[TRAINING.trainer.callbacks]]
name = "EarlyStopping"
monitor = "val_loss"
min_delta = 1e-4
patience = 30
mode = "min"
check_finite = true

[TRAINING.optimizer.default_optimizer]
name = "AdamW"
lr = 1e-3
weight_decay = 1e-2

[TRAINING.optimizer.default_scheduler]
name = "ReduceLROnPlateau"
mode = "min"
factor = 0.5
patience = 10
cooldown = 5
threshold = 1e-4
min_lr = 1e-6

[TRAINING.loss_function]
name = "normalized_vector_norm_loss"
module_path = "dlkit.domain.losses"
target_key = "targets.solutions"

[[TRAINING.metrics]]
name = "NormalizedVectorNormError"
module_path = "dlkit.domain.metrics"
norm_ord = 2
vector_dim = -1

[DATASET]
name = "FlexibleDataset"
memmap_cache = true

[DATAMODULE]
name = "InMemoryModule"

[DATAMODULE.dataloader]
num_workers = 0
batch_size = 32
pin_memory = true
shuffle = true

[OPTUNA]
enabled = false
n_trials = 40

[OPTUNA.model]
num_layers = {choices = [1, 6, 9]}
dropout = {low = 0.0, high = 0.3, step = 0.05}
hidden_size = {choices = [256, 512, 768]}
min_diag = {choices = [1e-5, 1e-4, 1e-3]}
"""


@pytest.fixture
def skip_ffnn_config() -> dict:
    """Embedded copy of the skip-ffnn base model config."""
    return tomllib.loads(_SKIP_FFNN_TOML)


@pytest.fixture
def embedded_factorized_config() -> dict:
    """Embedded copy of the embedded-factorized model config."""
    return tomllib.loads(_EMBEDDED_FACTORIZED_TOML)


@pytest.fixture
def embedded_spd_config() -> dict:
    """Embedded copy of the embedded-spd model config."""
    return tomllib.loads(_EMBEDDED_SPD_TOML)


@pytest.fixture
def embedded_spd_factorized_config() -> dict:
    """Embedded copy of the embedded-spd-factorized model config."""
    return tomllib.loads(_EMBEDDED_SPD_FACTORIZED_TOML)


# ---------------------------------------------------------------------------
# Dataset config tests — stable contracts, load from disk
# ---------------------------------------------------------------------------


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
        "configs/case-advanced.toml",
        "configs/case-ffnn.toml",
        "configs/case-linear.toml",
        "configs/case-parametrized.toml",
    ]

    for registry_path in registry_paths:
        config = _load_toml(registry_path)
        datasets = {entry["id"]: entry["path"] for entry in config["datasets"]}
        assert datasets["residuals-100"] == "datasets/residuals-100.toml"
        assert datasets["residuals-100-gaussian"] == "datasets/residuals-100-gaussian.toml"


def test_residual_experiments_use_residuals_100_dataset() -> None:
    """Every active residual-labelled experiment should use one of the residual datasets."""
    registry_paths = [
        "configs/case-advanced.toml",
        "configs/case-ffnn.toml",
        "configs/case-linear.toml",
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


# ---------------------------------------------------------------------------
# Model config content tests — use embedded fixtures, no disk dependency
# ---------------------------------------------------------------------------


def test_skip_ffnn_config_uses_residual_class(skip_ffnn_config: dict) -> None:
    """skip-ffnn config must use the constant-width factorized residual network."""
    assert skip_ffnn_config["MODEL"]["name"] == "ScaleEquivariantConstantWidthFactorizedFFNN"
    assert "residual" not in skip_ffnn_config["MODEL"]


def test_embedded_factorized_config_uses_residual_class(embedded_factorized_config: dict) -> None:
    """embedded-factorized config must use the embedded factorized residual network."""
    assert embedded_factorized_config["MODEL"]["name"] == "ScaleEquivariantEmbeddedFactorizedFFNN"
    assert "residual" not in embedded_factorized_config["MODEL"]


def test_embedded_spd_config_uses_residual_class(embedded_spd_config: dict) -> None:
    """embedded-spd config must use the embedded SPD residual network."""
    assert embedded_spd_config["MODEL"]["name"] == "ScaleEquivariantEmbeddedSPDFFNN"
    assert "residual" not in embedded_spd_config["MODEL"]


def test_embedded_spd_factorized_config_uses_residual_class(
    embedded_spd_factorized_config: dict,
) -> None:
    """embedded-spd-factorized config must use the embedded SPD-factorized residual network."""
    assert (
        embedded_spd_factorized_config["MODEL"]["name"]
        == "ScaleEquivariantEmbeddedSPDFactorizedFFNN"
    )
    assert "residual" not in embedded_spd_factorized_config["MODEL"]
