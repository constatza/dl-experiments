#!/usr/bin/env python
"""Manual utility for inspecting transforms with test-scoped configs."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import tomli_w

from neuralls.configuration.loader import load_experiment


def _write_configs(root: Path) -> tuple[Path, Path]:
    matrix_path = root / "matrix.txt"
    matrix_path.write_text("1.0\n", encoding="utf-8")

    data_config = root / "data.toml"
    with data_config.open("wb") as fh:
        tomli_w.dump(
            {
                "flow": {"dataset": "debug-data"},
                "source": {"matrix_path": str(matrix_path)},
                "generation": {"normalize": "matrix"},
                "output": {"data_dir": str(root / "processed")},
            },
            fh,
        )

    model_config = root / "model.toml"
    model_config.write_text(
        "\n".join(
            [
                "[SESSION]",
                'name = "debug-model"',
                "",
                "[MODEL]",
                'name = "DebugModel"',
                'module_path = "dlkit.nn.ffnn"',
                "",
                "[TRAINING]",
                "[TRAINING.trainer]",
                "max_epochs = 1",
                "",
                "[DATASET]",
                'name = "FlexibleDataset"',
                "",
                "[MLFLOW]",
                "enabled = false",
            ]
        ),
        encoding="utf-8",
    )
    return model_config, data_config


def main() -> None:
    """Load a minimal test-scoped experiment and print transform sections."""
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        model_config, data_config = _write_configs(root)
        experiment = load_experiment(
            model_config,
            data_config,
            output_root=root / "output",
        )

        dataset = experiment.settings.DATASET
        print(f"DATASET: {dataset.name}")
        print(f"FEATURES: {getattr(dataset, 'features', None)}")
        print(f"TARGETS: {getattr(dataset, 'targets', None)}")


if __name__ == "__main__":
    main()
