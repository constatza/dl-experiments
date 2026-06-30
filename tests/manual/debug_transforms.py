#!/usr/bin/env python
"""Manual utility for inspecting transforms with test-scoped configs."""

from __future__ import annotations

from pathlib import Path
import shutil

import tomli_w

from neuralls.composition.experiments.assembler import load_experiment


def _write_configs(root: Path) -> tuple[Path, Path]:
    matrix_path = root / "matrix.txt"
    matrix_path.write_text("1.0\n", encoding="utf-8")

    data_config = root / "data.toml"
    with data_config.open("wb") as fh:
        tomli_w.dump(
            {
                "id": "debug-data",
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
                "[run]",
                'type = "train"',
                "seed = 42",
                "",
                "[experiment]",
                'name = "debug-model"',
                "",
                "[model]",
                'name = "DebugModel"',
                'module_path = "dlkit.nn"',
                "",
                "[data]",
                'name = "FlexibleDataset"',
                "",
                "[data.module]",
                'name = "ArrayDataModule"',
                "",
                "[training.trainer]",
                "max_epochs = 1",
                "",
                "[training.optimizer.default_optimizer]",
                'name = "AdamW"',
                "lr = 1e-3",
            ]
        ),
        encoding="utf-8",
    )
    return model_config, data_config


def main() -> None:
    """Load a minimal test-scoped experiment and print transform sections."""
    root = Path(__file__).resolve().parent / ".tmp-debug-transforms"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        model_config, data_config = _write_configs(root)
        experiment = load_experiment(
            model_config,
            data_config,
            output_root=root / "output",
            dataset_registry_id="debug-data",
        )

        dataset = experiment.settings.data
        print(f"DATASET: {dataset.name}")
        print(f"FEATURES: {getattr(dataset, 'features', None)}")
        print(f"TARGETS: {getattr(dataset, 'targets', None)}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
