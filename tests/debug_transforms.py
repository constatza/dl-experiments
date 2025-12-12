#!/usr/bin/env python
"""Debug script to inspect transforms in DLKit settings."""

from __future__ import annotations

from pathlib import Path

from src.configuration import load_config


def main() -> None:
    """Load config and inspect transforms."""
    graph_cg_root = Path(__file__).resolve().parent.parent

    # Load the linear config
    config_path = graph_cg_root / "configs" / "linear.toml"
    data_config_path = graph_cg_root / "data-configs" / "collect-504-solutions.toml"

    print("=" * 80)
    print("LOADING CONFIG AND INSPECTING TRANSFORMS")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Data config: {data_config_path}")
    print()

    settings, context = load_config(config_path, data_config_path)

    print("=" * 80)
    print("DATASET CONFIGURATION")
    print("=" * 80)

    if hasattr(settings, "DATASET") and settings.DATASET:
        dataset = settings.DATASET
        print(f"Dataset name: {dataset.name}")
        print()

        # Check features
        if hasattr(dataset, "features") and dataset.features:
            print(f"Features ({len(dataset.features)} entries):")
            for i, feat in enumerate(dataset.features):
                print(f"  [{i}] name={feat.name}, path={feat.path}")
                if hasattr(feat, "transforms") and feat.transforms:
                    print(f"      Transforms ({len(feat.transforms)}):")
                    for j, t in enumerate(feat.transforms):
                        print(f"        [{j}] {t.name} (module: {t.module_path})")
                        if hasattr(t, "eps"):
                            print(f"            eps={t.eps}")
                else:
                    print("      Transforms: NONE")
        else:
            print("Features: NONE")

        print()

        # Check targets
        if hasattr(dataset, "targets") and dataset.targets:
            print(f"Targets ({len(dataset.targets)} entries):")
            for i, targ in enumerate(dataset.targets):
                print(f"  [{i}] name={targ.name}, path={targ.path}")
                if hasattr(targ, "transforms") and targ.transforms:
                    print(f"      Transforms ({len(targ.transforms)}):")
                    for j, t in enumerate(targ.transforms):
                        print(f"        [{j}] {t.name} (module: {t.module_path})")
                        if hasattr(t, "eps"):
                            print(f"            eps={t.eps}")
                else:
                    print("      Transforms: NONE")
        else:
            print("Targets: NONE")
    else:
        print("DATASET: NOT FOUND IN SETTINGS")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
