#!/usr/bin/env python3
"""Training script for graph-cg using src/ library modules and dlkit API."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import typer

from dlkit import train
from dlkit.tools.config.data_entries import Feature, Target

from src.common import load_config, get_paths_from_config, get_latest_checkpoint
from src.validation import validate_file_exists, validate_directory_writable


def train_model(
    config_path: str | Path,
    features_path: Optional[str | Path] = None,
    targets_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    accelerator: Optional[str] = None
) -> Path:
    """Train model programmatically.

    Args:
        config_path: Path to config file
        features_path: Override features path
        targets_path: Override targets path
        output_dir: Override output directory
        accelerator: Override accelerator

    Returns:
        Path to saved checkpoint
    """
    # Load config
    settings = load_config(config_path)

    # Update dataset paths if provided
    if features_path is not None or targets_path is not None:
        ds = settings.DATASET
        if ds is None:
            raise ValueError("Config is missing [DATASET] section")

        feats = ds.features
        targs = ds.targets

        if features_path is not None:
            features_path = validate_file_exists(features_path, "Features file")
            if feats and len(feats) > 0:
                new_feat = Feature(name=feats[0].name, path=str(features_path))
                feats = (new_feat,)
            else:
                feats = (Feature(name="x", path=str(features_path)),)

        if targets_path is not None:
            targets_path = validate_file_exists(targets_path, "Targets file")
            if targs and len(targs) > 0:
                new_targ = Target(name=targs[0].name, path=str(targets_path))
                targs = (new_targ,)
            else:
                targs = (Target(name="y", path=str(targets_path)),)

        ds = ds.model_copy(update={"features": feats, "targets": targs})
        settings = settings.model_copy(update={"DATASET": ds})

    # Update training settings if provided
    if output_dir is not None or accelerator is not None:
        training = settings.TRAINING
        if training is None:
            raise ValueError("Config is missing [TRAINING] section")

        trainer = training.trainer

        if output_dir is not None:
            output_dir = validate_directory_writable(output_dir, "Output directory")
            trainer = trainer.model_copy(update={"default_root_dir": str(output_dir)})

            # Update checkpoint callback directory
            callbacks = trainer.callbacks or []
            updated_callbacks = []
            for cb in callbacks:
                if hasattr(cb, 'dirpath') and cb.dirpath:
                    new_dirpath = Path(output_dir) / "checkpoints"
                    cb = cb.model_copy(update={"dirpath": str(new_dirpath)})
                updated_callbacks.append(cb)
            trainer = trainer.model_copy(update={"callbacks": updated_callbacks})

        if accelerator is not None:
            trainer = trainer.model_copy(update={"accelerator": accelerator})

        training = training.model_copy(update={"trainer": trainer})
        settings = settings.model_copy(update={"TRAINING": training})

    # Execute training
    train(settings)

    # Find and return checkpoint path
    paths = get_paths_from_config(settings)
    output_path = Path(paths.get('output_dir', './output'))
    checkpoint_dir = output_path / "checkpoints"

    checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        raise RuntimeError(f"No checkpoint found in {checkpoint_dir}")

    return checkpoint_path


def main(
    config: Path = typer.Option(Path(__file__).parent / "config-ffnn.toml", help="Path to TOML config"),
    features: Path | None = typer.Option(
        None, help="Override path to features (RHS) .npy"
    ),
    targets: Path | None = typer.Option(
        None, help="Override path to targets (solution) .npy"
    ),
    out_dir: Path | None = typer.Option(
        None, help="Override Trainer default_root_dir and checkpoint dir"
    ),
    accelerator: str | None = typer.Option(
        None, help="Override accelerator: cpu|gpu|auto|tpu"
    ),
):
    """Train model with optional dataset/output overrides."""
    print(f"Loading configuration from: {config}")

    try:
        checkpoint_path = train_model(
            config_path=config,
            features_path=features,
            targets_path=targets,
            output_dir=out_dir,
            accelerator=accelerator
        )

        print("Training completed successfully!")
        print(f"Checkpoint saved to: {checkpoint_path}")

    except Exception as e:
        print(f"Training failed: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(130)
