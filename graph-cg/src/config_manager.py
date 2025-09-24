"""Centralized configuration management for graph-cg project."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import typer

from dlkit import GeneralSettings
from dlkit.tools.config.data_entries import Feature, Target
from .validation import validate_file_exists, validate_directory_writable


class ConfigManager:
    """Centralized configuration manager with path resolution and validation."""

    def __init__(self, config_path: str | Path):
        """Initialize config manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = validate_file_exists(config_path, "Config file")
        self._settings: Optional[GeneralSettings] = None

    @property
    def settings(self) -> GeneralSettings:
        """Get loaded settings, loading if necessary."""
        if self._settings is None:
            self._settings = GeneralSettings.from_file(str(self.config_path))
        return self._settings

    def get_matrix_path(self, override: Optional[str | Path] = None) -> Path:
        """Get matrix file path with optional override."""
        if override:
            return validate_file_exists(override, "Matrix file")

        if not self.settings.PATHS or not self.settings.PATHS.matrix_path:
            raise ValueError("Matrix path not specified in config [PATHS] section")
        return validate_file_exists(self.settings.PATHS.matrix_path, "Matrix file")

    def get_rhs_path(self, override: Optional[str | Path] = None) -> Path:
        """Get RHS file path with optional override."""
        if override:
            return validate_file_exists(override, "RHS file")

        if not self.settings.PATHS or not self.settings.PATHS.rhs_path:
            raise ValueError("RHS path not specified in config [PATHS] section")
        return validate_file_exists(self.settings.PATHS.rhs_path, "RHS file")

    def get_checkpoint_path(self, override: Optional[str | Path] = None) -> Optional[Path]:
        """Get checkpoint file path with optional override."""
        if override:
            return validate_file_exists(override, "Checkpoint file")

        # Look in MODEL.checkpoint first, then PATHS.checkpoint_path
        checkpoint_path = None
        if hasattr(self.settings, 'MODEL') and hasattr(self.settings.MODEL, 'checkpoint'):
            checkpoint_path = getattr(self.settings.MODEL, 'checkpoint', None)
        if checkpoint_path is None and self.settings.PATHS and hasattr(self.settings.PATHS, 'checkpoint_path'):
            checkpoint_path = getattr(self.settings.PATHS, 'checkpoint_path', None)

        if checkpoint_path:
            return validate_file_exists(checkpoint_path, "Checkpoint file")
        return None

    def get_output_dir(self, override: Optional[str | Path] = None) -> Path:
        """Get output directory with optional override."""
        if override:
            return validate_directory_writable(override, "Output directory")

        output_path = Path("./output")
        if self.settings.PATHS and self.settings.PATHS.output_dir:
            output_path = Path(self.settings.PATHS.output_dir)
        return validate_directory_writable(output_path, "Output directory")

    def get_training_data_paths(
        self,
        features_override: Optional[str | Path] = None,
        targets_override: Optional[str | Path] = None
    ) -> tuple[Path, Path]:
        """Get training data paths (features, targets) with optional overrides."""
        settings = self.settings

        # Get features path
        if features_override:
            features_path = validate_file_exists(features_override, "Features file")
        else:
            if not settings.DATASET or not settings.DATASET.features:
                raise ValueError("Features path not specified in config [DATASET] section")
            features_path = validate_file_exists(settings.DATASET.features[0].path, "Features file")

        # Get targets path
        if targets_override:
            targets_path = validate_file_exists(targets_override, "Targets file")
        else:
            if not settings.DATASET or not settings.DATASET.targets:
                raise ValueError("Targets path not specified in config [DATASET] section")
            targets_path = validate_file_exists(settings.DATASET.targets[0].path, "Targets file")

        return features_path, targets_path

    def get_data_generation_paths(self) -> tuple[Path, Path, Path, Path]:
        """Get paths for data generation: (matrix, rhs, output_features, output_targets)."""
        matrix_path = self.get_matrix_path()
        rhs_path = self.get_rhs_path()

        # Determine output paths from dataset config
        if not self.settings.DATASET or not self.settings.DATASET.features or not self.settings.DATASET.targets:
            raise ValueError("Dataset paths not configured for data generation")

        features_path = Path(self.settings.DATASET.features[0].path)
        targets_path = Path(self.settings.DATASET.targets[0].path)

        # Ensure parent directories exist
        features_path.parent.mkdir(parents=True, exist_ok=True)
        targets_path.parent.mkdir(parents=True, exist_ok=True)

        return matrix_path, rhs_path, features_path, targets_path

    def update_dataset_paths(
        self,
        features_path: Optional[str | Path] = None,
        targets_path: Optional[str | Path] = None
    ) -> GeneralSettings:
        """Create updated settings with new dataset paths."""
        settings = self.settings
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

        # Update dataset settings
        ds = ds.model_copy(update={"features": feats, "targets": targs})
        return settings.model_copy(update={"DATASET": ds})

    def update_training_paths(
        self,
        output_dir: Optional[str | Path] = None,
        accelerator: Optional[str] = None
    ) -> GeneralSettings:
        """Create updated settings with new training paths."""
        settings = self.settings
        training = settings.TRAINING

        if training is None:
            raise ValueError("Config is missing [TRAINING] section")

        if output_dir is not None or accelerator is not None:
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

        return settings

    def get_solver_params(self) -> Dict[str, Any]:
        """Extract solver parameters from config."""
        extras = self.settings.EXTRAS
        solver_cfg: Dict[str, Any] = {}

        if extras is not None:
            extras_dict = extras.model_dump()
            raw_solver = extras_dict.get("solver", {})
            if isinstance(raw_solver, dict):
                solver_cfg = raw_solver

        # Extract with defaults
        tolerance = solver_cfg.get("tolerance", 1e-8)
        max_iterations = solver_cfg.get("max_iterations", 30)
        normalize_system = solver_cfg.get("normalize_system", True)
        stopping_criterion = solver_cfg.get("stopping_criterion", "tolerance")

        # Type conversion with fallbacks
        try:
            tolerance = float(tolerance)
        except (TypeError, ValueError):
            tolerance = 1e-8

        try:
            max_iterations = int(max_iterations)
        except (TypeError, ValueError):
            max_iterations = 30

        if isinstance(normalize_system, str):
            normalize_system = normalize_system.lower() in {"1", "true", "yes", "on"}

        return {
            "tolerance": tolerance,
            "max_iterations": max_iterations,
            "normalize_system": bool(normalize_system),
            "stopping_criterion": str(stopping_criterion),
        }


def load_config(config_path: str | Path) -> ConfigManager:
    """Load configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)