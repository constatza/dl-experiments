"""Configuration services for path resolution and workspace creation.

This module contains the logic for calculating directory structures, separating
path generation policies from data containers.
"""

from __future__ import annotations

from pathlib import Path

from .domain import ExperimentWorkspace


class WorkspaceFactory:
    """Factory for creating experiment workspaces."""

    def __init__(self, global_output_dir: Path, processed_root: Path):
        """Initialize with the global root for all artifacts."""
        self.base_dir = global_output_dir
        self.processed_root = processed_root

    def create(self, data_id: str, model_name: str) -> ExperimentWorkspace:
        """
        Calculate the nested directory structure for an experiment.
        
        Structure: global_output / data_id / model_name
        
        Args:
            data_id: Identifier for the dataset (e.g., 'test-solutions')
            model_name: Identifier for the model run (e.g., 'NormScaledFFNN')
            
        Returns:
            Resolved ExperimentWorkspace with all artifact paths.
        """
        experiment_root = self.base_dir / data_id / model_name
        data_dir = self.processed_root / data_id
        
        return ExperimentWorkspace(
            root_dir=experiment_root,
            data_dir=data_dir,
            checkpoint_dir=experiment_root / "checkpoints",
            figures_dir=experiment_root / "figures",
            predictions_dir=experiment_root / "predictions",
            run_id=model_name
        )
