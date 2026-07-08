"""Services for creating assignment workspaces.

WorkspaceFactory creates directory structures for assignments.
MLflow lifecycle is handled by dlkit, not here.
"""

from __future__ import annotations

from pathlib import Path

from neuralls.platform.config.models.workspace import AssignmentWorkspace


class WorkspaceFactory:
    """Factory for creating assignment workspaces."""

    def __init__(
        self,
        output_root: Path,
        processed_root: Path,
    ):
        """Initialize with base paths.

        Args:
            output_root: Master output root (SINGLE SOURCE OF TRUTH).
            processed_root: Where processed datasets live.
        """
        self.output_root = output_root
        self.processed_root = processed_root

    def create(
        self,
        dataset_id: str,
        run_id: str,
    ) -> AssignmentWorkspace:
        """Create workspace directory structure.

        Structure created:
            output_root/
              {dataset_id}/
                {run_id}/
                  checkpoints/
                  figures/
                  predictions/

        Args:
            dataset_id: Dataset identifier (e.g., 'test-solutions').
            run_id: Model/run identifier (e.g., 'NormScaledLinearFFNN').

        Returns:
            AssignmentWorkspace with all paths resolved.
        """
        # Assignment root: output_root / dataset_id / run_id
        root_dir = self.output_root / dataset_id / run_id

        # Data directory: processed_root / dataset_id
        data_dir = self.processed_root / dataset_id

        # Create workspace
        workspace = AssignmentWorkspace(
            dataset_id=dataset_id,
            run_id=run_id,
            root_dir=root_dir,
            data_dir=data_dir,
        )

        # Ensure directories exist
        workspace.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        workspace.figures_dir.mkdir(parents=True, exist_ok=True)
        workspace.predictions_dir.mkdir(parents=True, exist_ok=True)

        # Ensure data directory exists
        workspace.data_dir.mkdir(parents=True, exist_ok=True)

        return workspace
