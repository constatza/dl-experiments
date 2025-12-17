"""Central path orchestration helpers for graph-cg flows.

The design follows three principles:
- All directories derive from a small set of project roots.
- Each flow gets an isolated namespace via ``flow_id``.
- Stages (data, train, predict, compare) compose ``flow_id`` with
  ``dataset_id``/``run_id`` so collisions are impossible by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

from ..constants import (
    DEFAULT_FIGURES_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROCESSED_DATA_DIR,
    DEFAULT_PROJECT_ROOT,
)


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectRoots:
    """Base directories shared by all flows."""

    project_root: Path
    processed_root: Path
    output_root: Path
    figures_root: Path

    @classmethod
    def from_overrides(
        cls,
        *,
        project_root: str | Path | None = None,
        processed_root: str | Path | None = None,
        output_root: str | Path | None = None,
        figures_root: str | Path | None = None,
    ) -> ProjectRoots:
        base = Path(project_root or DEFAULT_PROJECT_ROOT)
        processed = Path(processed_root or DEFAULT_PROCESSED_DATA_DIR)
        output = Path(output_root or DEFAULT_OUTPUT_DIR)
        figures = Path(figures_root or DEFAULT_FIGURES_DIR)
        return cls(
            project_root=base,
            processed_root=processed,
            output_root=output,
            figures_root=figures,
        )

    @property
    def checkpoints_root(self) -> Path:
        return self.output_root / "checkpoints"


@dataclass(frozen=True)
class FlowPaths:
    """Namespace that isolates directories for a specific flow."""

    flow_id: str
    roots: ProjectRoots

    @property
    def processed_root(self) -> Path:
        return self.roots.processed_root

    @property
    def output_root(self) -> Path:
        return self.roots.output_root / self.flow_id

    @property
    def figures_root(self) -> Path:
        return self.roots.figures_root / self.flow_id

    def stage_dir(self, stage: str) -> Path:
        return self.output_root / stage


@dataclass(frozen=True)
class DataPaths:
    """Paths for generated datasets and RHS archives."""

    flow: FlowPaths
    dataset_id: str

    @property
    def base_dir(self) -> Path:
        return self.flow.processed_root / self.dataset_id

    @property
    def features_file(self) -> Path:
        """Path to features data (RHS samples in normalized.npz)."""
        return self.base_dir / "normalized.npz"

    @property
    def targets_file(self) -> Path:
        """Path to targets data (solutions in normalized.npz)."""
        return self.base_dir / "normalized.npz"

    @property
    def matrix_file(self) -> Path:
        """Path to dataset directory for matrix loading."""
        return self.base_dir

    @property
    def mother_rhs_file(self) -> Path:
        """Path to dataset directory for RHS loading."""
        return self.base_dir

    @property
    def normalization_file(self) -> Path:
        return self.base_dir / "normalization.json"

    @property
    def metadata_file(self) -> Path:
        return self.base_dir / "metadata.json"


@dataclass(frozen=True)
class TrainingPaths:
    """Directories for training artefacts tied to a dataset/run."""

    data: DataPaths
    run_id: str

    @property
    def base_dir(self) -> Path:
        return self.data.flow.roots.output_root / self.data.dataset_id / self.run_id

    @property
    def checkpoint_dir(self) -> Path:
        return self.base_dir / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.base_dir / "metrics"


@dataclass(frozen=True)
class PredictionPaths:
    """Locations for inference outputs produced from a training run."""

    training: TrainingPaths

    @property
    def base_dir(self) -> Path:
        return (
            self.training.data.flow.stage_dir("predict") / self.training.data.dataset_id
        )

    @property
    def predictions_file(self) -> Path:
        return self.base_dir / f"{self.training.run_id}-predictions.npz"

    @property
    def parity_plot(self) -> Path:
        return self.base_dir / f"{self.training.run_id}-parity.png"


@dataclass(frozen=True)
class ComparisonPaths:
    """Directories for solver and preconditioner comparisons."""

    flow: FlowPaths
    dataset_id: str

    @property
    def base_dir(self) -> Path:
        return self.flow.stage_dir("compare") / self.dataset_id

    @property
    def report_dir(self) -> Path:
        return self.base_dir / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.flow.figures_root / self.dataset_id


@dataclass(frozen=True)
class FlowContext:
    """Aggregate of all path structures for a flow."""

    flow: FlowPaths
    data: DataPaths
    training: TrainingPaths
    prediction: PredictionPaths
    comparison: ComparisonPaths
    run_id: str
    test_rhs: Path | None = None
    test_matrix: Path | None = None
    test_solutions_path: str | None = None

    def with_run_id(self, run_id: str) -> FlowContext:
        training = TrainingPaths(data=self.data, run_id=run_id)
        prediction = PredictionPaths(training=training)
        return FlowContext(
            flow=self.flow,
            data=self.data,
            training=training,
            prediction=prediction,
            comparison=self.comparison,
            run_id=run_id,
            test_rhs=self.test_rhs,
            test_matrix=self.test_matrix,
            test_solutions_path=self.test_solutions_path,
        )


# ---------------------------------------------------------------------------
# Helpers for reading configs
# ---------------------------------------------------------------------------


def parse_flow_keys(
    config: Mapping[str, object], config_path: Path | str | None = None
) -> tuple[str, str]:
    """Extract ``(flow_id, dataset_id)`` from a config mapping.

    Accepts either lowercase ``flow``/``dataset`` or uppercase variants
    to stay flexible across existing TOML styles.

    The flow.dataset field is now DEPRECATED and optional. If not provided,
    the dataset_id is derived from the config filename stem.

    The flow.id field is also optional - if not provided, dataset_id is used
    as the flow_id. This simplifies configs where the dataset name itself
    is a sufficient identifier.

    Args:
        config: Configuration dictionary containing [flow] section
        config_path: Optional path to config file (used to derive dataset_id from filename)

    Returns:
        Tuple of (flow_id, dataset_id)

    Raises:
        ValueError: If [flow] section is missing or malformed, or if dataset_id
            cannot be determined from either config or filename
    """
    flow_section = config.get("flow") or config.get("FLOW") or {}
    if not isinstance(flow_section, Mapping):
        raise ValueError("[flow] section must be a mapping")

    # dataset field is now optional - derive from config filename if not provided
    dataset_id = flow_section.get("dataset")

    if not isinstance(dataset_id, str) or not dataset_id:
        # Fall back to config filename stem
        if config_path is not None:
            dataset_id = Path(config_path).stem
        else:
            raise ValueError(
                "Missing flow.dataset and no config_path provided. "
                "Either specify flow.dataset in config or provide config_path parameter."
            )

    # Use dataset as flow_id if id is not provided
    flow_id = flow_section.get("id") or dataset_id

    return flow_id, dataset_id


__all__ = [
    "ProjectRoots",
    "FlowPaths",
    "DataPaths",
    "TrainingPaths",
    "PredictionPaths",
    "ComparisonPaths",
    "parse_flow_keys",
]
