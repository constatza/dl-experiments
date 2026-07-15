"""Preconditioner configuration models.

These Pydantic models validate preconditioner configurations from TOML files.
They support both factory creation and scheduling/comparison concerns.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Any, Annotated
from pathlib import Path

from pydantic import (
    TypeAdapter,
    BeforeValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


class PreconditionerType(StrEnum):
    """Preconditioner types."""

    NONE = "none"
    IDENTITY = "identity"
    JACOBI = "jacobi"
    ILU = "ilu"
    IC0 = "ic0"
    ICHOLESKY = "icholesky"
    NEURAL = "neural"
    AMG = "amg"
    NEURAL_AMG = "neural_amg"


def _normalize_null(data: dict) -> Any:
    """Normalize deprecated null-like aliases to the explicit none baseline."""
    if isinstance(data, dict):
        data = data.copy()
        if data.get("type") == "null":
            data["type"] = PreconditionerType.NONE
        return data
    return data


class RegisteredModelRefConfig(BaseModel):
    """Reference to a model registered in the MLflow Model Registry.

    Attributes:
        source: Discriminator field, always "registered".
        name: Registered model name when explicitly provided.
        alias: Model alias (e.g. ``"@solutions"``); ``@`` prefix is stripped.
        version: Explicit model version number.
        latest: If True, select the highest registered version number. This is
            "most recently registered," not "best-performing" — there is no
            metric-based (e.g. lowest validation loss) selection. Use an
            explicit `alias` pointed at a deliberately-chosen version if you
            need a stable, quality-vetted reference.
    """

    source: Literal["registered"] = "registered"
    name: str | None = Field(default=None, min_length=1)
    alias: str | None = None
    version: int | None = Field(default=None, ge=1)
    latest: bool | None = None
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_selector(self) -> RegisteredModelRefConfig:
        """Validate that exactly one selector is provided.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If not exactly one of alias, version, or latest=True is set.
        """
        selectors = [self.alias is not None, self.version is not None, self.latest is True]
        if sum(selectors) != 1:
            raise ValueError("Requires exactly one selector: alias, version, or latest=true.")
        return self


class LoggedModelRefConfig(BaseModel):
    """Reference to a model logged within an MLflow run.

    Attributes:
        source: Discriminator field, always "logged".
        run_id: Explicit MLflow run ID to reference.
        latest: If True, select the most recently *started* run matching the
            given filters (ordered by ``attributes.start_time DESC``). This is
            "most recent," not "best-performing" — there is no metric-based
            (e.g. lowest validation loss) selection.
        model_name: Filter by logged model name.
        experiment_name: Filter by experiment name.
        experiment_id: Filter by experiment ID.
        run_name: Filter by run name.
        artifact_path: Artifact path within the run (default: "model").
        tags: Optional tag filters.
    """

    source: Literal["logged"] = "logged"
    run_id: str | None = Field(default=None, min_length=1)
    latest: bool | None = None
    model_name: str | None = None
    experiment_name: str | None = None
    experiment_id: str | None = None
    run_name: str | None = None
    artifact_path: str = "model"
    tags: dict[str, str] | None = None
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_selector(self) -> LoggedModelRefConfig:
        """Validate that exactly one selection mode is active.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If not exactly one of run_id or latest=True is set.
        """
        selectors = [self.run_id is not None, self.latest is True]
        if sum(selectors) != 1:
            raise ValueError("Requires exactly one selector: run_id or latest=true.")
        return self

    @model_validator(mode="after")
    def validate_latest_filters(self) -> LoggedModelRefConfig:
        """Validate that latest=True is accompanied by at least one filter.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If latest=True but no filter fields are set.
        """
        if self.latest is not True:
            return self
        filters = [
            self.model_name,
            self.experiment_name,
            self.experiment_id,
            self.run_name,
            self.tags,
        ]
        if not any(f is not None for f in filters):
            raise ValueError(
                "latest=true requires at least one filter "
                "(model_name, experiment_name, experiment_id, run_name, or tags)."
            )
        return self


ModelRefConfig = Annotated[
    RegisteredModelRefConfig | LoggedModelRefConfig,
    Field(discriminator="source"),
]


class BasePreconditionerConfig(BaseModel):
    """Shared fields for all preconditioners.

    Includes scheduling fields shared by all preconditioner variants.
    """

    name: str
    start_iter: int = Field(
        default=0,
        ge=0,
        description="Iteration at which the primary preconditioner becomes active.",
    )
    limit_iters: int = Field(default=-1, description="Iterations to apply; -1 means unlimited.")
    fallback: PreconditionerType = Field(
        default=PreconditionerType.IDENTITY,
        description="Fallback preconditioner type when limited.",
    )

    model_config = ConfigDict(
        extra="ignore",  # Allow extra fields from conversion
        frozen=True,
    )


class StandardPreconditionerConfig(BasePreconditionerConfig):
    """Non-parametric, static preconditioners (identity, jacobi, ilu, icholesky)."""

    type: Literal[
        PreconditionerType.NONE,
        PreconditionerType.IDENTITY,
        PreconditionerType.JACOBI,
        PreconditionerType.ILU,
        PreconditionerType.ICHOLESKY,
    ]


class IC0PreconditionerConfig(BasePreconditionerConfig):
    """IC(0) preconditioner configuration with threshold parameter."""

    type: Literal[PreconditionerType.IC0] = PreconditionerType.IC0
    threshold: float = Field(
        default=1e-14,
        description="Drop tolerance - entries with |value| < threshold are treated as zeros",
    )


class NeuralPreconditionerConfig(BasePreconditionerConfig):
    """Neural preconditioner configuration."""

    type: Literal[PreconditionerType.NEURAL] = PreconditionerType.NEURAL
    checkpoint_path: Path | None = None
    assignment: str | None = None
    config_path: Path | None = None
    data_config_path: Path | None = None
    model_ref: ModelRefConfig | None = None
    resolved_checkpoint_path: Path | None = None
    extra_input_names: tuple[str, ...] = Field(
        default=(),
        description=(
            "Names of extra dataset arrays to bind before CG, beyond the residual. "
            "E.g. ('matrix',) for the stiffness matrix, ('coordinates',) for node coords. "
            "The comparison workflow loads matching named arrays from the dataset directory."
        ),
    )

    @field_validator("checkpoint_path", "config_path", "data_config_path", mode="before")
    @classmethod
    def _expand_paths(cls, v: object, info: ValidationInfo) -> object:
        """Expand ${NEURALLS_*} placeholders and resolve neural preconditioner paths.

        Args:
            v: Raw value from config field.
            info: Pydantic validation info carrying context.

        Returns:
            Resolved absolute path string, or original value if not a string.
        """
        if v is None or info.context is None or not isinstance(v, str):
            return v
        from neuralls.platform.config.context import ConfigContext, expand_config_path

        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))

    @field_validator("extra_input_names", mode="before")
    @classmethod
    def _coerce_extra_names_to_tuple(cls, v: object) -> tuple[str, ...] | object:
        """Coerce list to tuple for extra_input_names field.

        Args:
            v: Raw value that may be list, tuple, or other type.

        Returns:
            Tuple of strings, or original value if not list/tuple.
        """
        if isinstance(v, (list, tuple)):
            return tuple(str(x) for x in v)
        return v


class AggregationCoarseningConfig(BaseModel):
    """Classical smoothed-aggregation coarsening (SA-AMG)."""

    method: Literal["aggregation"] = "aggregation"
    omega: float = Field(default=0.67, gt=0.0, description="Prolongation Jacobi-smoothing damping.")
    model_config = ConfigDict(extra="forbid", frozen=True)


class PODCoarseningConfig(BaseModel):
    """POD-2G coarsening (Nikolopoulos et al. 2022, §3.3-3.5).

    The prolongation/restriction operator is a POD basis fit to a snapshot
    ensemble of high-fidelity solutions (or, for a sharper coarse space,
    CG error traces e_k = x* - x_k, which are richer than plain solutions
    since e_0 == x* and later k emphasize slow-converging directions) —
    read from an already-generated dataset directory, not raw files, so the
    same validation/normalization/manifest guarantees as every other
    dataset in this repo apply.
    """

    method: Literal["pod"] = "pod"
    dataset_dir: Path = Field(
        ...,
        description="Generated dataset directory whose `solutions` array supplies POD snapshots.",
    )
    n_snapshots: int = Field(
        default=-1, description="Number of snapshot files to load; -1 means all matched."
    )
    rank: int | float = Field(
        default=8,
        description=(
            "Fixed number of POD modes to retain (int), or minimum cumulative "
            "captured energy to retain (float in (0, 1] — e.g. 0.9999)."
        ),
    )
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("dataset_dir", mode="before")
    @classmethod
    def _expand_dataset_dir(cls, v: object, info: ValidationInfo) -> object:
        """Expand ``${NEURALLS_*}`` placeholders and anchor the dataset directory.

        Args:
            v: Raw field value.
            info: Pydantic validation context.

        Returns:
            Resolved path string, or original value if not a string.
        """
        from neuralls.platform.config.context import expand_path_field

        return expand_path_field(v, info)

    @field_validator("rank")
    @classmethod
    def _validate_rank(cls, v: int | float) -> int | float:
        """Enforce the per-mode constraint matching whichever branch was matched.

        Args:
            v: The parsed ``rank`` value — an int (mode count) or float
                (energy threshold).

        Returns:
            The validated value, unchanged.

        Raises:
            ValueError: If an int rank is < 1, or a float rank is outside (0, 1].
        """
        if isinstance(v, float):
            if not (0.0 < v <= 1.0):
                raise ValueError(f"rank as an energy threshold must be in (0, 1], got {v}")
        elif v < 1:
            raise ValueError(f"rank as a mode count must be >= 1, got {v}")
        return v


CoarseningConfig = Annotated[
    AggregationCoarseningConfig | PODCoarseningConfig,
    Field(discriminator="method"),
]


class AMGPreconditionerConfig(BasePreconditionerConfig):
    """AMG-family preconditioner configuration (multigrid coarsening + cycle).

    ``coarsening`` selects the strategy that builds the prolongation/
    restriction operator — ``AggregationCoarseningConfig`` (classical SA-AMG)
    or ``PODCoarseningConfig`` (POD-2G). Required, no default: the caller must
    state which coarsening strategy an AMG config means. A future coarsening
    strategy (e.g. a hierarchical multi-level POD) is added the same way: one
    more class in the ``CoarseningConfig`` union, not a new
    ``PreconditionerType`` and not new fields on this class — the underlying
    ``AMGPreconditioner`` is already generic over any ``CoarseningStrategy``.
    """

    type: Literal[PreconditionerType.AMG] = PreconditionerType.AMG
    n_levels: int = Field(default=2, ge=2, description="Total number of grid levels.")
    pre_smoothing_steps: int = Field(default=2, ge=0, description="Pre-smoothing iterations.")
    post_smoothing_steps: int = Field(default=2, ge=0, description="Post-smoothing iterations.")
    smoother_omega: float = Field(default=0.67, gt=0.0, description="Weighted Jacobi damping.")
    coarsening: CoarseningConfig


class NeuralTransferConfig(BaseModel):
    """Config for one neural transfer operator (prolongation or restriction).

    Required inputs (the extra arrays the network expects beyond its vector input)
    are NOT declared here — they are read at runtime from
    ``ExtraInputPredictorPort.required_inputs``, which is populated by the adapter
    from the model config.  This eliminates duplication with the DLKit TOML.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)
    checkpoint_path: Path | None = None
    model_ref: ModelRefConfig | None = None
    config_path: Path | None = None
    data_config_path: Path | None = None
    resolved_checkpoint_path: Path | None = None

    @field_validator("checkpoint_path", "config_path", "data_config_path", mode="before")
    @classmethod
    def _expand_paths(cls, v: object, info: ValidationInfo) -> object:
        """Expand ``${NEURALLS_*}`` placeholders in path fields.

        Args:
            v: Raw field value.
            info: Pydantic validation context.

        Returns:
            Resolved path string, or original value if not a string.
        """
        if v is None or info.context is None or not isinstance(v, str):
            return v
        from neuralls.platform.config.context import ConfigContext, expand_config_path

        return expand_config_path(v, ConfigContext.from_pydantic_context(info.context))


class NeuralAMGPreconditionerConfig(BasePreconditionerConfig):
    """AMG preconditioner that uses neural networks for prolongation/restriction.

    The ``extra_input_names`` required by each network are NOT declared here —
    they are read at runtime from ``ExtraInputPredictorPort.required_inputs`` so
    that the DLKit model config remains the single source of truth.
    """

    type: Literal[PreconditionerType.NEURAL_AMG] = PreconditionerType.NEURAL_AMG
    n_levels: int = Field(default=2, ge=2, description="Total number of grid levels.")
    pre_smoothing_steps: int = Field(default=2, ge=0)
    post_smoothing_steps: int = Field(default=2, ge=0)
    smoother_omega: float = Field(default=0.67, gt=0.0)
    aggregation_omega: float = Field(default=0.67, gt=0.0)
    prolongation: NeuralTransferConfig
    restriction: NeuralTransferConfig | None = None


ConcretePreconditionerConfig = (
    StandardPreconditionerConfig
    | IC0PreconditionerConfig
    | NeuralPreconditionerConfig
    | AMGPreconditionerConfig
    | NeuralAMGPreconditionerConfig
)

_StrictPreconditionerConfig = Annotated[
    ConcretePreconditionerConfig,
    Field(discriminator="type"),
]

PreconditionerConfig = Annotated[
    _StrictPreconditionerConfig,
    BeforeValidator(_normalize_null),
]

parse_preconditioner_config = TypeAdapter(PreconditionerConfig).validate_python
