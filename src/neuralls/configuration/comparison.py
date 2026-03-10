"""Comparison configuration dataclasses and strict parser models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field
from pydantic import model_validator

from neuralls.constants import (
    DEFAULT_RTOL,
    DEFAULT_ATOL,
    DEFAULT_M_MAX,
)
from neuralls.configuration.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerConfig,
    RegisteredModelRefConfig,
)


NormalizeSystem: TypeAlias = Literal[
    "none", "matrix", "rhs", "both", "diagonal", "spectral"
]


@dataclass(frozen=True)
class SolverParams:
    """Numerical solver parameters."""

    rtol: float
    atol: float
    max_iterations: int
    stopping_criterion: Literal["residual_norm", "fixed_iterations"]
    m_max: int
    breakdown_tol: float | None


@dataclass(frozen=True)
class ComparisonData:
    """Input data and normalization context for a comparison run."""

    matrix_path: Path
    rhs_path: Path
    rhs_index: int = 0
    dataset_alias: str | None = None
    normalize_system: NormalizeSystem = "matrix"


@dataclass(frozen=True)
class ComparisonGeneral:
    """Grouped comparison context."""

    params: SolverParams
    data: ComparisonData


@dataclass(frozen=True)
class ComparisonConfig:
    """Runtime domain model for comparison config."""

    general: ComparisonGeneral
    preconditioners: tuple[PreconditionerConfig, ...]


class _SolverParamsModel(BaseModel):
    rtol: float = Field(default=DEFAULT_RTOL, gt=0.0)
    atol: float = Field(default=DEFAULT_ATOL, ge=0.0)
    max_iterations: int = Field(default=100, ge=1)
    stopping_criterion: Literal["residual_norm", "fixed_iterations"] = "residual_norm"
    m_max: int = Field(default=DEFAULT_M_MAX, ge=-1)
    breakdown_tol: float | None = Field(default=None, ge=0.0)
    model_config = ConfigDict(extra="forbid", frozen=True)


class _ComparisonDataModel(BaseModel):
    matrix_path: Path
    rhs_path: Path
    rhs_index: int = 0
    dataset_alias: str | None = None
    normalize_system: NormalizeSystem = "matrix"
    model_config = ConfigDict(extra="forbid", frozen=True)


class _ComparisonGeneralModel(BaseModel):
    params: _SolverParamsModel
    data: _ComparisonDataModel
    model_config = ConfigDict(extra="forbid", frozen=True)


class _ComparisonConfigModel(BaseModel):
    general: _ComparisonGeneralModel
    preconditioners: list[PreconditionerConfig]
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_preconditioners(self) -> _ComparisonConfigModel:
        if not self.preconditioners:
            raise ValueError("Comparison config must define at least one [[preconditioners]] entry.")
        names = [spec.name for spec in self.preconditioners]
        if len(names) != len(set(names)):
            raise ValueError("Comparison config cannot define duplicate preconditioner names.")
        neural_specs = [
            spec for spec in self.preconditioners if isinstance(spec, NeuralPreconditionerConfig)
        ]
        if not neural_specs:
            return self

        for spec in neural_specs:
            if spec.model_ref is None:
                raise ValueError(
                    f"Neural preconditioner '{spec.name}' must define model_ref."
                )
            if spec.checkpoint_path is not None:
                raise ValueError(
                    f"Neural preconditioner '{spec.name}' cannot use checkpoint_path in comparison configs."
                )
            if (
                isinstance(spec.model_ref, RegisteredModelRefConfig)
                and spec.model_ref.name is None
                and spec.experiment is None
            ):
                raise ValueError(
                    f"Neural preconditioner '{spec.name}' must define experiment when "
                    "registered model_ref.name is omitted."
                )

        requires_dataset_alias = any(
            isinstance(spec.model_ref, RegisteredModelRefConfig)
            and spec.model_ref.alias is not None
            and spec.model_ref.alias.strip() == "@dataset"
            and spec.experiment is None
            for spec in neural_specs
        )
        if requires_dataset_alias and self.general.data.dataset_alias is None:
            raise ValueError(
                "general.data.dataset_alias is required when model_ref alias is '@dataset' "
                "and no neural preconditioner experiment binding is provided."
            )
        return self


def parse_comparison_config(raw: dict[str, object]) -> ComparisonConfig:
    """Parse the current comparison TOML payload."""
    parsed = _ComparisonConfigModel.model_validate(raw)
    return ComparisonConfig(
        general=ComparisonGeneral(
            params=SolverParams(
                rtol=parsed.general.params.rtol,
                atol=parsed.general.params.atol,
                max_iterations=parsed.general.params.max_iterations,
                stopping_criterion=parsed.general.params.stopping_criterion,
                m_max=parsed.general.params.m_max,
                breakdown_tol=parsed.general.params.breakdown_tol,
            ),
            data=ComparisonData(
                matrix_path=parsed.general.data.matrix_path,
                rhs_path=parsed.general.data.rhs_path,
                rhs_index=parsed.general.data.rhs_index,
                dataset_alias=parsed.general.data.dataset_alias,
                normalize_system=parsed.general.data.normalize_system,
            ),
        ),
        preconditioners=tuple(parsed.preconditioners),
    )
