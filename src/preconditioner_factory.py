"""Factory layer for preconditioners with typed configs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
from pydantic import BaseModel, Field, ValidationError
from scipy.sparse.linalg import LinearOperator, spilu

from .math_utils import _to_csc
from .neural_inference import create_neural_preconditioner

PreconditionerFn = Callable[[np.ndarray], np.ndarray]


class BasePreconditionerConfig(BaseModel):
    """Shared fields for all preconditioners."""

    name: str
    limit_iters: int = Field(
        default=-1, description="Iterations to apply; -1 means unlimited."
    )
    fallback: str = Field(
        default="identity", description="Fallback preconditioner name when limited."
    )

    class Config:
        extra = "ignore"
        frozen = True


class NonePreconditionerConfig(BasePreconditionerConfig):
    type: Literal["none"] = "none"


class IdentityPreconditionerConfig(BasePreconditionerConfig):
    type: Literal["identity"] = "identity"


class JacobiPreconditionerConfig(BasePreconditionerConfig):
    type: Literal["jacobi"] = "jacobi"


class ILUPreconditionerConfig(BasePreconditionerConfig):
    type: Literal["ilu"] = "ilu"


class NeuralPreconditionerConfig(BasePreconditionerConfig):
    type: Literal["neural"] = "neural"
    checkpoint_path: Path
    config_path: Path | None = None
    data_config_path: Path | None = None
    limit_iters: int = Field(
        default=-1, description="Iterations to apply; -1 means unlimited."
    )


PreconditionerConfig = Union[
    NonePreconditionerConfig,
    IdentityPreconditionerConfig,
    JacobiPreconditionerConfig,
    ILUPreconditionerConfig,
    NeuralPreconditionerConfig,
]

_CONFIG_BY_TYPE: dict[str, type[PreconditionerConfig]] = {
    "none": NonePreconditionerConfig,
    "identity": IdentityPreconditionerConfig,
    "jacobi": JacobiPreconditionerConfig,
    "ilu": ILUPreconditionerConfig,
    "neural": NeuralPreconditionerConfig,
}


def parse_preconditioner_config(entry: dict[str, Any]) -> BasePreconditionerConfig:
    """Parse a raw mapping into a typed preconditioner config."""
    precond_type = str(entry.get("type", "none"))
    model = _CONFIG_BY_TYPE.get(precond_type)
    if model is None:
        raise ValueError(f"Unknown preconditioner type: {precond_type}")
    try:
        return model(**entry)
    except ValidationError as exc:
        raise ValueError(
            f"Invalid preconditioner config for '{entry.get('name')}': {exc}"
        ) from exc


def build_preconditioner_configs(
    specs: Sequence[dict[str, Any]],
) -> list[BasePreconditionerConfig]:
    return [parse_preconditioner_config(spec) for spec in specs]


def make_identity_preconditioner() -> PreconditionerFn:
    return lambda x: x


def make_jacobi_preconditioner(A: np.ndarray) -> PreconditionerFn:
    diag = np.diag(A)
    eps = 1e-8
    diag_safe = np.where(np.abs(diag) > eps, np.abs(diag), eps)
    diag_inv = 1.0 / diag_safe
    return lambda x: diag_inv * x


def make_ilu_preconditioner(A: np.ndarray) -> PreconditionerFn:
    A_csc = _to_csc(A)
    ilu = spilu(A_csc)
    ilu_dtype = np.asarray(A, copy=False).dtype
    return lambda x: ilu.solve(x.astype(ilu_dtype, copy=False))


def make_preconditioner_operator(
    factory: Callable[[np.ndarray], PreconditionerFn], A: np.ndarray
) -> LinearOperator:
    return _to_linear_operator(factory(A), A)


def make_neural_preconditioner(
    A: np.ndarray,
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
) -> PreconditionerFn:
    predictor = create_neural_preconditioner(
        checkpoint_path,
        config_path,
        data_config_path,
    )
    matrix = np.asarray(A, dtype=np.float64, copy=False)

    def _precondition(residual: np.ndarray) -> np.ndarray:
        residual_vec = np.asarray(residual, dtype=np.float64, copy=False)
        solution = predictor(matrix, residual_vec)
        return np.asarray(solution, dtype=np.float64, copy=False)

    return _precondition


def _to_linear_operator(
    matvec: PreconditionerFn,
    A: np.ndarray,
) -> LinearOperator:
    array = np.asarray(A)
    dtype = array.dtype  # type: ignore[reportAttributeAccessIssue]
    return LinearOperator(
        shape=array.shape,
        dtype=dtype,
        matvec=lambda x: matvec(np.asarray(x, dtype=dtype, copy=False)),
    )


BuilderFn = Callable[[np.ndarray, BasePreconditionerConfig], PreconditionerFn]


class PreconditionerFactory:
    """Central factory dispatching builds by preconditioner type."""

    def __init__(self) -> None:
        self._registry: dict[str, BuilderFn] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register("none", lambda A, _cfg: make_identity_preconditioner())
        self.register("identity", lambda A, _cfg: make_identity_preconditioner())
        self.register("jacobi", lambda A, _cfg: make_jacobi_preconditioner(A))
        self.register("ilu", lambda A, _cfg: make_ilu_preconditioner(A))
        self.register(
            "neural",
            lambda A, cfg: make_neural_preconditioner(
                A,
                checkpoint_path=cfg.checkpoint_path,  # type: ignore[attr-defined]
                config_path=getattr(cfg, "config_path", None),
                data_config_path=getattr(cfg, "data_config_path", None),
            ),
        )

    def register(self, type_name: str, builder: BuilderFn) -> None:
        self._registry[type_name] = builder

    def create(self, A: np.ndarray, cfg: BasePreconditionerConfig) -> PreconditionerFn:
        builder = self._registry.get(cfg.type)
        if builder is None:
            raise ValueError(f"Unsupported preconditioner type: {cfg.type}")
        return builder(A, cfg)

    def create_many(
        self, A: np.ndarray, configs: Sequence[BasePreconditionerConfig]
    ) -> tuple[dict[str, PreconditionerFn], dict[str, str]]:
        preconditioners: dict[str, PreconditionerFn] = {}
        solver_types: dict[str, str] = {}
        for cfg in configs:
            precond_fn = self.create(A, cfg)
            preconditioners[cfg.name] = precond_fn
            solver_types[cfg.name] = cfg.type
        return preconditioners, solver_types


def build_preconditioners(
    A: np.ndarray, configs: Sequence[BasePreconditionerConfig]
) -> tuple[dict[str, PreconditionerFn], dict[str, str]]:
    return PreconditionerFactory().create_many(A, configs)


def create_all_preconditioners(
    A: np.ndarray,
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
    *,
    pca_path: str | Path | None = None,
) -> tuple[dict[str, PreconditionerFn]]:
    """Maintain a minimal compatibility helper for default preconditioners."""
    configs: list[BasePreconditionerConfig] = [
        NonePreconditionerConfig(name="none"),
        JacobiPreconditionerConfig(name="jacobi"),
        ILUPreconditionerConfig(name="ilu"),
    ]
    if checkpoint_path is not None:
        configs.append(
            NeuralPreconditionerConfig(
                name="neural",
                checkpoint_path=Path(checkpoint_path),
                config_path=Path(config_path) if config_path else None,
                data_config_path=Path(data_config_path) if data_config_path else None,
            )
        )
    preconditioners, _ = build_preconditioners(A, configs)
    return (preconditioners,)
