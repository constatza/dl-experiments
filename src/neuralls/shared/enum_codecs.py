"""Pure codecs for semantic enum arrays shared across layers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from neuralls.shared.types import RhsKind, TargetKind

_RHS_KIND_TO_CODE: dict[RhsKind, int] = {
    RhsKind.UNKNOWN: 0,
    RhsKind.NON_RESIDUAL: 1,
    RhsKind.RESIDUAL: 2,
    RhsKind.SEARCH_DIRECTION_PRODUCT: 3,
}
_CODE_TO_RHS_KIND: dict[int, RhsKind] = {value: key for key, value in _RHS_KIND_TO_CODE.items()}

_TARGET_KIND_TO_CODE: dict[TargetKind, int] = {
    TargetKind.UNKNOWN: 0,
    TargetKind.SOLUTION: 1,
    TargetKind.ITERATE: 2,
    TargetKind.ERROR: 3,
    TargetKind.SEARCH_DIRECTION: 4,
}
_CODE_TO_TARGET_KIND: dict[int, TargetKind] = {
    value: key for key, value in _TARGET_KIND_TO_CODE.items()
}


def encode_rhs_kind_array(kinds: Sequence[RhsKind]) -> np.ndarray:
    """Encode RHS semantic enums to a compact uint8 array."""
    return np.asarray([_RHS_KIND_TO_CODE[kind] for kind in kinds], dtype=np.uint8)


def decode_rhs_kind_array(codes: np.ndarray) -> tuple[RhsKind, ...]:
    """Decode a persisted uint8 RHS semantic array into enums."""
    return tuple(_CODE_TO_RHS_KIND[int(code)] for code in np.asarray(codes).reshape(-1))


def encode_target_kind_array(kinds: Sequence[TargetKind]) -> np.ndarray:
    """Encode target semantic enums to a compact uint8 array."""
    return np.asarray([_TARGET_KIND_TO_CODE[kind] for kind in kinds], dtype=np.uint8)


def decode_target_kind_array(codes: np.ndarray) -> tuple[TargetKind, ...]:
    """Decode a persisted uint8 target semantic array into enums."""
    return tuple(_CODE_TO_TARGET_KIND[int(code)] for code in np.asarray(codes).reshape(-1))
