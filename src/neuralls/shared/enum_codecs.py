"""Pure codecs for semantic enum arrays shared across layers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from neuralls.shared.types import RowKind


def encode_row_kind_array(kinds: Sequence[RowKind]) -> np.ndarray:
    """Encode RowKind enums to a compact uint8 array.

    Args:
        kinds: Sequence of RowKind values to encode.

    Returns:
        Compact uint8 numpy array of row kind codes.
    """
    return np.asarray(kinds, dtype=np.uint8)


def decode_row_kind_array(codes: np.ndarray) -> tuple[RowKind, ...]:
    """Decode a persisted uint8 row-kind array into enums.

    Args:
        codes: Compact uint8 numpy array of row kind codes.

    Returns:
        Tuple of RowKind enum values.
    """
    return tuple(RowKind(int(code)) for code in np.asarray(codes).reshape(-1))
