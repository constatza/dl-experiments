"""Compatibility re-export for semantic enum codecs."""

from neuralls.shared.enum_codecs import (
    decode_row_kind_array,
    encode_row_kind_array,
)

__all__ = [
    "encode_row_kind_array",
    "decode_row_kind_array",
]
