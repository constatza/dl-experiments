"""Compatibility re-export for semantic enum codecs."""

from neuralls.shared.enum_codecs import (
    decode_rhs_kind_array,
    decode_target_kind_array,
    encode_rhs_kind_array,
    encode_target_kind_array,
)

__all__ = [
    "encode_rhs_kind_array",
    "decode_rhs_kind_array",
    "encode_target_kind_array",
    "decode_target_kind_array",
]
